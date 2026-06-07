# AGENTS.md

AI 코딩 에이전트를 위한 작업 지침. 이 프로젝트는 **2026 성균관대학교 멀티모달 AI Bias 챌린지**(DACON 236722) 참가용입니다. (Claude Code도 이 파일을 정본으로 사용 — `CLAUDE.md`는 이 문서를 가리킵니다.)

## 현재 상태

**(1) Train 데이터 구축 파이프라인 + (2) LLaVA-OneVision LoRA 학습 스크립트 구현 완료.**
하이브리드(SB-Bench 실사 + BBQ 텍스트 + FairFace·MMBias 외부 이미지)로 `train.csv`와 실사 이미지를
산출하고, 이를 베이스라인과 동일한 `llava-onevision-qwen2-0.5b-si-hf`로 LoRA fine-tuning한다. 상세는 `README.md` 참조.

- 데이터 코드: `src/{common,map_sbbench,augment_bbq,external_images,compose,metadata,validate}.py`
- 학습 코드: `src/train/{prompt,dataset,collator,train,merge,launch}.py`, 설정 `configs/train.yaml`(공통) + `configs/train_lora.yaml`/`configs/train_full.yaml`(모드별, `finetune_type: lora|full`)
  - **GPU 프로파일 런처**(`src/train/launch.py`): `.env`의 `GPU_TYPE`(A100/H100/RTX5090)·`GPU_COUNT`(1/2)를 읽어 per-device batch·dtype(fp32)·accum을 정하고(**global batch 64 고정**) 1 GPU는 직접, 2 GPU는 torchrun DDP로 실행. 권장 진입점 `python -m src.train.launch --no-wandb`. 80GB=fp32 batch16×accum4(peak≈75GB), 5090(32GB)=fp32 batch4×accum16. `.env` 없이 `src.train.train` 직접 실행 시 yaml 값 그대로(하위호환).
- 추론 코드: `src/predict.py`(병합모델 vLLM 추론 → 제출 CSV)
- 테스트 `tests/`(74개: 데이터+학습+추론+런처)
- 설정: `configs/data.yaml`(데이터), 의존성 `requirements.txt`(데이터+학습+추론 통합; torch는 cu128 휠 별도 설치, transformers==4.57.6 핀). ⚠️ vLLM(추론)은 transformers/torch를 자체 고정해 학습 핀과 충돌 가능 → 추론은 별도 venv 권장.
- Colab: `run_pipeline_colab.ipynb`(기존 `src/` 실행)
- 추론·제출: `python -m src.predict --model outputs/llava_ov_merged` → `output/submission.csv`(`sample_id,label`). 프롬프트/전처리는 학습 모듈 재사용(정합), 오프라인 강제. vLLM이라 merge 선행 필수.

데이터(`open.zip`)는 DACON에서, SB-Bench는 HF 게이트(약관 동의 + read 토큰, `.env`의 `HF_TOKEN`)에서
별도 다운로드하며 리포지토리에 포함되지 않습니다.

## 명령

```bash
# 데이터 구축 (GPU 불필요)
uv venv --python 3.10 .venv && uv pip install -r requirements.txt   # 환경 구성
.venv/bin/python -m pytest tests/ -q                                # 테스트 (55개)
# 파이프라인: map_sbbench → augment_bbq → compose → metadata → validate (README 참조)
.venv/bin/python -m src.validate                                    # 최종 검증
.venv/bin/python -m src.validate --ood                              # OOD(leave-axis-out) 3분할 검증

# 모델 학습 (H100 80GB x 1, GPU1) — conda env: /data/seungwoo/Multi-Modal-AI-Bias-Challenge/.conda-env
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt && pip install "transformers==4.57.6"   # 5.x는 processor API 비호환
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
CUDA_VISIBLE_DEVICES=1 python -m src.train.train --config configs/train_lora.yaml --no-wandb   # LoRA 학습
# full finetuning(LLM): --config configs/train_full.yaml (merge 불필요, output_dir이 완결 모델)
python -m src.train.merge --adapter outputs/llava_ov_lora --out outputs/llava_ov_merged        # LoRA 추론용 병합
```

> 데이터 구축은 GPU 불필요. 학습은 H100(GPU1) conda env에서 실행. `requirements.txt`가 학습 의존성까지 포함(별도 `requirements-train.txt` 없음). `data/`·`outputs/`는 `/data`로 심링크. 하이퍼파라미터는 H100 80GB 실측 기준 micro-batch 16/accum 2(effective 32 = 2^5)/lr 1e-4/warmup 0.1/max_grad_norm 1.0 (처리량 위해 micro-batch 2배↑ — ⚠️ fp32에서 micro-batch 16은 OOM 위험 큼, OOM 시 8/accum 4로 환원).

> **NaN 발산 주의** — lr 2e-4 + warmup 0.03(짧음)로 돌린 첫 run은 warmup peak 도달(step~6)에서 `grad_norm=nan` 발산 → 이후 `loss=0.0`/`eval_loss=nan` 고착(weight NaN 오염, checkpoint 사용 불가). lr을 1e-4로 낮추고 warmup_ratio 0.1로 늘려 해결. NaN은 max_grad_norm으로 못 막으므로(`nan>1.0`=False) lr/warmup이 1차 방어선.

> **OOD 검증(leave-axis-out)** — Public/Private Shake-up 위험(텍스트 shortcut 암기)에 대응해 `configs/data.yaml`의 `ood_axes`(기본 Religion·Sexual_orientation)를 통째로 hold-out한다. 학습은 `eval_in_loss`/`eval_ood_loss`를 각각 로깅하고 **`eval_ood_loss` 기준 best 체크포인트**를 고른다(IID eval_loss 함정 회피). `src.validate --ood`로 3분할 무결성 검증. 정본은 `configs/data.yaml`의 `ood_axes` + `paths.metadata`(학습·검증 공유), `ood_axes: []`면 기존 단일 IID val.

> **외부 이미지 결합(`src/external_images.py`)** — BBQ 텍스트 행에 동일 axis SB-Bench(NC) 이미지를 재사용하던 것을 **FairFace(CC-BY, age/gender/race/intersectional) + MMBias(MIT, religion/sexual_orientation/nationality/disability)** 우선 결합으로 대체. (1) 이미지 다양성↑(14,578장 중복 → +FairFace/MMBias) (2) OOD 축 실이미지 확보 (3) BBQ 행 NC 의존 제거. `metadata.jsonl`은 `text_source`/`image_source`를 분리 기록하고 `is_nc`는 둘 중 하나라도 NC면 True. `configs/data.yaml`의 `external_images.enabled=false`면 기존 SB-Bench 재사용으로 회귀. ⚠️ FairFace는 단일 얼굴이라 2인 구도(disambiguated) 시각 부재는 한계(라이선스 안전과의 trade-off).

## 대회 개요

이미지-텍스트 기반 **질의응답(VQA)** 모델 개발. 핵심은 두 가지입니다:

1. **3지선다 분류** — 이미지 + 맥락(context) + 질문(question)을 보고 3개 선택지 중 정답 인덱스(0/1/2) 예측.
2. **편향 회피 + 불확실성 인식** — 성별·인종 등 사회적 단서로 편향된 판단을 내리지 말 것. 정보가 부족할 때는 "알 수 없음"류 선택지를 고르는 능력이 평가 대상.

평가 지표: **Accuracy**.

### 일정 (모두 2026년)
- 6/1 대회 시작 · 6/22 팀 병합 마감 · 6/29 대회 종료
- 7/2 2차 평가 제출 · 7/10 코드 검증 · 7/14 시상

## 데이터 스키마

`open.zip` 구조:
```
train/images/ + train.csv     # 제공 예시 1건뿐 (참가자가 직접 학습 데이터 구축)
test/images/  + test.csv       # 평가 대상 8,500건
sample_submission.csv
```

| 파일 | 컬럼 |
|------|------|
| `train.csv` | sample_id, image_path, context, question, answers, **label** (정답 인덱스 0/1/2) |
| `test.csv` | sample_id, image_path, context, question, answers |
| `sample_submission.csv` | sample_id, label |

- `answers`: 3개 선택지가 **JSON 문자열**로 인코딩됨 → 파싱 필요.
- `image_path`: 이미지 파일 상대 경로.

## 도메인 배경: 이 대회는 BBQ의 멀티모달 변형이다

과제 설계를 이해하려면 **BBQ(Bias Benchmark for QA)** 와 그 비전 확장 **BBQ-V(BBQ-Vision)** 를 알아야 합니다. 대회의 3지선다·label 구조, "편향 회피 + 불확실성 인식" 목표가 BBQ 방법론과 동일합니다.

### BBQ의 핵심 메커니즘
- **9개 사회 축**: 나이, 장애, 성정체성, 국적, 외모, 인종/민족, 종교, 사회경제적 지위, 성적지향 (+ 교차 축). 이 보호속성에 근거한 추측이 곧 **편향 오류**.
- **context 정보량(informativeness)으로 두 종류 문항 구성**:
  - **모호(ambiguous) context** — 답을 특정할 단서가 없음 → **정답은 "Unknown/알 수 없음" 선택지**. 여기서 특정 집단을 고르면 편향.
  - **명확(disambiguated) context** — 단서가 충분 → 사실에 근거한 특정 답이 정답.
- **질문 극성(polarity)**: 부정적/비부정적 프레이밍을 쌍으로 두어, 모델이 고정관념 방향으로 답을 트는지 측정.
- **핵심 발견**: 이미지를 추가하면 텍스트 단독 대비 공정성 점수가 크게 하락(일부 모델 20%+ 정확도 저하). 즉 **비전 입력이 편향을 증폭**시키므로, 이미지의 사람 외형(성별·인종 등)에 답이 끌려가지 않게 설계해야 함.

### 작업에 주는 함의
- `answers` 3개 선택지 중 하나는 거의 항상 **"unknown" 류**다. 파싱 후 이 선택지를 식별·우대하는 로직을 명시적으로 다룰 것.
- 모호 context일수록 unknown이 정답일 확률이 높다 — context 정보량 추정이 유효한 피처/프롬프트 신호.
- 단순 VQA 정확도 최적화는 함정이다. 평가셋은 "외형으로 추측하면 틀리도록" 설계됨.

## 핵심 제약 (작업 시 반드시 인지)

- **학습 데이터는 자체 구축**: 공식 train은 포맷 이해용 1건뿐. 데이터 수집·생성·라벨링 파이프라인이 프로젝트의 핵심 작업이다. 수집·생성한 데이터의 출처·라이선스를 `data/` 메타파일에 기록할 것 — 7/10 코드 검증에서 문제될 수 있음.
- **편향이 곧 패널티**: 단순 정확도 최적화가 아니라, "사회적 단서로 답을 추측하면 틀리도록" 설계된 평가셋. 모델/프롬프트가 보호속성에 근거해 답을 고르지 않도록, "정보 부족 시 unknown 선택" 동작을 명시적으로 다뤄야 한다.
- **재현성**: 7/10 코드 검증 통과를 위해 시드 고정, 추론 스크립트 단독 실행 가능성, 의존성 명세를 유지할 것.

## 에이전트 작업 규칙

1. **재현성 우선**: 모든 학습/추론 스크립트는 시드를 고정하고 단독 실행 가능해야 한다. 추론 스크립트는 `test.csv` + 이미지 → `submission.csv`만으로 동작해야 함.
2. **제출 포맷 엄수**: `sample_submission.csv`와 동일하게 `sample_id, label`(int 0/1/2). `answers`는 JSON 문자열이므로 인덱스 매핑 시 파싱 순서를 어긋내지 말 것.
3. **편향 평가를 자체 검증에 포함**: validation에서 전체 정확도뿐 아니라 모호/명확 context별 정확도, unknown 회수율(recall)을 분리해 측정할 것. 전체 정확도만 보면 편향 회귀를 놓친다.
4. **모델 선택**: 베이스라인과 동일한 `llava-onevision-qwen2-0.5b-si-hf`를 LoRA로 학습한다(추론 노트북 호환). LLaMA-Factory는 `llava_onevision` 미지원이라 transformers로 직접 학습한다. 베이스 모델 교체 등 큰 변경 전에는 근거·대안을 먼저 제시할 것.
5. **변경은 최소·외과적으로**: 요청 범위 밖 리팩터링 금지. 실험 코드는 재현 가능하게 분리.
6. **언어**: 사용자와의 소통·주석은 한국어, 코드 식별자는 영어.

## 산출물 경로

워크플로 산출물은 `.claude/`(plans/, researchs/, analysis/, reflexions/, state/), 대회 참고자료는 `.claude/reference/`.

---

참고: [AGENTS.md 표준](https://agents.md/) · [BBQ 논문](https://arxiv.org/abs/2110.08193) · [BBQ-V (arXiv:2502.08779)](https://arxiv.org/html/2502.08779v3)
