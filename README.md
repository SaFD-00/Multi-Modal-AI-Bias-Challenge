# Train 데이터 구축 파이프라인 (SKKU 멀티모달 AI Bias 챌린지)

DACON 236722용 학습 데이터 자체 구축 파이프라인. 이미지+context+question → 3지선다(label 0/1/2),
**편향 회피 + 정보 부족 시 "Unknown" 선택**이 핵심. 공식 train이 1건뿐이라 데이터 구축이 과제다.

## 전략

하이브리드: **SB-Bench MCQ 실사**(분포 정합 주력) + **BBQ 원본 텍스트**(9축·규모 보강) +
**FairFace·MMBias 외부 이미지**(BBQ 행 이미지 다양화 + 라이선스 정화).

| 소스 | 라이선스 | 역할 |
|---|---|---|
| SB-Bench (`ucf-crcv/SB-Bench`, `real` split) | **CC BY-NC 4.0** (비상업) | 실사 이미지 + MCQ (주력) |
| BBQ (`nyu-mll/BBQ`, GitHub JSONL) | CC-BY-4.0 | 9축×ambig/disambig×polarity 부족 셀 보강 (텍스트) |
| FairFace (`HuggingFaceM4/FairFace`) | **CC-BY-4.0** | BBQ Age/Gender/Race/Intersectional 행 이미지 (균형 얼굴) |
| MMBias (`sepehrjng92/MMBias`, GitHub zip) | **MIT** | BBQ Religion/Sexual_orientation/Nationality/Disability 행 이미지 |

> **이미지 출처 분리**: BBQ 텍스트 행은 기존에 동일 axis SB-Bench(NC) 이미지를 재사용(14,578장
> 중복 큼)했으나, 이제 FairFace(CC-BY)·MMBias(MIT)를 **우선** 결합한다 → (1) 이미지 다양성↑
> (2) OOD 축(Religion/Sexual_orientation) 실이미지 확보 (3) BBQ 행을 CC-BY/MIT만으로 구성(NC 의존 제거).
> 외부에 없는 축(SES/Physical_appearance)만 SB-Bench로 폴백. `external_images.enabled=false`면 비활성.

> ⚠️ **SB-Bench는 CC BY-NC 4.0(비상업)**. 텍스트·이미지 출처/라이선스를 각각 `data/metadata.jsonl`에
> 분리 기록하며(`text_source/image_source`, `is_nc`는 둘 중 하나라도 NC면 True), 7/10 코드 검증에 대비한다.

## 사전 준비

```bash
# 1) 가상환경 + 의존성
uv venv --python 3.10 .venv         # 또는 python -m venv
uv pip install -r requirements.txt

# 2) 대회 데이터(open.zip)에서 test.csv 추출 (누수 제거 + Unknown 분포 측정용)
#    .claude/references/open.zip → data/raw/test/test.csv

# 3) SB-Bench HF 게이트
#    - https://huggingface.co/datasets/ucf-crcv/SB-Bench 에서 약관 동의
#    - https://huggingface.co/settings/tokens 에서 read 토큰 발급
#    - .env 에 HF_TOKEN=hf_... 설정
```

## 실행

```bash
.venv/bin/python -m src.map_sbbench   # SB-Bench real split 다운로드(~12.3GB) + 이미지 저장
.venv/bin/python -m pytest tests/ -q  # 순수 함수 55개 GREEN (데이터 34 + 학습 21)
.venv/bin/python -m src.augment_bbq   # 부족 셀 BBQ 보강 + FairFace/MMBias 이미지 결합(최초 1회 다운로드)
.venv/bin/python -m src.compose       # Unknown 재다양화/위치 균등 → train.csv
.venv/bin/python -m src.metadata      # test 누수 제거 + 출처/라이선스 기록
.venv/bin/python -m src.validate      # 스키마·분포·bias score·재현성·베이스라인 스모크
.venv/bin/python -m src.validate --ood  # OOD(leave-axis-out) 3분할 무결성·분포 검증
```

산출물: `data/processed/train/{train.csv, images/}`, `data/metadata.jsonl`.

> **Colab**: `multi-modal-ai-bias-challenge.ipynb` 는 위 `src/` 소스를 그대로 실행하는 노트북이다.
> Google Drive 마운트(또는 zip 업로드)로 프로젝트를 올린 뒤 `pip install -r requirements.txt` →
> `python -m src.*` 순으로 동작한다.

## 실행 결과 (`target_per_cell=2400`)

| 항목 | 값 |
|---|---|
| train.csv | **45,004행** |
| 출처 | BBQ 36,305 (CC-BY-4.0) + SB-Bench 8,699 (CC-BY-NC-4.0) |
| 실사 이미지 | 14,578장, 누락 0 |
| test 누수 제거 | 7,310건 (test가 SB-Bench/BBQ 파생 → 제거 필수) |
| label / Unknown 위치 | 각 0/1/2 균등, Unknown 10종 전부(test 비례) |
| 재현성 | 동일 시드 → 동일 train.csv 해시 |

> SB-Bench `real` split은 **전부 ambiguous**(정답=항상 Unknown). disambiguated 예시는 BBQ가 전담하므로
> 하이브리드가 필수다. 규모를 더 키우면 여력이 남은 Intersectional/Race/Gender/SES 축에 집중된다.

## 모델 학습 (LoRA)

베이스라인과 **동일한** `llava-hf/llava-onevision-qwen2-0.5b-si-hf`를 transformers로 LoRA
fine-tuning한다. (LLaMA-Factory는 `llava_onevision` 템플릿/플러그인이 없어 사용 불가 → 직접 학습.)
학습 후 adapter를 base에 **merge**하면 베이스라인 추론 노트북이 모델 경로만 바꿔 그대로 로드한다.

| 항목 | 값 |
|---|---|
| 모델 | `llava-onevision-qwen2-0.5b-si-hf` (베이스라인 고정, 추론 호환) |
| 방식 | LoRA (LLM에 adapter) → **merge 필수**, 또는 full (LLM 전체) → merge 불필요. 비전타워/projector는 항상 freeze |
| 환경 | A100/H100 80GB 또는 RTX5090, 1~2 GPU — `.env`의 `GPU_TYPE`/`GPU_COUNT`로 선택 |
| 데이터 | `train.csv` 시드 고정 split. OOD 검증은 `ood_axes` 축을 hold-out(leave-axis-out) |
| reason 합성 | 정답이 Unknown류면 "정보 부족", 특정 옵션이면 옵션 명시 (편향 회피 강화) |
| 모니터링 | **WANDB** (키 없으면 tensorboard 자동 폴백) |

```bash
# H100 환경 (conda env: /data/seungwoo/Multi-Modal-AI-Bias-Challenge/.conda-env, python 3.10)
#   torch/torchvision: cu128 휠, transformers==4.57.6 핀(상위 5.x는 processor API 비호환)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt          # 학습 의존성 포함(torch/transformers/peft/accelerate/wandb)
pip install "transformers==4.57.6"
# (선택) wandb: .env 에 WANDB_API_KEY=... 추가 — 없으면 tensorboard로 폴백

# 학습 — GPU 프로파일 런처(.env의 GPU_TYPE/GPU_COUNT를 읽어 batch/dtype/accum·torchrun 자동 구성)
python -m src.train.launch --no-wandb                                # ← 권장. LoRA, global batch 64 유지
python -m src.train.launch --config configs/train_full.yaml --no-wandb   # full finetuning(LLM)
python -m src.train.merge --adapter outputs/llava_ov_lora --out outputs/llava_ov_merged   # LoRA만 병합(full은 불필요)
# 추론·제출은 아래 "추론 및 제출"(src.predict) 참고
```

**GPU 프로파일 (`.env` 설정 → `src.train.launch`)** — 어떤 조합이든 **global(effective) batch는 64로 고정**:

| `GPU_TYPE` | `GPU_COUNT` | per-device × accum × gpu | dtype | 실행 |
|---|---|---|---|---|
| `A100`/`H100` (80GB) | 1 | 16 × 4 × 1 = 64 | fp32 | 직접 |
| `A100`/`H100` (80GB) | 2 | 16 × 2 × 2 = 64 | fp32 | torchrun DDP |
| `RTX5090` (32GB) | 1 | 4 × 16 × 1 = 64 | fp32 | 직접 |
| `RTX5090` (32GB) | 2 | 4 × 8 × 2 = 64 | fp32 | torchrun DDP |

- `.env`: `GPU_TYPE`(A100/H100/RTX5090), `GPU_COUNT`(1/2), `GPU_DEVICES`(선택, 예 `"1"`·`"0,1"`). 예시는 `.env.example`.
- 런처가 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`·`CUDA_VISIBLE_DEVICES`를 자동 설정하고, 80GB는 fp32 batch 16(실측 peak≈75GB), 5090(32GB)은 fp32 batch 4로 OOM을 회피한다.
- 학습 설정은 `configs/train.yaml`(공통 base) + `configs/train_lora.yaml`/`configs/train_full.yaml`(모드별 override) 구조. `--config`로 모드별 파일을 받아 공통 위에 덮어쓴다(`finetune_type: lora|full`).
- 런처 없이 `python -m src.train.train --config ...` 직접 실행하면 위 yaml 값 그대로 사용(하위호환).

> **대용량 경로**: `data/`·`outputs/`는 루트 볼륨 보호를 위해 `/data/seungwoo/Multi-Modal-AI-Bias-Challenge/`로 심링크되어 있다.
> **GPU 학습 사전 검증으로 적용된 코드 수정** — ① `collator.py`: `apply_chat_template`(텍스트 렌더) + `processor(text=, images=)` 2단계 분리(transformers 4.5x `images` 인자 중복 회피), ② `train.py`: `gradient_checkpointing` 시 `enable_input_require_grads()` 호출(grad 미전파 오류 방지).

> 학습 입력 프롬프트는 베이스라인 추론 `prompt_text`와 **문자열 단위로 동일**하게 재현된다
> (`src/train/prompt.py`, 정합 회귀를 `tests/test_train.py`가 검증). 스모크는 `--max-samples 64 --no-wandb`.

### OOD 검증셋 (leave-axis-out)

Public/Private **Shake-up**(운영진 자체제작 Private) 설계에서 핵심 위험은 IID 과적합이 아니라 학습 분포의
텍스트 shortcut 암기다. `configs/data.yaml`의 `ood_axes`(기본 `Religion`·`Sexual_orientation`, 약 9%)를 통째로
hold-out해 **"학습에서 안 본 편향 축"의 일반화**를 측정한다. `ood_axes`가 비면 기존 단일 IID val로 동작.

- **학습**(`train.py`): OOD 활성 시 `eval_dataset={"in":…, "ood":…}`로 `eval_in_loss`·`eval_ood_loss`를 각각
  로깅하고 **`eval_ood_loss` 기준 best 체크포인트**를 고른다 → IID eval_loss로 과적합 체크포인트를 고르는 함정 회피.
- **검증**: `python -m src.validate --ood` — train / in-domain-val / ood-val 3분할의 무결성(sample_id 겹침=0,
  OOD가 지정 축만 포함)과 축·극성 분포를 리포트.
- 정본은 `configs/data.yaml`의 `ood_axes` + `paths.metadata` (학습·검증 공유). 배경: `.claude/researchs/` epoch 분석 문서.

| 학습 모듈 | 역할 |
|---|---|
| `src/train/prompt.py` | 베이스라인 동일 프롬프트 + reason 합성 + target JSON (추론 정합 단일 진실원) |
| `src/train/dataset.py` | train.csv 로드 + 결정적 split + leave-axis-out OOD 3분할 |
| `src/train/collator.py` | 멀티모달 collator + assistant 토큰만 학습(prompt/image 토큰 -100 마스킹) |
| `src/train/train.py` | LoRA/full + HF Trainer + wandb (런처 주입 batch/dtype override 지원) |
| `src/train/launch.py` | `.env`(GPU_TYPE/GPU_COUNT) → batch/dtype/accum·torchrun 자동 구성, global batch 64 고정 |
| `src/train/merge.py` | LoRA → base 병합(추론용 HF 체크포인트) |
| `src/predict.py` | 병합모델 vLLM 추론 → 제출 CSV(`sample_id,label`) |

## 추론 및 제출

대회 1차 제출물(`output/submission.csv`, **`sample_id,label`**)을 만든다. 프롬프트/이미지
전처리(img_size=224)는 `src/train/prompt.py`·`src/train/collator.py`를 재사용해 **학습과 동일**하다.
최종 답변은 LLM이 JSON(`{"reason","answer_id"}`)을 **생성**하고 거기서 `answer_id`만 파싱한다(룰 매핑 아님).

```bash
# 의존성은 requirements.txt로 통합 — ⚠️ vLLM은 학습 transformers 핀과 충돌 가능하니 추론은 별도 venv 권장
pip install -r requirements.txt        # 또는 별도 venv에서: pip install vllm pydantic
python -m src.predict --model outputs/llava_ov_merged \
    --test-csv data/raw/test/test.csv --images-dir data/raw/test \
    --out output/submission.csv --img-size 224
```

- **병합 필수**: vLLM은 멀티모달 LoRA 직접 로드가 어려워 `src.train.merge` 산출물(`outputs/llava_ov_merged`)을 로드한다.
- **오프라인**: `src.predict`가 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`를 강제한다(외부 API/통신 금지 규칙).
- **기준 평가환경**: RTX A6000 48GB, Python 3.10, CUDA 12.4, PyTorch 2.6.0 / 추론 ≤0.5s/샘플.
- 스모크: `--max-samples 8 --out output/_smoke.csv` 로 소량 확인.

## 설정 (`configs/data.yaml`)

- `seed`: 재현성 시드(42)
- `safe_only`: `true`면 SB-Bench 제외, BBQ만으로 구축(라이선스 안전 모드)
- `external_images`: BBQ 행에 FairFace(CC-BY)·MMBias(MIT) 이미지 결합 — `enabled`, `fairface_max`(저장 얼굴 수)
- `target_per_cell`: 셀(9축×ambig/disambig×2polarity)당 상한 — 규모 제어 (현재 2400 → ~45k)
- `unknown_distribution`: `proportional`(test 관측 비례) | `uniform`
- `unknown_lexicon`: test 관측 Unknown 표현 10종(정본)

## 파이프라인 구조

| 단계 | 모듈 | 역할 |
|---|---|---|
| T1 | `src/map_sbbench.py` | SB-Bench → 중간 스키마 `mapped.jsonl`, 이미지 저장, ambig 휴리스틱 도출 |
| T2 | `src/augment_bbq.py` | BBQ로 부족 셀 보강 + 외부 이미지 풀(FairFace/MMBias) 우선 결합, SB-Bench 폴백 |
| T2' | `src/external_images.py` | FairFace(CC-BY)·MMBias(MIT) → 축별 이미지 풀 `{axis: [(ref, source, license)]}` |
| T3 | `src/compose.py` ★ | Unknown 재다양화 + 위치 재셔플 + label 재매핑 + 비율 샘플링 |
| T4 | `src/metadata.py` | test 누수 제거(정규화 해시) + 텍스트·이미지 출처/라이선스 분리 메타 |
| T5 | `src/validate.py` | 스키마/분포/bias score/재현성 + 베이스라인 스모크 · `--ood`: OOD 3분할 무결성 |

중간 스키마(`mapped.jsonl`): `uid, source, license, axis, polarity, ambig, context, question,
options, label, unknown_idx, unknown_text, image_ref, image_source, image_license, norm_key, meta`.

## 검증 성공 기준

`validate` 전부 PASS · train.csv가 test.csv 스키마와 정합 · Unknown 10종이 test 비례로 재현 +
위치 0/1/2 균등 · metadata에 전 샘플 출처/라이선스(NC 플래그) 기록 · 베이스라인 로더로 프롬프트
생성 성공 · **test 누수 0건** · 동일 시드 → 동일 산출물 해시.
