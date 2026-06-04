# Train 데이터 구축 파이프라인 (SKKU 멀티모달 AI Bias 챌린지)

DACON 236722용 학습 데이터 자체 구축 파이프라인. 이미지+context+question → 3지선다(label 0/1/2),
**편향 회피 + 정보 부족 시 "Unknown" 선택**이 핵심. 공식 train이 1건뿐이라 데이터 구축이 과제다.

## 전략

하이브리드: **SB-Bench MCQ 실사**(분포 정합 주력) + **BBQ 원본 텍스트**(9축·규모 보강).

| 소스 | 라이선스 | 역할 |
|---|---|---|
| SB-Bench (`ucf-crcv/SB-Bench`, `real` split) | **CC BY-NC 4.0** (비상업) | 실사 이미지 + MCQ |
| BBQ (`nyu-mll/BBQ`, GitHub JSONL) | CC-BY-4.0 | 9축×ambig/disambig×polarity 부족 셀 보강 |

> ⚠️ **SB-Bench는 CC BY-NC 4.0(비상업)**. 비상업 학술 대회 용도로만 사용하며, 전 샘플의
> 출처/라이선스를 `data/metadata.jsonl`에 기록한다(7/10 코드 검증 대비).

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
.venv/bin/python -m src.augment_bbq   # 부족 셀 BBQ 보강 (GitHub JSONL)
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
| 방식 | LoRA (비전타워/projector freeze, LLM에 adapter) → **merge 필수** |
| 환경 | H100 80GB × 1 (GPU1, `CUDA_VISIBLE_DEVICES=1`) |
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 단편화 OOM 방지
CUDA_VISIBLE_DEVICES=1 python -m src.train.train --config configs/train_lora.yaml --no-wandb   # 학습
python -m src.train.merge --adapter outputs/llava_ov_lora --out outputs/llava_ov_merged        # 병합
# 베이스라인 노트북 EngineArgs(model="outputs/llava_ov_merged") 로 추론
```

> **대용량 경로**: `data/`·`outputs/`는 루트 볼륨 보호를 위해 `/data/seungwoo/Multi-Modal-AI-Bias-Challenge/`로 심링크되어 있다.
> **GPU 학습 사전 검증으로 적용된 코드 수정** — ① `collator.py`: `apply_chat_template`(텍스트 렌더) + `processor(text=, images=)` 2단계 분리(transformers 4.5x `images` 인자 중복 회피), ② `train.py`: `gradient_checkpointing` 시 `enable_input_require_grads()` 호출(grad 미전파 오류 방지).

> 학습 입력 프롬프트는 베이스라인 추론 `prompt_text`와 **문자열 단위로 동일**하게 재현된다
> (`src/train/prompt.py`, 정합 회귀를 `tests/test_train.py`가 검증). 스모크는 `--max-samples 64 --no-wandb`.

### OOD 검증셋 (leave-axis-out)

Public/Private **Shake-up**(운영진 자체제작 Private) 설계에서 핵심 위험은 IID 과적합이 아니라 학습 분포의
텍스트 shortcut 암기다. `config.yaml`의 `ood_axes`(기본 `Religion`·`Sexual_orientation`, 약 9%)를 통째로
hold-out해 **"학습에서 안 본 편향 축"의 일반화**를 측정한다. `ood_axes`가 비면 기존 단일 IID val로 동작.

- **학습**(`train.py`): OOD 활성 시 `eval_dataset={"in":…, "ood":…}`로 `eval_in_loss`·`eval_ood_loss`를 각각
  로깅하고 **`eval_ood_loss` 기준 best 체크포인트**를 고른다 → IID eval_loss로 과적합 체크포인트를 고르는 함정 회피.
- **검증**: `python -m src.validate --ood` — train / in-domain-val / ood-val 3분할의 무결성(sample_id 겹침=0,
  OOD가 지정 축만 포함)과 축·극성 분포를 리포트.
- 정본은 `config.yaml`의 `ood_axes` + `paths.metadata` (학습·검증 공유). 배경: `.claude/researchs/` epoch 분석 문서.

| 학습 모듈 | 역할 |
|---|---|
| `src/train/prompt.py` | 베이스라인 동일 프롬프트 + reason 합성 + target JSON (추론 정합 단일 진실원) |
| `src/train/dataset.py` | train.csv 로드 + 결정적 split + leave-axis-out OOD 3분할 |
| `src/train/collator.py` | 멀티모달 collator + assistant 토큰만 학습(prompt/image 토큰 -100 마스킹) |
| `src/train/train.py` | LoRA + HF Trainer + wandb |
| `src/train/merge.py` | LoRA → base 병합(추론용 HF 체크포인트) |

## 설정 (`config.yaml`)

- `seed`: 재현성 시드(42)
- `safe_only`: `true`면 SB-Bench 제외, BBQ만으로 구축(라이선스 안전 모드)
- `target_per_cell`: 셀(9축×ambig/disambig×2polarity)당 상한 — 규모 제어 (현재 2400 → ~45k)
- `unknown_distribution`: `proportional`(test 관측 비례) | `uniform`
- `unknown_lexicon`: test 관측 Unknown 표현 10종(정본)

## 파이프라인 구조

| 단계 | 모듈 | 역할 |
|---|---|---|
| T1 | `src/map_sbbench.py` | SB-Bench → 중간 스키마 `mapped.jsonl`, 이미지 저장, ambig 휴리스틱 도출 |
| T2 | `src/augment_bbq.py` | BBQ로 부족 셀 보강, 이미지는 동일 axis SB-Bench 재사용 |
| T3 | `src/compose.py` ★ | Unknown 재다양화 + 위치 재셔플 + label 재매핑 + 비율 샘플링 |
| T4 | `src/metadata.py` | test 누수 제거(정규화 해시) + 출처/라이선스 메타 |
| T5 | `src/validate.py` | 스키마/분포/bias score/재현성 + 베이스라인 스모크 · `--ood`: OOD 3분할 무결성 |

중간 스키마(`mapped.jsonl`): `uid, source, license, axis, polarity, ambig, context, question,
options, label, unknown_idx, unknown_text, image_ref, norm_key, meta`.

## 검증 성공 기준

`validate` 전부 PASS · train.csv가 test.csv 스키마와 정합 · Unknown 10종이 test 비례로 재현 +
위치 0/1/2 균등 · metadata에 전 샘플 출처/라이선스(NC 플래그) 기록 · 베이스라인 로더로 프롬프트
생성 성공 · **test 누수 0건** · 동일 시드 → 동일 산출물 해시.
