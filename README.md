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
.venv/bin/python -m pytest tests/ -q  # 순수 함수 90개 GREEN (데이터 + 학습 + 추론 + 런처 + 레지스트리/경로)
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

## 모델 학습 (다중 VLM)

사용 가능 family(`src/train/models.py` `MODEL_REGISTRY`가 정본)를 transformers로 LoRA/full
fine-tuning한다. (LLaMA-Factory 미사용 → 직접 학습.) `--model {family}`로 모델을 선택하며,
모델별 분기(model_id·비전 freeze·pixel 배칭·프롬프트 렌더·LoRA 타깃)는 레지스트리에 모인다.

| family | model_id | 비고 |
|---|---|---|
| `llava_ov` | `llava-hf/llava-onevision-qwen2-0.5b-si-hf` | 파일럿·확정 모델(0.5B). 베이스라인 chat 래핑 보존 |
| `qwen2_5_vl` | `Qwen/Qwen2.5-VL-7B-Instruct` | 본 모델 후보(7B). processor chat template |
| `mimo_vl` | `XiaomiMiMo/MiMo-VL-7B-RL` | 본 모델 후보(7B, Qwen2.5-VL 아키텍처 기반) |

| 항목 | 값 |
|---|---|
| 방식 | LoRA (LLM에 adapter) → **merge 필수**, 또는 full (LLM 전체) → merge 불필요. 비전타워/projector는 항상 freeze |
| 출력 | `outputs/{family}/adapters/lora`(LoRA) · `outputs/{family}/merged/{lora,full}`(추론용) · `outputs/{family}/eval/` (자동 산출, `src/train/paths.py`) |
| 환경 | A100/H100 80GB 또는 RTX5090, 1~2 GPU — `.env`의 `GPU_TYPE`/`GPU_COUNT`로 선택. **7B full FT는 80GB+ 필요(5090 불가) → 7B는 LoRA 권장** |
| 데이터 | `train.csv` 시드 고정 split. OOD 검증은 `ood_axes` 축을 hold-out(leave-axis-out) |
| reason 합성 | 정답이 Unknown류면 "정보 부족", 특정 옵션이면 옵션 명시 (편향 회피 강화) |
| 모니터링 | **WANDB** (키 없으면 tensorboard 자동 폴백) |

```bash
# 학습 환경(.venv-train, RTX5090) — 아래 RTX5090 블록 참조. torch cu128 + transformers==4.57.6 핀.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 학습 — GPU 프로파일 런처(.env의 GPU_TYPE/GPU_COUNT를 읽어 batch/dtype/accum·torchrun 자동 구성)
.venv-train/bin/python -m src.train.launch --model qwen2_5_vl --config configs/train_lora.yaml --no-wandb  # 7B LoRA
.venv-train/bin/python -m src.train.launch --model llava_ov  --config configs/train_full.yaml  --no-wandb  # 0.5B full
.venv-train/bin/python -m src.train.merge --family qwen2_5_vl    # LoRA → outputs/qwen2_5_vl/merged/lora (full은 불필요)
# 추론·제출은 아래 "추론 및 제출"(src.predict) 참고
```

> **RTX5090 로컬 학습 venv** (데이터/추론 `.venv`엔 torch 없음 → 학습 전용 venv를 별도 구축, `.gitignore` 등록):
> ```bash
> uv venv --python 3.10 .venv-train
> uv pip install --python .venv-train/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cu128
> uv pip install --python .venv-train/bin/python "transformers==4.57.6" peft accelerate tensorboard pandas pillow pyyaml tqdm python-dotenv datasets
> .venv-train/bin/python -m src.train.launch --model qwen2_5_vl --config configs/train_lora.yaml --no-wandb   # .env GPU_DEVICES=1 → GPU1
> ```
> 실측: torch 2.11.0+cu128 / tf 4.57.6 / peft 0.19.1 / accel 1.13.0, RTX5090(sm_120) OK. WANDB 키 없으면 tensorboard 폴백(loss는 `outputs/{family}/.../runs/`).
> ⚠️ MiMo-VL은 tf 4.57.6 네이티브 미지원 시 `trust_remote_code`(레지스트리에 설정됨) 또는 버전 조정 필요 → 학습 머신에서 1차 검증.

> **LLaVA-OV Full FT 1차 결과 (2026-06-07, RTX5090×1, `configs/train_full.yaml`)**: `outputs/llava_ov/merged/full`(best=epoch1) — `src.eval_holdout` 실측
> **in-domain acc 0.9990(ambig 1.0000/disambig 0.9978), OOD(Religion·SO) acc 0.9545(ambig 0.9997/disambig 0.8207), 갭 +0.0445.**
> 핵심 편향지표(ambiguous=unknown 회수)는 미학습 OOD축에서도 0.9997. 약점은 OOD disambiguated 0.82(미학습 축 시각추론). 이는 0.5B 파일럿 성능이며 7B 본 모델로 개선을 노린다.

**GPU 프로파일 (`.env` 설정 → `src.train.launch`)** — 어떤 조합이든 **global(effective) batch는 64로 고정**:

| `GPU_TYPE` | `GPU_COUNT` | per-device × accum × gpu | dtype | 실행 |
|---|---|---|---|---|
| `A100`/`H100` (80GB) | 1 | 16 × 4 × 1 = 64 | fp32 | 직접 |
| `A100`/`H100` (80GB) | 2 | 16 × 2 × 2 = 64 | fp32 | torchrun DDP |
| `RTX5090` (32GB) | 1 | 4 × 16 × 1 = 64 | fp32 | 직접 |
| `RTX5090` (32GB) | 2 | 4 × 8 × 2 = 64 | fp32 | torchrun DDP |

- 위 표는 **0.5B(llava_ov)** 실측값이다. **7B(qwen2_5_vl/mimo_vl)는 `--model {family}`를 넘기면 per-device batch를 보수적으로 하향**(A100/H100=2, 5090=1; `launch.FAMILY_PER_DEVICE`)하고 accum을 재계산해 global 64를 유지한다(시작값, 실측 튜닝 권장).
- `.env`: `GPU_TYPE`(A100/H100/RTX5090), `GPU_COUNT`(1/2), `GPU_DEVICES`(선택, 예 `"1"`·`"0,1"`). 예시는 `.env.example`.
- 런처가 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`·`CUDA_VISIBLE_DEVICES`를 자동 설정한다.
- 학습 설정은 `configs/train.yaml`(공통 base, `model:` 기본 family) + `configs/train_lora.yaml`/`configs/train_full.yaml`(모드별 override) 구조. `--config`로 모드별 파일을, `--model`로 family를 받는다(`finetune_type: lora|full`). 출력경로는 family+모드로 자동 산출.
- 런처 없이 `python -m src.train.train --model {family} --config ...` 직접 실행하면 위 yaml 값 그대로 사용(하위호환).

> **대용량 경로**: `data/`·`outputs/`는 루트 볼륨 보호를 위해 `/data/seungwoo/Multi-Modal-AI-Bias-Challenge/`로 심링크되어 있다.
> **GPU 학습 사전 검증으로 적용된 코드 수정** — ① `collator.py`: `apply_chat_template`(텍스트 렌더) + `processor(text=, images=)` 2단계 분리(transformers 4.5x `images` 인자 중복 회피), ② `train.py`: `gradient_checkpointing` 시 `enable_input_require_grads()` 호출(grad 미전파 오류 방지).

> **프롬프트 정합 + family별 메시지 구조** — `src/train/prompt.py`가 학습·추론 공통 정본(정합 회귀를 `tests/test_train.py`가 검증). `llava_ov`는 베이스라인 `prompt_text`와 **문자열 단위 동일**한 단일 user turn을 유지하고, **chat_template 계열(qwen2_5_vl/mimo_vl)은 system(역할+규칙)/user(이미지+데이터) 2-turn으로 분리**한다.
> **편향 회피 프롬프트 A/B 토글** — `BIAS_PROMPT_V2=1`이면 편향 회피·불확실성 지시(균형형: 부족→unknown / 충분→특정 답, 9축 명시)를 삽입한다(chat_template은 system, llava_ov는 PRE 뒤). 기본 off=베이스라인 정합. ⚠️ 학습·병합·평가·추론에서 **동일하게** 설정해야 정합(불일치 시 성능 왜곡). A/B는 두 안의 산출물 경로가 같으니 순차 학습→평가→기록(또는 `mv` 백업). 스모크는 `--max-samples 64 --no-wandb`.

### OOD 검증셋 (leave-axis-out)

Public/Private **Shake-up**(운영진 자체제작 Private) 설계에서 핵심 위험은 IID 과적합이 아니라 학습 분포의
텍스트 shortcut 암기다. `configs/data.yaml`의 `ood_axes`(기본 `Religion`·`Sexual_orientation`, 약 9%)를 통째로
hold-out해 **"학습에서 안 본 편향 축"의 일반화**를 측정한다. `ood_axes`가 비면 기존 단일 IID val로 동작.

- **학습**(`train.py`): OOD 활성 시 `eval_dataset={"in":…, "ood":…}`로 `eval_in_loss`·`eval_ood_loss`를 각각
  로깅하고 **`eval_ood_loss` 기준 best 체크포인트**를 고른다 → IID eval_loss로 과적합 체크포인트를 고르는 함정 회피.
- **검증**: `python -m src.validate --ood` — train / in-domain-val / ood-val 3분할의 무결성(sample_id 겹침=0,
  OOD가 지정 축만 포함)과 축·극성 분포를 리포트.
- 정본은 `configs/data.yaml`의 `ood_axes` + `paths.metadata` (학습·검증 공유). 배경: `.claude/researchs/` epoch 분석 문서.
- **실제 accuracy 평가**(loss 아님): `python -m src.eval_holdout --model-family llava_ov --split both` —
  train.py와 동일 분할을 재현해 in/OOD의 정답률 + ambig(unknown 회수)/disambig별 정확도를 vLLM 추론으로 측정.
  full(merged/full)은 그대로, LoRA는 `--variant lora`(merged/lora, merge 선행). vLLM 신/구 버전 API 모두 호환(structured_outputs↔guided_decoding).

| 학습 모듈 | 역할 |
|---|---|
| `src/train/models.py` | 모델 family 레지스트리(model_id·freeze·pixel 배칭·렌더·LoRA 타깃) + 로드/배칭 헬퍼 |
| `src/train/paths.py` | family별 출력경로 산출(`outputs/{family}/{adapters/lora, merged/{lora,full}, eval}`) |
| `src/train/prompt.py` | 프롬프트 빌드(llava_ov 단일 turn / chat_template은 system·user 분리) + 편향회피 토글(`BIAS_PROMPT_V2`) + family별 추론 렌더 + reason/target JSON |
| `src/train/dataset.py` | train.csv 로드 + 결정적 split + leave-axis-out OOD 3분할 |
| `src/train/collator.py` | family 비의존 멀티모달 collator + assistant 토큰만 학습(-100 마스킹), mm 배칭은 레지스트리 위임 |
| `src/train/train.py` | `--model {family}` LoRA/full + HF Trainer + wandb (런처 주입 batch/dtype override 지원) |
| `src/train/launch.py` | `.env`(GPU_TYPE/GPU_COUNT)+family → batch/dtype/accum·torchrun 자동 구성, global batch 64 고정 |
| `src/train/merge.py` | `--family` LoRA → base 병합(추론용 HF 체크포인트, 범용 Auto 클래스) |
| `src/predict.py` | family 자동감지 vLLM 추론 → 제출 CSV(`sample_id,label`) |
| `src/eval_holdout.py` | family 자동감지 leave-axis-out accuracy(in/OOD, ambig/disambig) |

## 추론 및 제출

대회 1차 제출물(`outputs/{family}/eval/submission.csv`, **`sample_id,label`**)을 만든다. 프롬프트(family별
렌더)/이미지 전처리(img_size=224)는 `src/train/prompt.py`·`src/train/collator.py`를 재사용해 **학습과 동일**하다.
최종 답변은 LLM이 JSON(`{"reason","answer_id"}`)을 **생성**하고 거기서 `answer_id`만 파싱한다(룰 매핑 아님).

```bash
# 의존성은 requirements.txt로 통합 — ⚠️ vLLM은 학습 transformers 핀과 충돌 가능하니 추론은 별도 venv(.venv) 권장
pip install -r requirements.txt        # 또는 별도 venv에서: pip install vllm pydantic
python -m src.predict --model-family llava_ov        # → outputs/llava_ov/eval/submission.csv
python -m src.predict --model outputs/qwen2_5_vl/merged/lora   # 경로 직접 지정(family 자동감지)
```

- **family 자동감지**: `--model-family`(명시) 또는 `--model` 경로(`outputs/{family}/...`)·config에서 감지. full은 `merged/full`, LoRA는 `src.train.merge` 산출물(`merged/lora`)을 로드(vLLM 멀티모달 LoRA 직접 로드 제약).
- **오프라인**: `src.predict`가 `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE`를 강제한다(외부 API/통신 금지 규칙).
- **기준 평가환경(COMPETITION.md §6)**: RTX A6000 48GB, Python 3.10, CUDA 12.4, PyTorch 2.6.0, Ubuntu 20.04 / 추론 ≤0.5s/샘플(Test 8,500≈70분). 7B는 A6000 bf16 적재 가능하나 속도는 기준 환경에서 측정 필요.
- 스모크: `--max-samples 8 --out outputs/{family}/eval/_smoke.csv` 로 소량 확인.

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
