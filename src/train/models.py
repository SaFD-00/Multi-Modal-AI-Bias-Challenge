"""다중 VLM(family) 레지스트리 + 학습/추론 공통 헬퍼.

지원 family (대회 규칙상 2026-06-01 이전 가중치 공개된 모델만 사용):
  - llava_ov   : LLaVA-OneVision-0.5B (파일럿·확정 모델). 기존 동작을 그대로 보존한다.
  - qwen2_5_vl : Qwen2.5-VL-7B-Instruct.
  - mimo_vl    : MiMo-VL-7B (Xiaomi, Qwen2.5-VL 아키텍처 기반).

모델별 분기 정보(model_id / 비전 freeze 패턴 / pixel 배칭 방식 / 프롬프트 렌더링 / LoRA 타깃)를
이 한 곳에 모은다. train/collator/merge/predict/eval_holdout은 family 키만으로 동작한다.

무거운 의존(torch/transformers)은 함수 내부에서 지연 import → .venv(torch 없음)에서도
레지스트리 데이터·family 매핑을 단위테스트할 수 있다.
"""
from __future__ import annotations

import json
from pathlib import Path

# LLM transformer 블록 표준 LoRA 타깃(세 모델 LLM 모듈 명명 동일).
_LLM_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULT_FAMILY = "llava_ov"

MODEL_REGISTRY = {
    "llava_ov": {
        "model_id": "llava-hf/llava-onevision-qwen2-0.5b-si-hf",
        "freeze": ["vision_tower", "multi_modal_projector"],
        "mm_batch": "llava_ov",   # pixel_values(anyres patches) + image_sizes
        "render": "llava_ov",     # 기존 하드코딩 chat 래핑(추론 정합 보존)
        "lora_targets": _LLM_LORA_TARGETS,
        "model_types": ("llava_onevision",),
        "trust_remote_code": False,
    },
    "qwen2_5_vl": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "freeze": ["visual"],
        "mm_batch": "qwen",       # pixel_values(flatten) + image_grid_thw
        "render": "chat_template",
        "lora_targets": _LLM_LORA_TARGETS,
        "model_types": ("qwen2_5_vl",),
        "trust_remote_code": False,
    },
    "mimo_vl": {
        "model_id": "XiaomiMiMo/MiMo-VL-7B-RL",
        "freeze": ["visual"],
        "mm_batch": "qwen",       # Qwen2.5-VL 아키텍처 기반 → 동일 배칭
        "render": "chat_template",
        "lora_targets": _LLM_LORA_TARGETS,
        # MiMo-VL은 Qwen2.5-VL 아키텍처 기반이라 config model_type이 qwen2_5_vl일 수 있다.
        "model_types": ("mimo_vl", "qwen2_5_vl"),
        "trust_remote_code": True,  # tf 4.57.6 네이티브 미지원 시 대비
    },
}


def get_spec(family: str) -> dict:
    if family not in MODEL_REGISTRY:
        raise ValueError(f"알 수 없는 family={family!r} (지원: {list(MODEL_REGISTRY)})")
    return MODEL_REGISTRY[family]


def model_id(family: str) -> str:
    return get_spec(family)["model_id"]


def lora_targets(family: str) -> list:
    return list(get_spec(family)["lora_targets"])


def render_mode(family: str) -> str:
    return get_spec(family)["render"]


def family_from_model_id(mid) -> str | None:
    """model_id 문자열에서 family 추론(merge 시 PeftConfig base 경로 매핑용)."""
    if mid in MODEL_REGISTRY:
        return mid
    mid_l = str(mid).lower()
    for fam, spec in MODEL_REGISTRY.items():
        if spec["model_id"].lower() in mid_l:
            return fam
    if "llava-onevision" in mid_l or "llava_onevision" in mid_l:
        return "llava_ov"
    if "mimo" in mid_l:
        return "mimo_vl"
    if "qwen2.5-vl" in mid_l or "qwen2_5_vl" in mid_l:
        return "qwen2_5_vl"
    return None


def detect_family(model_path) -> str | None:
    """체크포인트 디렉터리에서 family 자동감지 (추론용).

    우선순위: 경로 슬러그(outputs/{family}/...) → config.json model_type.
    경로 슬러그를 먼저 보는 이유: MiMo-VL은 model_type이 qwen2_5_vl일 수 있어 config만으로는
    qwen2_5_vl과 구분이 안 되므로(추론 동작은 동일), 경로에 적힌 family를 신뢰한다.
    """
    parts = Path(model_path).parts
    for fam in MODEL_REGISTRY:
        if fam in parts:
            return fam
    cfg = Path(model_path) / "config.json"
    if cfg.exists():
        try:
            mt = json.loads(cfg.read_text(encoding="utf-8")).get("model_type", "")
        except Exception:  # noqa: BLE001 - 깨진 config는 감지 실패로 처리
            mt = ""
        for fam, spec in MODEL_REGISTRY.items():
            if mt in spec["model_types"]:
                return fam
    return None


def freeze_vision(model, family: str) -> None:
    """family별 비전타워/프로젝터 freeze (LoRA/full 모드 모두 LLM만 학습)."""
    patterns = get_spec(family)["freeze"]
    for name, p in model.named_parameters():
        if any(pat in name for pat in patterns):
            p.requires_grad = False


def load_model_and_processor(family: str, bf16: bool = True):
    """family에 맞는 모델+processor 로드(범용 Auto 클래스) 후 비전 컴포넌트 freeze."""
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    spec = get_spec(family)
    trc = spec["trust_remote_code"]
    processor = AutoProcessor.from_pretrained(spec["model_id"], trust_remote_code=trc)
    model = AutoModelForImageTextToText.from_pretrained(
        spec["model_id"],
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        trust_remote_code=trc,
    )
    freeze_vision(model, family)
    return model, processor


def load_base_model(base_model_id: str, bf16: bool = True):
    """merge용 base 모델 로드(범용 Auto 클래스). family 추론으로 trust_remote_code 결정."""
    import torch
    from transformers import AutoModelForImageTextToText

    fam = family_from_model_id(base_model_id)
    trc = get_spec(fam)["trust_remote_code"] if fam else False
    return AutoModelForImageTextToText.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16 if bf16 else torch.float32,
        trust_remote_code=trc,
    )


def collate_multimodal(family: str, mm_list: list) -> dict:
    """family별 멀티모달 텐서 배칭.

    mm_list는 collator._encode_one이 보존한, processor의 원본(batch-of-1) 멀티모달 출력 dict 리스트.

    - llava_ov: pixel_values=(1, P, C, H, W). anyres라 이미지마다 patch 수(P)가 달라 patch 차원을
                배치 최대값으로 zero-pad 후 stack. 모델은 image_sizes로 실제 patch 수를 계산해
                패딩 patch를 무시한다. image_sizes=(1, 2) → dim0 concat.
    - qwen    : pixel_values=(총 patch, dim) flatten 형태 → dim0 concat. image_grid_thw=(1, 3) → concat.
                모델은 grid_thw로 샘플별 이미지 patch를 분할한다.
    """
    import torch

    mode = get_spec(family)["mm_batch"]
    if mode == "llava_ov":
        pvs = [m["pixel_values"][0] for m in mm_list]  # (1,P,...) → (P,...)
        max_patches = max(p.shape[0] for p in pvs)
        padded = []
        for p in pvs:
            if p.shape[0] < max_patches:
                pad = torch.zeros((max_patches - p.shape[0], *p.shape[1:]), dtype=p.dtype)
                p = torch.cat([p, pad], dim=0)
            padded.append(p)
        return {
            "pixel_values": torch.stack(padded),
            "image_sizes": torch.cat([m["image_sizes"] for m in mm_list], dim=0),
        }
    if mode == "qwen":
        return {
            "pixel_values": torch.cat([m["pixel_values"] for m in mm_list], dim=0),
            "image_grid_thw": torch.cat([m["image_grid_thw"] for m in mm_list], dim=0),
        }
    raise ValueError(f"알 수 없는 mm_batch 모드 (family={family}, mode={mode})")
