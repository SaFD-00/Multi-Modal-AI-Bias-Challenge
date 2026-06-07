"""모델 family별 산출물 경로 헬퍼.

출력 구조(프로젝트 루트 기준):
    outputs/{family}/adapters/lora/   # LoRA adapter (lora 모드 학습 output_dir)
    outputs/{family}/merged/lora/     # LoRA를 base에 병합한 추론용 완결 모델(merge 산출)
    outputs/{family}/merged/full/     # full 모드 학습 산출물(완결 모델, 병합 불필요)
    outputs/{family}/eval/            # eval 결과 + submission.csv

family: llava_ov / mimo_vl / qwen2_5_vl (src.train.models.MODEL_REGISTRY 키).
경로 문자열만 다루며 torch/transformers 의존이 없어 .venv(데이터/추론)에서도 import 가능.
"""
from __future__ import annotations

from pathlib import Path

from ..common import project_root


def outputs_root() -> Path:
    return project_root() / "outputs"


def family_dir(family: str) -> Path:
    return outputs_root() / family


def adapter_dir(family: str) -> Path:
    """LoRA adapter 디렉터리 (lora 모드 train output_dir)."""
    return family_dir(family) / "adapters" / "lora"


def merged_dir(family: str, variant: str) -> Path:
    """추론용 완결 모델 디렉터리.

    variant: 'full'(full 모드 학습 산출물) | 'lora'(LoRA 병합 산출물).
    """
    if variant not in ("lora", "full"):
        raise ValueError(f"variant는 'lora' 또는 'full' (받음: {variant!r})")
    return family_dir(family) / "merged" / variant


def eval_dir(family: str) -> Path:
    return family_dir(family) / "eval"


def submission_path(family: str) -> Path:
    return eval_dir(family) / "submission.csv"
