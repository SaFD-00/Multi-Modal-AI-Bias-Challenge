"""GPU 프로파일 기반 학습 런처.

`.env`의 `GPU_TYPE`/`GPU_COUNT`를 읽어 per-device batch·dtype·accum을 정하고(global batch 32 고정),
1 GPU는 직접, 2 GPU는 torchrun DDP로 `src.train.train`을 실행한다. 어떤 GPU/개수든
effective(global) batch는 32(=2^5)로 유지된다.

실행:
    python -m src.train.launch --no-wandb                         # .env 설정대로 학습 시작(llava_ov)
    python -m src.train.launch --model qwen2_5_vl --config configs/train_lora.yaml --no-wandb
    GPU_TYPE=RTX5090 GPU_COUNT=2 python -m src.train.launch        # 환경변수 직접 지정도 가능

`.env` 설정:
    GPU_TYPE=A100        # A100 | H100 | RTX5090
    GPU_COUNT=1          # 1 | 2
    GPU_DEVICES=1        # (선택) 물리 GPU 인덱스. 예 "1" 또는 "0,1". 미설정 시 0..GPU_COUNT-1

⚠️ per-device batch는 LLaVA-OV-0.5B 실측 기준이다. 7B(qwen2_5_vl/mimo_vl)는 --model로 family를
넘기면 메모리 수요에 맞춰 보수적 batch로 낮춘다(FAMILY_PER_DEVICE, 시작값이며 실측 튜닝 권장).
7B full FT는 80GB+ 필요(5090 32GB 불가) → LoRA 권장.
"""
from __future__ import annotations

import os
import subprocess
import sys

from ..common import load_config, project_root

GLOBAL_BATCH = 64  # effective batch 고정값 (= per_device × accum × gpu_count)

# GPU_TYPE → {별칭, per-device batch, bf16}.
#  80GB(A100/H100): fp32 batch 16 (실측 peak≈75GB/80GB). 5090(32GB): fp32 batch 4.
#  bf16=False(fp32)는 forward overflow→loss=nan 발산을 막는 정본 처방.
PROFILES = {
    "A100":    {"aliases": ("A100",), "per_device": 16, "bf16": False},
    "H100":    {"aliases": ("H100",), "per_device": 16, "bf16": False},
    "RTX5090": {"aliases": ("RTX5090", "5090"), "per_device": 4, "bf16": False},
}

# 7B 계열은 0.5B 대비 메모리 수요가 커 per-device batch를 프로파일별로 낮춘다(시작값, 실측 튜닝 권장).
# accum은 plan_launch가 global batch 64를 맞추도록 자동 재계산한다. 미등록 family는 PROFILES 기본값.
FAMILY_PER_DEVICE = {
    "qwen2_5_vl": {"A100": 2, "H100": 2, "RTX5090": 1},
    "mimo_vl":    {"A100": 2, "H100": 2, "RTX5090": 1},
}


def resolve_profile(gpu_type: str) -> tuple[str, dict]:
    """GPU_TYPE 문자열을 프로파일로 매핑. 영숫자만 비교해 표기 변종 흡수."""
    key = "".join(ch for ch in str(gpu_type).upper() if ch.isalnum())
    for name, spec in PROFILES.items():
        if any(a in key for a in spec["aliases"]):
            return name, spec
    raise ValueError(f"알 수 없는 GPU_TYPE={gpu_type!r} (지원: A100 / H100 / RTX5090)")


def plan_launch(gpu_type: str, gpu_count, family=None) -> dict:
    """GPU 종류·개수(+모델 family)로 학습 실행 계획 산출. global batch는 항상 GLOBAL_BATCH.

    family가 FAMILY_PER_DEVICE에 있으면(7B 계열) per-device batch를 프로파일별 보수값으로 낮추고,
    accum을 재계산해 global batch 64를 유지한다. 미지정/미등록 family는 PROFILES 기본값(0.5B).
    """
    name, spec = resolve_profile(gpu_type)
    gpu_count = int(gpu_count)
    if gpu_count not in (1, 2):
        raise ValueError(f"GPU_COUNT는 1 또는 2 (받음: {gpu_count})")
    per_dev = spec["per_device"]
    if family in FAMILY_PER_DEVICE:
        per_dev = FAMILY_PER_DEVICE[family].get(name, per_dev)
    accum = max(1, round(GLOBAL_BATCH / (per_dev * gpu_count)))
    return {
        "profile": name,
        "gpu_count": gpu_count,
        "per_device_batch": per_dev,
        "accum": accum,
        "bf16": spec["bf16"],
        "global_batch": per_dev * accum * gpu_count,
    }


def _parse_family(argv) -> str | None:
    """passthrough argv에서 --model {family} / --model={family} 추출(런처 batch 보정용)."""
    for i, a in enumerate(argv):
        if a == "--model" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--model="):
            return a.split("=", 1)[1]
    return None


def main() -> None:
    root = project_root()
    load_config()  # .env → os.environ (GPU_TYPE/GPU_COUNT/GPU_DEVICES, HF_TOKEN 로드)
    passthrough = sys.argv[1:]  # --model / --config / --no-wandb / --max-samples 그대로 전달
    family = _parse_family(passthrough)
    plan = plan_launch(os.environ.get("GPU_TYPE", "A100"), os.environ.get("GPU_COUNT", "1"), family=family)

    devices = os.environ.get("GPU_DEVICES") or ",".join(str(i) for i in range(plan["gpu_count"]))
    env = dict(os.environ)
    env.update({
        "TRAIN_PER_DEVICE_BATCH": str(plan["per_device_batch"]),
        "TRAIN_ACCUM": str(plan["accum"]),
        "TRAIN_BF16": "1" if plan["bf16"] else "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # fp32 단편화 완화
        "CUDA_VISIBLE_DEVICES": devices,
    })
    print(f"[launch] {plan['profile']}×{plan['gpu_count']} model={family or 'llava_ov'} "
          f"| per_device={plan['per_device_batch']} accum={plan['accum']} bf16={plan['bf16']} "
          f"global={plan['global_batch']} | CUDA_VISIBLE_DEVICES={devices}")

    if plan["gpu_count"] > 1:
        cmd = ["torchrun", "--standalone", "--nproc_per_node", str(plan["gpu_count"]),
               "-m", "src.train.train", *passthrough]
    else:
        cmd = [sys.executable, "-m", "src.train.train", *passthrough]
    print("[launch]", " ".join(cmd))
    raise SystemExit(subprocess.run(cmd, env=env, cwd=str(root)).returncode)


if __name__ == "__main__":
    main()
