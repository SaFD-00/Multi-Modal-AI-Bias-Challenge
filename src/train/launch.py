"""GPU 프로파일 기반 학습 런처.

`.env`의 `GPU_TYPE`/`GPU_COUNT`를 읽어 per-device batch·dtype·accum을 정하고(global batch 32 고정),
1 GPU는 직접, 2 GPU는 torchrun DDP로 `src.train.train`을 실행한다. 어떤 GPU/개수든
effective(global) batch는 32(=2^5)로 유지된다.

실행:
    python -m src.train.launch --no-wandb        # .env 설정대로 학습 시작
    GPU_TYPE=RTX5090 GPU_COUNT=2 python -m src.train.launch   # 환경변수 직접 지정도 가능

`.env` 설정:
    GPU_TYPE=A100        # A100 | H100 | RTX5090
    GPU_COUNT=1          # 1 | 2
    GPU_DEVICES=1        # (선택) 물리 GPU 인덱스. 예 "1" 또는 "0,1". 미설정 시 0..GPU_COUNT-1
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


def resolve_profile(gpu_type: str) -> tuple[str, dict]:
    """GPU_TYPE 문자열을 프로파일로 매핑. 영숫자만 비교해 표기 변종 흡수."""
    key = "".join(ch for ch in str(gpu_type).upper() if ch.isalnum())
    for name, spec in PROFILES.items():
        if any(a in key for a in spec["aliases"]):
            return name, spec
    raise ValueError(f"알 수 없는 GPU_TYPE={gpu_type!r} (지원: A100 / H100 / RTX5090)")


def plan_launch(gpu_type: str, gpu_count) -> dict:
    """GPU 종류·개수로 학습 실행 계획 산출. global batch는 항상 GLOBAL_BATCH."""
    name, spec = resolve_profile(gpu_type)
    gpu_count = int(gpu_count)
    if gpu_count not in (1, 2):
        raise ValueError(f"GPU_COUNT는 1 또는 2 (받음: {gpu_count})")
    per_dev = spec["per_device"]
    accum = max(1, round(GLOBAL_BATCH / (per_dev * gpu_count)))
    return {
        "profile": name,
        "gpu_count": gpu_count,
        "per_device_batch": per_dev,
        "accum": accum,
        "bf16": spec["bf16"],
        "global_batch": per_dev * accum * gpu_count,
    }


def main() -> None:
    root = project_root()
    load_config()  # .env → os.environ (GPU_TYPE/GPU_COUNT/GPU_DEVICES, HF_TOKEN 로드)
    plan = plan_launch(os.environ.get("GPU_TYPE", "A100"), os.environ.get("GPU_COUNT", "1"))

    devices = os.environ.get("GPU_DEVICES") or ",".join(str(i) for i in range(plan["gpu_count"]))
    env = dict(os.environ)
    env.update({
        "TRAIN_PER_DEVICE_BATCH": str(plan["per_device_batch"]),
        "TRAIN_ACCUM": str(plan["accum"]),
        "TRAIN_BF16": "1" if plan["bf16"] else "0",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",  # fp32 단편화 완화
        "CUDA_VISIBLE_DEVICES": devices,
    })
    print(f"[launch] {plan['profile']}×{plan['gpu_count']} | per_device={plan['per_device_batch']} "
          f"accum={plan['accum']} bf16={plan['bf16']} global={plan['global_batch']} "
          f"| CUDA_VISIBLE_DEVICES={devices}")

    passthrough = sys.argv[1:]  # --config / --no-wandb / --max-samples 그대로 전달
    if plan["gpu_count"] > 1:
        cmd = ["torchrun", "--standalone", "--nproc_per_node", str(plan["gpu_count"]),
               "-m", "src.train.train", *passthrough]
    else:
        cmd = [sys.executable, "-m", "src.train.train", *passthrough]
    print("[launch]", " ".join(cmd))
    raise SystemExit(subprocess.run(cmd, env=env, cwd=str(root)).returncode)


if __name__ == "__main__":
    main()
