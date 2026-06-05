"""GPU 프로파일 런처 순수 함수 검증 (GPU/subprocess 불필요)."""
import pytest

from src.train.launch import GLOBAL_BATCH, plan_launch, resolve_profile


def test_resolve_profile_aliases():
    assert resolve_profile("A100")[0] == "A100"
    assert resolve_profile("H100")[0] == "H100"
    assert resolve_profile("RTX5090")[0] == "RTX5090"
    # 표기 변종 흡수
    assert resolve_profile("A100/H100 80GB")[0] == "A100"   # 첫 매칭(A100)
    assert resolve_profile("nvidia-h100-80gb")[0] == "H100"
    assert resolve_profile("5090")[0] == "RTX5090"
    assert resolve_profile("rtx 5090")[0] == "RTX5090"


def test_resolve_profile_unknown_raises():
    with pytest.raises(ValueError):
        resolve_profile("V100")
    with pytest.raises(ValueError):
        resolve_profile("")


def test_plan_keeps_global_batch():
    # 모든 (GPU, 개수) 조합에서 effective batch는 GLOBAL_BATCH(=64)로 고정
    for gpu in ("A100", "H100", "RTX5090"):
        for n in (1, 2):
            plan = plan_launch(gpu, n)
            assert plan["global_batch"] == GLOBAL_BATCH == 64
            assert plan["per_device_batch"] * plan["accum"] * plan["gpu_count"] == 64


def test_plan_80gb_profile():
    p1 = plan_launch("A100", 1)
    assert (p1["per_device_batch"], p1["accum"], p1["bf16"]) == (16, 4, False)  # 16×4 = 64
    p2 = plan_launch("H100", 2)
    assert (p2["per_device_batch"], p2["accum"]) == (16, 2)  # 16×2×2 = 64


def test_plan_5090_profile():
    p1 = plan_launch("RTX5090", 1)
    assert (p1["per_device_batch"], p1["accum"], p1["bf16"]) == (4, 16, False)  # 4×16 = 64
    p2 = plan_launch("RTX5090", 2)
    assert (p2["per_device_batch"], p2["accum"]) == (4, 8)  # 4×8×2 = 64


def test_plan_rejects_bad_gpu_count():
    for bad in (0, 3, 4):
        with pytest.raises(ValueError):
            plan_launch("A100", bad)
    # 문자열 개수도 허용(.env는 문자열)
    assert plan_launch("A100", "2")["gpu_count"] == 2
