"""모델 레지스트리·family 매핑·출력경로 헬퍼 단위테스트 (모델 비의존, .venv에서 실행).

torch/transformers를 import하지 않는 순수 로직만 검증한다(레지스트리 데이터, family 추론,
경로 계산). 실제 모델 로드·collator 인코딩은 GPU 환경 스모크에서 확인.
"""
import pytest

from src.train import models as M
from src.train import paths as P


# --- 레지스트리 기본 구조 ---
def test_registry_has_three_families():
    assert set(M.MODEL_REGISTRY) == {"llava_ov", "mimo_vl", "qwen2_5_vl"}


def test_model_id_and_lora_targets():
    assert M.model_id("llava_ov") == "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
    assert M.model_id("qwen2_5_vl") == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert "q_proj" in M.lora_targets("qwen2_5_vl")
    # lora_targets는 복사본 반환(레지스트리 오염 방지)
    M.lora_targets("llava_ov").append("xxx")
    assert "xxx" not in M.MODEL_REGISTRY["llava_ov"]["lora_targets"]


def test_get_spec_unknown_raises():
    with pytest.raises(ValueError):
        M.get_spec("gpt5")


def test_render_mode_branches():
    assert M.render_mode("llava_ov") == "llava_ov"
    assert M.render_mode("qwen2_5_vl") == "chat_template"
    assert M.render_mode("mimo_vl") == "chat_template"


# --- family_from_model_id ---
def test_family_from_model_id():
    assert M.family_from_model_id("llava-hf/llava-onevision-qwen2-0.5b-si-hf") == "llava_ov"
    assert M.family_from_model_id("Qwen/Qwen2.5-VL-7B-Instruct") == "qwen2_5_vl"
    assert M.family_from_model_id("XiaomiMiMo/MiMo-VL-7B-RL") == "mimo_vl"
    assert M.family_from_model_id("llava_ov") == "llava_ov"  # family 키 그대로
    assert M.family_from_model_id("meta-llama/Llama-3") is None


# --- detect_family: 경로 슬러그 우선 ---
def test_detect_family_by_path_slug():
    assert M.detect_family("outputs/qwen2_5_vl/merged/lora") == "qwen2_5_vl"
    assert M.detect_family("outputs/mimo_vl/merged/full") == "mimo_vl"
    assert M.detect_family("outputs/llava_ov/merged/full") == "llava_ov"


def test_detect_family_unknown_path():
    assert M.detect_family("/tmp/some-random-checkpoint") is None


def test_detect_family_by_config(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen2_5_vl"}', encoding="utf-8")
    assert M.detect_family(str(tmp_path)) == "qwen2_5_vl"


# --- 출력경로 헬퍼 ---
def test_path_helpers_structure():
    assert P.adapter_dir("qwen2_5_vl").as_posix().endswith("outputs/qwen2_5_vl/adapters/lora")
    assert P.merged_dir("qwen2_5_vl", "full").as_posix().endswith("outputs/qwen2_5_vl/merged/full")
    assert P.merged_dir("llava_ov", "lora").as_posix().endswith("outputs/llava_ov/merged/lora")
    assert P.eval_dir("mimo_vl").as_posix().endswith("outputs/mimo_vl/eval")
    assert P.submission_path("llava_ov").as_posix().endswith("outputs/llava_ov/eval/submission.csv")


def test_merged_dir_bad_variant():
    with pytest.raises(ValueError):
        P.merged_dir("llava_ov", "merged")
