"""추론/제출 순수 함수 테스트.

GPU/vllm 없이 동작하는 부분만 검증한다(JSON 파싱·프롬프트 래핑).
실제 vLLM 추론은 GPU 환경 스모크로 분리.
"""

import json

import pytest

from src.predict import build_chat_prompt, extract_answer_id, normalize_answer_id
from src.train.prompt import build_inference_prompt, build_prompt_text

ANSWERS = json.dumps(["A man", "A woman", "Cannot be determined"])


# --- extract_answer_id ---
def test_extract_clean_json():
    assert extract_answer_id('{"reason": "x", "answer_id": "2"}') == "2"


def test_extract_json_with_surrounding_text():
    text = 'Sure! {"reason": "x", "answer_id": "1"} done.'
    assert extract_answer_id(text) == "1"


def test_extract_broken_json_falls_back_to_zero():
    assert extract_answer_id('{"answer_id": ') == "0"


def test_extract_out_of_range_falls_back_to_zero():
    assert extract_answer_id('{"reason": "x", "answer_id": "3"}') == "0"


def test_extract_empty_and_none():
    assert extract_answer_id("") == "0"
    assert extract_answer_id(None) == "0"


# --- normalize_answer_id ---
def test_normalize():
    assert normalize_answer_id("0") == "0"
    assert normalize_answer_id("2") == "2"
    assert normalize_answer_id("9") == "0"
    assert normalize_answer_id(None) == "0"


# --- build_chat_prompt: 학습 프롬프트를 그대로 포함 + chat 래핑 ---
def test_chat_prompt_wraps_training_prompt():
    inner = build_prompt_text("ctx", "q?", ANSWERS)
    wrapped = build_chat_prompt("ctx", "q?", ANSWERS)
    assert inner in wrapped
    assert wrapped.startswith("<|im_start|>user <image>\n")
    assert wrapped.endswith("<|im_start|>assistant\n")


# --- build_inference_prompt: family 분기 ---
def test_inference_prompt_llava_ov_equals_chat_prompt():
    # llava_ov는 processor 없이 기존 chat 래핑과 글자 단위 동일(확정 모델 정합 보존).
    assert build_inference_prompt("llava_ov", None, "ctx", "q?", ANSWERS) == \
        build_chat_prompt("ctx", "q?", ANSWERS)


def test_inference_prompt_chat_template_requires_processor():
    # qwen2_5_vl/mimo_vl은 processor.apply_chat_template 필요 → processor=None이면 명시적 에러.
    with pytest.raises(ValueError):
        build_inference_prompt("qwen2_5_vl", None, "ctx", "q?", ANSWERS)
