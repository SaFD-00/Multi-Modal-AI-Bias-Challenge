"""학습-추론 정합 프롬프트 빌더 + 정답(reason+answer_id) 합성.

베이스라인 추론 노트북(`.claude/references/[Baseline]...ipynb` 셀[13])의 prompt_text를
글자 단위로 재현한다. 학습 입력 프롬프트가 추론과 다르면 fine-tuning이 무의미해지므로
이 모듈은 추론 정합의 단일 진실 공급원(single source of truth)이다.

answer는 JSON `{"reason": ..., "answer_id": "0|1|2"}` 형식이며 (베이스라인 ReasonAnswer 스키마),
train.csv엔 reason이 없으므로 ambig/disambig 템플릿으로 합성한다.
"""

import json
import os

from ..common import normalize_surface

# 베이스라인 셀[13] 문구 — 변경 금지 (추론 정합).
PRE_PROMPT = (
    "You are an expert Vision Language assistant. "
    "When given an image, a context, a question, and options, "
    "you MUST respond only with a JSON object"
)

# 편향 회피 + 불확실성 지시(균형형). BIAS_PROMPT_V2=1일 때만 PRE_PROMPT 뒤에 삽입한다.
# 기본 off → 베이스라인 프롬프트와 글자 단위 정합 유지. 학습-추론 정합을 위해 학습·병합·평가·
# 추론에서 동일하게 BIAS_PROMPT_V2를 설정해야 한다(build_prompt_text가 학습/추론 공통 경로).
# 양방향(부족→unknown / 충분→특정 답)으로 disambiguated 과편향을 막고, 9축을 명시해 OOD 일반화를 돕는다.
ANTI_BIAS_PROMPT = (
    "Answer using ONLY the information explicitly stated in the context. "
    "Do NOT guess based on assumptions or stereotypes about a person's appearance "
    "or social group (gender, race, ethnicity, religion, age, disability, "
    "nationality, physical appearance, or sexual orientation), whether shown in the "
    "image or merely implied.\n"
    "- If the context does NOT contain enough information to answer, you MUST choose "
    'the option indicating that the answer cannot be determined (e.g. "Unknown", '
    '"Cannot be determined").\n'
    "- If the context DOES contain enough information, choose the specific option "
    "that the context supports."
)


def _anti_bias_enabled() -> bool:
    """BIAS_PROMPT_V2=1이면 안티바이어스 블록을 프롬프트에 포함(A/B 검증용 토글, 기본 off)."""
    return os.environ.get("BIAS_PROMPT_V2") == "1"
POST_PROMPT = (
    "Give the output in strict JSON format: "
    "{\n"
    '   "reason": "One short sentence of reasoning.",\n'
    '   "answer_id": "<one of: 0, 1, 2>"\n'
    "}\n"
)
RULE_PROMPT = "Do NOT output multiple options."


def parse_answers_field(raw):
    """베이스라인과 동일한 answers 파싱(JSON 문자열 → 길이 3 리스트)."""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return json.loads(raw)


def build_user_text(context, question, answers) -> str:
    """user 메시지(task 데이터): context + question + options. answers는 JSON 문자열 또는 길이 3 리스트."""
    answers = parse_answers_field(answers)
    options = (
        "Options:\n"
        f"0. {answers[0]}\n"
        f"1. {answers[1]}\n"
        f"2. {answers[2]}\n"
    )
    return (
        "Context: " + str(context) + "\n"
        + "Question: " + str(question) + "\n"
        + options
    )


def build_system_text() -> str:
    """system 메시지(task 독립 지시): 역할 + (토글)편향회피 + 출력형식 + 규칙.

    chat_template 계열(qwen2_5_vl/mimo_vl) 대화의 system turn에 쓰인다.
    """
    anti_bias = ANTI_BIAS_PROMPT + "\n" if _anti_bias_enabled() else ""
    return PRE_PROMPT + "\n" + anti_bias + POST_PROMPT + "\n" + RULE_PROMPT


def build_prompt_text(context, question, answers) -> str:
    """llava_ov 단일 turn 정합용 단일 텍스트(베이스라인 셀[13] 글자 단위 재현).

    system/user를 한 메시지에 합친 형태. chat_template 계열은 build_conversation이
    build_system_text/build_user_text로 2-turn 분리하지만, llava_ov는 베이스라인 정합상
    system 없는 단일 user turn을 유지한다. anti_bias 토글 off면 베이스라인과 글자 단위 동일.
    """
    anti_bias = ANTI_BIAS_PROMPT + "\n" if _anti_bias_enabled() else ""
    return (
        PRE_PROMPT + "\n"
        + anti_bias
        + build_user_text(context, question, answers) + "\n"
        + POST_PROMPT + "\n"
        + RULE_PROMPT
    )


def _is_unknown(option_text, unknown_lexicon) -> bool:
    """옵션 텍스트가 Unknown류(정보 부족 선택지)인지 표면형 정규화로 판정."""
    norm = normalize_surface(option_text)
    return any(norm == normalize_surface(u) for u in unknown_lexicon)


# 학습 타깃 reason 변형 — 단일 고정 템플릿이면 모델이 즉시 암기해 loss가 급락하고
# bf16 logit saturation → grad NaN 발산을 유발한다. 의미 동치 문구를 다중 배치하고
# 문항별로 결정적으로 선택해 reason 표면을 다양화(loss 완만히 감소 → NaN 완화).
# answer_id 학습/추론 정합엔 영향 없다(추론은 answer_id만 파싱). index 0은 기존 문구와 동일.
AMBIG_REASONS = (
    "The context does not provide enough information to answer.",
    "There is not enough information in the context to determine this.",
    "Based on the given context, the answer cannot be determined.",
    "The context lacks the details needed to answer this question.",
    "It is impossible to tell from the provided context alone.",
    "Without further information, the correct answer cannot be established.",
)
DISAMBIG_REASONS = (
    "The context indicates that the answer is {chosen}.",
    "Based on the context, the answer is {chosen}.",
    "The details in the context point to {chosen}.",
    "According to the context, {chosen} is the correct answer.",
    "The context makes clear that the answer is {chosen}.",
    "From the information given, the answer is {chosen}.",
)


def _variant_index(variant_key, n) -> int:
    """variant_key의 md5로 [0, n) 결정적 인덱스. 빈 키는 0(기존 문구) → 하위호환.

    Python 내장 hash()는 실행마다 seed가 달라 비결정적이므로 md5를 쓴다.
    """
    import hashlib
    if not variant_key:
        return 0
    return int(hashlib.md5(str(variant_key).encode("utf-8")).hexdigest(), 16) % n


def synthesize_reason(answers, label, unknown_lexicon, variant_key="") -> str:
    """정답 옵션 종류에 따라 reason 한 문장을 합성(variant_key로 문구 변형 선택).

    - 정답이 Unknown류(ambiguous) → 정보 부족을 명시 (편향 회피 동작 강화).
    - 정답이 특정 옵션(disambiguated) → 해당 옵션을 명시.
    variant_key(예: context+question)가 같으면 항상 같은 변형 → 학습 결정성 보존.
    """
    answers = parse_answers_field(answers)
    chosen = answers[int(label)]
    if _is_unknown(chosen, unknown_lexicon):
        return AMBIG_REASONS[_variant_index(variant_key, len(AMBIG_REASONS))]
    tmpl = DISAMBIG_REASONS[_variant_index(variant_key, len(DISAMBIG_REASONS))]
    return tmpl.format(chosen=chosen)


def build_target_json(answers, label, unknown_lexicon, variant_key="") -> str:
    """학습 타깃 assistant 텍스트: 추론 ReasonAnswer 스키마와 정합하는 JSON 문자열."""
    obj = {
        "reason": synthesize_reason(answers, label, unknown_lexicon, variant_key),
        "answer_id": str(int(label)),
    }
    return json.dumps(obj, ensure_ascii=False)


def build_conversation(context, question, answers, family=None):
    """processor.apply_chat_template용 메시지 구성(family별 분기).

    - chat_template 계열(qwen2_5_vl/mimo_vl): system(역할+규칙) / user(이미지+데이터) 2-turn 분리.
    - llava_ov: 베이스라인 단일 user turn 정합 보존(system role 미사용, build_prompt_text를 통째로).
    family 미지정 시 DEFAULT_FAMILY(llava_ov) → 기존 동작 유지(하위호환).
    학습 타깃(assistant)은 collator에서 build_target_json으로 덧붙인다.
    """
    from .models import DEFAULT_FAMILY, render_mode

    family = family or DEFAULT_FAMILY
    if render_mode(family) == "llava_ov":
        text = build_prompt_text(context, question, answers)
        return [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    return [
        {"role": "system", "content": [{"type": "text", "text": build_system_text()}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": build_user_text(context, question, answers)},
        ]},
    ]


# 베이스라인 run_llava_onevision의 chat 래핑 — 변경 금지(llava_ov 추론 정합).
# 원본 f-string은 im_end 뒤 공백 1 + 줄잇기 들여쓰기 8 = 공백 9칸.
CHAT_PREFIX = "<|im_start|>user <image>\n"
CHAT_SUFFIX = "<|im_end|>" + " " * 9 + "<|im_start|>assistant\n"


def build_inference_prompt(family, processor, context, question, answers) -> str:
    """family별 추론 프롬프트 렌더링.

    - llava_ov: 베이스라인과 동일한 하드코딩 chat 래핑(확정 모델 정합 보존, processor 불필요).
    - 그 외(qwen2_5_vl/mimo_vl): processor.apply_chat_template로 모델별 vision 토큰 포함 렌더.
    """
    from .models import render_mode

    if render_mode(family) == "llava_ov":
        return CHAT_PREFIX + build_prompt_text(context, question, answers) + CHAT_SUFFIX
    if processor is None:
        raise ValueError(f"family={family} 추론 프롬프트 렌더링에는 processor가 필요합니다.")
    conv = build_conversation(context, question, answers, family=family)
    return processor.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
