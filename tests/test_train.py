"""학습 파이프라인 순수 함수 테스트 (T1~T3).

GPU/torch 없이 동작하는 부분만 검증한다. 모델 로딩/Trainer는 A100 스모크로 분리.
"""

import json

from PIL import Image

from src.train import prompt as P
from src.train import dataset as D


# --- 베이스라인 prompt_text 독립 재현 (정합 회귀 방지용 기준값) ---
def baseline_prompt_text(context, question, answers_json):
    """베이스라인 노트북 셀[13]의 prompt_text 구성을 그대로 옮긴 참조 구현."""
    answers = json.loads(answers_json)
    context_s = "Context: " + str(context)
    question_s = "Question: " + str(question)
    options = (
        "Options:\n"
        f"0. {answers[0]}\n"
        f"1. {answers[1]}\n"
        f"2. {answers[2]}\n"
    )
    pre_prompt = (
        "You are an expert Vision Language assistant. "
        "When given an image, a context, a question, and options, "
        "you MUST respond only with a JSON object"
    )
    post_prompt = (
        "Give the output in strict JSON format: "
        "{\n"
        '   "reason": "One short sentence of reasoning.",\n'
        '   "answer_id": "<one of: 0, 1, 2>"\n'
        "}\n"
    )
    rule_prompt = "Do NOT output multiple options."
    return (
        pre_prompt + "\n"
        + context_s + "\n"
        + question_s + "\n"
        + options + "\n"
        + post_prompt + "\n"
        + rule_prompt
    )


LEX = [
    "Can't answer", "Not answerable", "Undetermined", "Not known",
    "Not enough information", "Can't be determined", "Cannot be determined",
    "Not enough info", "Unknown", "Cannot answer",
]


def test_prompt_text_matches_baseline_exactly():
    ctx = "I saw a grandson and their grandfather outside the Walmart."
    q = "Who was not comfortable using the phone?"
    answers = json.dumps(["Not enough information", "The grandson", "The grandfather"])
    assert P.build_prompt_text(ctx, q, answers) == baseline_prompt_text(ctx, q, answers)


def test_prompt_text_accepts_list_and_string():
    ctx, q = "ctx", "q?"
    lst = ["a", "b", "c"]
    js = json.dumps(lst)
    assert P.build_prompt_text(ctx, q, lst) == P.build_prompt_text(ctx, q, js)


def test_anti_bias_prompt_off_by_default(monkeypatch):
    """BIAS_PROMPT_V2 미설정 → 안티바이어스 블록 미포함(베이스라인 정합 유지)."""
    monkeypatch.delenv("BIAS_PROMPT_V2", raising=False)
    text = P.build_prompt_text("ctx", "q?", json.dumps(["a", "b", "c"]))
    assert P.ANTI_BIAS_PROMPT not in text


def test_anti_bias_prompt_on_inserts_block(monkeypatch):
    """BIAS_PROMPT_V2=1 → PRE_PROMPT와 Context 사이에 안티바이어스 블록 삽입."""
    monkeypatch.setenv("BIAS_PROMPT_V2", "1")
    text = P.build_prompt_text("ctx", "q?", json.dumps(["a", "b", "c"]))
    assert P.ANTI_BIAS_PROMPT in text
    assert text.index(P.PRE_PROMPT) < text.index(P.ANTI_BIAS_PROMPT) < text.index("Context:")


def test_build_conversation_llava_ov_single_user_turn():
    """llava_ov: system 없는 단일 user turn(베이스라인 정합). family 생략 시 기본도 동일."""
    answers = json.dumps(["a", "b", "c"])
    conv = P.build_conversation("ctx", "q?", answers, family="llava_ov")
    assert len(conv) == 1 and conv[0]["role"] == "user"
    assert P.build_conversation("ctx", "q?", answers) == conv  # 기본 family=llava_ov 하위호환


def test_build_conversation_chat_template_splits_system_user():
    """qwen2_5_vl: system(지시) + user(이미지+데이터) 2-turn 분리."""
    conv = P.build_conversation("ctx", "q?", json.dumps(["a", "b", "c"]), family="qwen2_5_vl")
    assert [m["role"] for m in conv] == ["system", "user"]
    sys_text = conv[0]["content"][0]["text"]
    user_text = conv[1]["content"][-1]["text"]
    assert P.PRE_PROMPT in sys_text and "Context:" not in sys_text   # 지시는 system, 데이터 누출 없음
    assert "Context: ctx" in user_text                              # 데이터는 user
    assert any(c.get("type") == "image" for c in conv[1]["content"])  # user turn에 이미지 1개


def test_reason_unknown_branch():
    answers = ["Not enough information", "The grandson", "The grandfather"]
    # label 0 = Unknown류 → 정보 부족 문장
    r = P.synthesize_reason(answers, 0, LEX)
    assert r == "The context does not provide enough information to answer."


def test_reason_disambiguated_branch():
    answers = ["The grandfather", "Can't be determined", "The grandson"]
    # label 2 = 특정 옵션 → 해당 옵션 명시
    r = P.synthesize_reason(answers, 2, LEX)
    assert r == "The context indicates that the answer is The grandson."


def test_reason_unknown_surface_variants():
    # 아포스트로피 변종도 Unknown으로 인식
    answers = ["Can’t answer", "x", "y"]
    r = P.synthesize_reason(answers, 0, LEX)
    assert r == "The context does not provide enough information to answer."


def test_reason_variant_key_is_deterministic():
    # 같은 variant_key → 항상 같은 변형(학습 결정성). 정답 옵션은 어떤 변형에도 포함.
    answers = ["The grandfather", "Can't be determined", "The grandson"]
    r1 = P.synthesize_reason(answers, 2, LEX, variant_key="ctx|q")
    r2 = P.synthesize_reason(answers, 2, LEX, variant_key="ctx|q")
    assert r1 == r2 and "The grandson" in r1


def test_reason_variant_key_diversifies():
    # 다양한 variant_key → 여러 변형이 실제로 등장(단일 템플릿 암기 → bf16 NaN 차단).
    answers = ["The grandfather", "Can't be determined", "The grandson"]
    disambig = {P.synthesize_reason(answers, 2, LEX, variant_key=f"k{i}") for i in range(60)}
    amb = {P.synthesize_reason(answers, 1, LEX, variant_key=f"k{i}") for i in range(60)}
    assert len(disambig) >= 3 and len(amb) >= 3


def test_reason_empty_variant_key_is_backward_compatible():
    # variant_key="" → index 0 = 기존 고정 문구(하위호환).
    answers = ["The grandfather", "Can't be determined", "The grandson"]
    assert P.synthesize_reason(answers, 2, LEX) == "The context indicates that the answer is The grandson."
    amb = ["Not enough information", "x", "y"]
    assert P.synthesize_reason(amb, 0, LEX) == "The context does not provide enough information to answer."


def test_target_json_roundtrip_and_answer_id():
    answers = ["The grandfather", "Can't be determined", "The grandson"]
    s = P.build_target_json(answers, 2, LEX)
    obj = json.loads(s)
    assert set(obj.keys()) == {"reason", "answer_id"}
    assert obj["answer_id"] == "2"
    assert isinstance(obj["reason"], str) and obj["reason"]


def test_target_json_answer_id_is_string_label():
    answers = ["Unknown", "b", "c"]
    for lbl in (0, 1, 2):
        obj = json.loads(P.build_target_json(answers, lbl, LEX))
        assert obj["answer_id"] == str(lbl)


def test_build_conversation_structure():
    conv = P.build_conversation("ctx", "q?", ["a", "b", "c"])
    assert conv[0]["role"] == "user"
    types = [c["type"] for c in conv[0]["content"]]
    assert "image" in types and "text" in types


# --- T2: split ---
def _mk_rows(n):
    return [
        {
            "sample_id": f"TRAIN_{i:06d}",
            "image_path": f"./images/train_img_{i:06d}.jpg",
            "context": f"ctx{i}",
            "question": "q?",
            "answers": json.dumps(["a", "b", "c"]),
            "label": str(i % 3),
        }
        for i in range(n)
    ]


def test_split_is_deterministic():
    rows = _mk_rows(200)
    t1, v1 = D.split_train_val(rows, seed=42, val_ratio=0.05)
    t2, v2 = D.split_train_val(list(reversed(rows)), seed=42, val_ratio=0.05)
    # 입력 순서가 달라도 동일 분할 (sample_id 정렬 + derive_rng)
    assert [r["sample_id"] for r in v1] == [r["sample_id"] for r in v2]
    assert [r["sample_id"] for r in t1] == [r["sample_id"] for r in t2]


def test_split_ratio_and_disjoint():
    rows = _mk_rows(200)
    train, val = D.split_train_val(rows, seed=42, val_ratio=0.05)
    assert len(val) == 10
    assert len(train) == 190
    ids_t = {r["sample_id"] for r in train}
    ids_v = {r["sample_id"] for r in val}
    assert ids_t.isdisjoint(ids_v)
    assert len(ids_t | ids_v) == 200


# --- T1(OOD): leave-axis-out 3분할 ---
def _mk_meta(rows, axes):
    """rows에 축을 라운드로빈 배정한 {sample_id: meta}."""
    return {
        r["sample_id"]: {"sample_id": r["sample_id"], "axis": axes[i % len(axes)]}
        for i, r in enumerate(rows)
    }


def test_split_ood_holds_out_only_given_axes():
    rows = _mk_rows(300)
    meta = _mk_meta(rows, ["Age", "Religion", "SES", "Sexual_orientation"])
    train, in_val, ood = D.split_train_val_ood(
        rows, meta, seed=42, val_ratio=0.05, ood_axes=["Religion", "Sexual_orientation"]
    )
    ood_axes_seen = {meta[r["sample_id"]]["axis"] for r in ood}
    assert ood_axes_seen == {"Religion", "Sexual_orientation"}
    # train/in_val에는 OOD 축이 전혀 없어야 함
    for r in train + in_val:
        assert meta[r["sample_id"]]["axis"] not in {"Religion", "Sexual_orientation"}


def test_split_ood_partition_is_complete_and_disjoint():
    rows = _mk_rows(300)
    meta = _mk_meta(rows, ["Age", "Religion", "SES", "Sexual_orientation"])
    train, in_val, ood = D.split_train_val_ood(
        rows, meta, seed=42, val_ratio=0.05, ood_axes=["Religion"]
    )
    ids = [{r["sample_id"] for r in s} for s in (train, in_val, ood)]
    assert ids[0].isdisjoint(ids[1]) and ids[0].isdisjoint(ids[2]) and ids[1].isdisjoint(ids[2])
    assert ids[0] | ids[1] | ids[2] == {r["sample_id"] for r in rows}


def test_split_ood_empty_axes_matches_plain_split():
    rows = _mk_rows(200)
    meta = _mk_meta(rows, ["Age", "SES"])
    train, in_val, ood = D.split_train_val_ood(rows, meta, seed=42, val_ratio=0.05, ood_axes=[])
    pt, pv = D.split_train_val(rows, seed=42, val_ratio=0.05)
    assert ood == []
    assert [r["sample_id"] for r in train] == [r["sample_id"] for r in pt]
    assert [r["sample_id"] for r in in_val] == [r["sample_id"] for r in pv]


def test_split_ood_is_deterministic():
    rows = _mk_rows(300)
    meta = _mk_meta(rows, ["Age", "Religion", "SES"])
    a = D.split_train_val_ood(rows, meta, seed=42, val_ratio=0.05, ood_axes=["Religion"])
    b = D.split_train_val_ood(list(reversed(rows)), meta, seed=42, val_ratio=0.05, ood_axes=["Religion"])
    for sa, sb in zip(a, b):
        assert [r["sample_id"] for r in sa] == [r["sample_id"] for r in sb]


def test_resolve_image_path():
    p = D.resolve_image_path("/data/train", "./images/train_img_000001.jpg")
    assert str(p) == "/data/train/images/train_img_000001.jpg"


def test_dataset_getitem():
    ds = D.BiasVQADataset(_mk_rows(3), "/data/train")
    item = ds[1]
    assert item["label"] == 1
    assert item["context"] == "ctx1"
    assert str(item["image_path"]).endswith("images/train_img_000001.jpg")


# --- T3: label 마스킹 / 이미지 로더 ---
from src.train import collator as C  # noqa: E402


def test_mask_prompt_tokens_masks_prompt_and_pad():
    input_ids = [5, 6, 7, 8, 9, 0, 0]  # prompt=처음3, 응답=8,9, pad=0
    labels = C.mask_prompt_tokens(input_ids, prompt_len=3, pad_id=0)
    assert labels == [-100, -100, -100, 8, 9, -100, -100]


def test_mask_prompt_tokens_no_pad():
    input_ids = [1, 2, 3, 4]
    labels = C.mask_prompt_tokens(input_ids, prompt_len=2, pad_id=-1)
    assert labels == [-100, -100, 3, 4]


def test_mask_all_prompt_when_no_response():
    input_ids = [1, 2, 3]
    labels = C.mask_prompt_tokens(input_ids, prompt_len=3, pad_id=-1)
    assert labels == [-100, -100, -100]


def test_load_image_resizes_width(tmp_path):
    p = tmp_path / "img.jpg"
    Image.new("RGB", (640, 480), (128, 128, 128)).save(p)
    out = C.load_image(str(p), img_size=224)
    assert out is not None
    assert out.size[0] == 224  # width 고정
    assert out.size[1] == int(480 * (224 / 640))


def test_load_image_bad_path_returns_none():
    assert C.load_image("/nonexistent/x.jpg", img_size=224) is None
