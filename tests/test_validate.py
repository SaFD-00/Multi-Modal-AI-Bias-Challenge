"""validate 순수 검증기 자체 검증 + 누수 제거 + 재현성."""
import json
import random

from src.common import leak_key, normalize_text
from src.compose import to_csv_row
from src.metadata import build_test_keyset, filter_leaks
from src.validate import (
    build_baseline_prompt,
    check_distribution,
    check_schema,
    check_split_integrity,
    compute_bias_scores,
)

LEXICON = [
    "Can't answer", "Not answerable", "Undetermined", "Not known",
    "Not enough information", "Can't be determined", "Cannot be determined",
    "Not enough info", "Unknown", "Cannot answer",
]


def _row(opts, label):
    return {
        "sample_id": "TRAIN_000000", "image_path": "", "context": "c",
        "question": "q", "answers": json.dumps(opts), "label": label,
    }


def test_check_schema_pass():
    rows = [_row(["A", "B", "Unknown"], 1)]
    assert check_schema(rows, LEXICON) == []


def test_check_schema_detects_errors():
    assert check_schema([_row(["A", "B"], 0)], LEXICON)            # 길이 2
    assert check_schema([_row(["A", "B", "C"], 3)], LEXICON)        # label 3
    assert check_schema([_row(["A", "B", "C"], 0)], LEXICON)        # Unknown 없음
    bad = {"sample_id": "x", "answers": "not json", "label": 0,
           "image_path": "", "context": "c", "question": "q"}
    assert check_schema([bad], LEXICON)


def test_check_distribution_uniform_pass():
    rows = []
    for i in range(300):
        pos = i % 3
        opts = ["A", "B"]
        opts.insert(pos, LEXICON[i % 10])
        rows.append(_row(opts, pos))
    assert check_distribution(rows, LEXICON, tol=0.05) == []


def test_check_distribution_position_bias_fails():
    rows = [_row([LEXICON[i % 10], "A", "B"], 0) for i in range(100)]  # 전부 위치 0
    errs = check_distribution(rows, LEXICON, tol=0.05)
    assert any("위치" in e for e in errs)


def test_check_distribution_missing_terms_fails():
    rows = [_row([LEXICON[0], "A", "B"], 0) for _ in range(30)]  # 1종만
    errs = check_distribution(rows, LEXICON, tol=0.5)
    assert any("미등장" in e for e in errs)


def test_compute_bias_scores_known_case():
    # disambig 4개: 2개 정답, 2개 편향(non-unknown 오답). unk_idx=2
    preds = [0, 1, 0, 1]
    golds = [0, 1, 1, 0]
    ambig = [False, False, False, False]
    unk = [2, 2, 2, 2]
    s = compute_bias_scores(preds, golds, ambig, unk)
    # n_non_unknown=4, n_biased=2 → s_DIS = 2*(2/4)-1 = 0.0
    assert abs(s["s_DIS"] - 0.0) < 1e-9
    assert s["disambig_accuracy"] == 0.5


def test_compute_bias_scores_empty_safe():
    s = compute_bias_scores([], [], [], [])
    assert s["s_DIS"] == 0.0 and s["s_AMB"] == 0.0


def _ood_splits():
    """train/in_val/ood 3분할 + meta. ood 축 = Religion만."""
    def mk(ids, axis):
        return [{"sample_id": sid} for sid in ids], {sid: {"axis": axis} for sid in ids}
    tr, m1 = mk([f"TRAIN_{i:06d}" for i in range(0, 80)], "Age")
    iv, m2 = mk([f"TRAIN_{i:06d}" for i in range(80, 90)], "Age")
    od, m3 = mk([f"TRAIN_{i:06d}" for i in range(90, 100)], "Religion")
    meta = {**m1, **m2, **m3}
    return tr, iv, od, meta


def test_split_integrity_pass():
    tr, iv, od, meta = _ood_splits()
    assert check_split_integrity(tr, iv, od, meta, ["Religion"]) == []


def test_split_integrity_detects_overlap():
    tr, iv, od, meta = _ood_splits()
    tr2 = tr + [od[0]]  # OOD 샘플이 train에도 → 겹침 + 축 누출
    errs = check_split_integrity(tr2, iv, od, meta, ["Religion"])
    assert any("겹침" in e for e in errs)
    assert any("누출" in e for e in errs)


def test_split_integrity_detects_foreign_axis_in_ood():
    tr, iv, od, meta = _ood_splits()
    meta[od[0]["sample_id"]] = {"axis": "Age"}  # OOD에 비-OOD 축 섞임
    errs = check_split_integrity(tr, iv, od, meta, ["Religion"])
    assert any("ood_axes 외 축" in e for e in errs)


def test_normalize_text_and_leak_key():
    assert normalize_text("  Hello   World ") == "hello world"
    k1 = leak_key("The CONTEXT.", "Who?")
    k2 = leak_key("the   context.", "who?")
    assert k1 == k2  # 대소문자/공백 차이 무시


def test_filter_leaks_removes_test_rows():
    test_rows = [{"context": "Shared context here.", "question": "Who is it?"}]
    keyset = build_test_keyset(test_rows)
    records = [
        {"context": "shared CONTEXT here.", "question": "who is it?"},  # 누수
        {"context": "Different ctx.", "question": "Other q?"},          # 보존
    ]
    kept, removed = filter_leaks(records, keyset)
    assert removed == 1
    assert len(kept) == 1
    assert kept[0]["context"] == "Different ctx."


def test_baseline_prompt_contains_options():
    p = build_baseline_prompt("ctx", "q", ["A", "B", "Unknown"])
    assert "Options:\n0. A\n1. B\n2. Unknown" in p
    assert "Context: ctx" in p and "Question: q" in p


def test_reproducibility_compose_hash():
    from src.compose import compose_record
    rec = {
        "uid": "u1", "context": "c", "question": "q",
        "options": ["P", "Q", "Cannot be determined"], "label": 0,
        "unknown_idx": 2, "unknown_text": "Cannot be determined",
        "axis": "Age", "polarity": "neg", "ambig": False, "image_ref": None,
    }
    out1 = [to_csv_row(compose_record(rec, i % 3, random.Random(42), LEXICON, None), f"TRAIN_{i:06d}") for i in range(20)]
    out2 = [to_csv_row(compose_record(rec, i % 3, random.Random(42), LEXICON, None), f"TRAIN_{i:06d}") for i in range(20)]
    assert out1 == out2
