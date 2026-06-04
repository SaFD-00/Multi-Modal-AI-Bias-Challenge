"""compose 순수 함수 검증 — 정답 보존 불변식이 핵심."""
import json
import random
from collections import Counter

import pytest

from src.compose import (
    assign_unknown_positions,
    compose_record,
    place_unknown_at,
    rediversify_unknown,
    reshuffle_options_and_remap_label,
    select_by_ratio,
    to_csv_row,
)

LEXICON = [
    "Can't answer", "Not answerable", "Undetermined", "Not known",
    "Not enough information", "Can't be determined", "Cannot be determined",
    "Not enough info", "Unknown", "Cannot answer",
]


def test_reshuffle_preserves_answer_all_seeds():
    options = ["A", "B", "Cannot be determined"]
    for label in range(3):
        for seed in range(50):
            rng = random.Random(seed)
            new_opts, new_label = reshuffle_options_and_remap_label(options, label, rng)
            assert new_opts[new_label] == options[label]
            assert sorted(new_opts) == sorted(options)  # 멀티셋 보존


def test_place_unknown_at_target_and_preserve():
    options = ["A", "B", "Cannot be determined"]
    unk_idx = 2
    for label in range(3):
        for target in range(3):
            for seed in range(30):
                rng = random.Random(seed)
                new_opts, new_label, new_unk = place_unknown_at(options, label, unk_idx, target, rng)
                assert new_unk == target
                assert new_opts[target] == options[unk_idx]  # Unknown이 목표 위치
                assert new_opts[new_label] == options[label]  # 정답 보존
                assert sorted(new_opts) == sorted(options)


def test_rediversify_in_lexicon_and_deterministic():
    for seed in range(20):
        r1 = random.Random(seed)
        r2 = random.Random(seed)
        out1 = rediversify_unknown("Cannot be determined", r1, LEXICON)
        out2 = rediversify_unknown("Cannot be determined", r2, LEXICON)
        assert out1 in LEXICON
        assert out1 == out2  # 동일 시드 → 동일 출력


def test_rediversify_follows_distribution():
    dist = {t: 0 for t in LEXICON}
    dist["Unknown"] = 1  # 100% Unknown
    rng = random.Random(0)
    outs = [rediversify_unknown("x", rng, LEXICON, dist) for _ in range(100)]
    assert set(outs) == {"Unknown"}


def test_assign_positions_uniform():
    rng = random.Random(0)
    seq = assign_unknown_positions(300, rng)
    cnt = Counter(seq)
    assert cnt[0] == cnt[1] == cnt[2] == 100  # 정확히 균등 (300 = 3*100)
    # 비배수 n도 ±1 이내
    seq2 = assign_unknown_positions(301, random.Random(1))
    c2 = Counter(seq2)
    assert max(c2.values()) - min(c2.values()) <= 1


def test_compose_record_full_invariants():
    rec = {
        "uid": "sb-1", "context": "ctx", "question": "q",
        "options": ["P", "Q", "Cannot be determined"], "label": 0,
        "unknown_idx": 2, "unknown_text": "Cannot be determined",
        "axis": "Age", "polarity": "neg", "ambig": False, "image_ref": "./images/x.jpg",
        "source": "sb-bench", "license": "CC-BY-NC-4.0",
    }
    for target in range(3):
        rng = random.Random(target)
        out = compose_record(rec, target, rng, LEXICON, None)
        assert out["unknown_idx"] == target
        assert out["options"][target] in LEXICON          # Unknown 재다양화됨
        assert out["options"][out["label"]] == "P"         # 원 정답(idx0=P) 보존


def test_compose_record_unknown_is_answer_in_ambig():
    """ambig 행: 정답이 Unknown일 때 재다양화해도 label이 새 Unknown을 가리킴."""
    rec = {
        "uid": "sb-2", "context": "ctx", "question": "q",
        "options": ["P", "Q", "Cannot be determined"], "label": 2,
        "unknown_idx": 2, "unknown_text": "Cannot be determined",
        "axis": "Age", "polarity": "neg", "ambig": True, "image_ref": None,
        "source": "bbq", "license": "CC-BY-4.0",
    }
    for target in range(3):
        out = compose_record(rec, target, random.Random(target), LEXICON, None)
        assert out["label"] == target              # 정답 = Unknown 위치
        assert out["options"][out["label"]] in LEXICON


def test_to_csv_row_json_roundtrip():
    rec = {
        "options": ["P", "Q", "Unknown"], "label": 1,
        "context": "c", "question": "q", "image_ref": "./images/x.jpg",
    }
    row = to_csv_row(rec, "TRAIN_000001")
    parsed = json.loads(row["answers"])
    assert len(parsed) == 3
    assert parsed[row["label"]] == "Q"
    assert row["sample_id"] == "TRAIN_000001"


def test_select_by_ratio_caps_per_cell():
    records = []
    for i in range(100):
        records.append({
            "uid": f"u{i:03d}", "axis": "Age", "ambig": True, "polarity": "neg",
            "options": ["a", "b", "Unknown"], "label": 0, "unknown_idx": 2,
        })
    sel = select_by_ratio(records, target_per_cell=10, rng=random.Random(0))
    assert len(sel) == 10  # 단일 셀 상한 적용


def test_select_by_ratio_deterministic():
    records = [{
        "uid": f"u{i:03d}", "axis": "Age", "ambig": (i % 2 == 0), "polarity": "neg",
        "options": ["a", "b", "Unknown"], "label": 0, "unknown_idx": 2,
    } for i in range(40)]
    a = select_by_ratio(records, 5, random.Random(0))
    b = select_by_ratio(records, 5, random.Random(0))
    assert [r["uid"] for r in a] == [r["uid"] for r in b]
