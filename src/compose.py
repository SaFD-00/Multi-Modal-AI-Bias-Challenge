"""T3 — 구성·정렬 (★핵심).

mapped.jsonl → 비율 샘플링 → Unknown 재다양화 + 위치 재셔플 + label 재매핑 →
대회 train.csv 직렬화. 정답 보존 불변식이 집중되는 모듈이므로 순수 함수에 테스트를 집중한다.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.common import (
    TRAIN_COLUMNS,
    derive_rng,
    load_config,
    normalize_surface,
    resolve_path,
)


# --- 순수 함수 ---

def reshuffle_options_and_remap_label(
    options: list[str], label: int, rng: random.Random
) -> tuple[list[str], int]:
    """선택지 순서를 무작위화하고 label을 재매핑.

    불변식: new_options[new_label] == options[label] (정답 보존).
    선택지 멀티셋은 보존된다.
    """
    n = len(options)
    perm = list(range(n))
    rng.shuffle(perm)  # perm[new_pos] = old_pos
    new_options = [options[old] for old in perm]
    new_label = perm.index(label)
    return new_options, new_label


def place_unknown_at(
    options: list[str], label: int, unknown_idx: int, target_pos: int, rng: random.Random
) -> tuple[list[str], int, int]:
    """Unknown 슬롯을 target_pos로 보내고 나머지 두 슬롯은 무작위 배치.

    반환: (new_options, new_label, new_unknown_idx). 정답·Unknown 위치 모두 동기 갱신.
    """
    n = len(options)
    others = [i for i in range(n) if i != unknown_idx]
    rng.shuffle(others)
    # new_pos -> old_idx 매핑 구성
    src_by_newpos = [None] * n
    src_by_newpos[target_pos] = unknown_idx
    fill = [p for p in range(n) if p != target_pos]
    for newpos, old in zip(fill, others):
        src_by_newpos[newpos] = old
    new_options = [options[src_by_newpos[p]] for p in range(n)]
    new_label = src_by_newpos.index(label)
    return new_options, new_label, target_pos


def rediversify_unknown(
    unknown_text: str, rng: random.Random, lexicon: list[str], dist: dict | None = None
) -> str:
    """Unknown 표면형을 lexicon 10종 중 하나로 치환.

    dist(표현→가중치)가 주어지면 그 분포로, 없으면 균등 추출. test 관측 비례 재현용.
    """
    if dist:
        terms = list(dist.keys())
        weights = [dist[t] for t in terms]
        return rng.choices(terms, weights=weights, k=1)[0]
    return rng.choice(lexicon)


def assign_unknown_positions(n: int, rng: random.Random, k: int = 3) -> list[int]:
    """0..k-1 위치를 균등 배분한 길이 n 시퀀스(라운드로빈 후 셔플).

    무작위 추출만 쓰면 소표본에서 위치 편향이 생기므로 균등 배분을 보장한다.
    """
    seq = [i % k for i in range(n)]
    rng.shuffle(seq)
    return seq


def cell_key(rec: dict) -> tuple[str, bool, str]:
    return (rec["axis"], bool(rec["ambig"]), rec["polarity"])


def select_by_ratio(
    records: list[dict], target_per_cell: int, rng: random.Random
) -> list[dict]:
    """셀(axis × ambig × polarity)별 상한 target_per_cell까지 샘플링.

    ambig:disambig=1:1, polarity 1:1, 9축 균등은 셀별 동일 상한으로 자연 달성.
    결정성을 위해 uid 정렬 후 셔플.
    """
    by_cell: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        by_cell[cell_key(r)].append(r)
    selected: list[dict] = []
    for key in sorted(by_cell.keys()):
        bucket = sorted(by_cell[key], key=lambda r: r["uid"])
        rng.shuffle(bucket)
        selected.extend(bucket[:target_per_cell])
    selected.sort(key=lambda r: r["uid"])
    return selected


def compose_record(
    rec: dict, target_pos: int, rng: random.Random, lexicon: list[str], dist: dict | None
) -> dict:
    """한 레코드를 구성: Unknown 텍스트 재다양화 → 위치/순서 재셔플 + label·unknown_idx 재매핑."""
    options = list(rec["options"])
    label = int(rec["label"])
    unk_idx = int(rec["unknown_idx"])
    # 1) Unknown 텍스트 재다양화
    options[unk_idx] = rediversify_unknown(rec.get("unknown_text", ""), rng, lexicon, dist)
    # 2) Unknown을 target_pos로 보내며 전체 재셔플 + label 재매핑
    new_opts, new_label, new_unk = place_unknown_at(options, label, unk_idx, target_pos, rng)
    return {
        "uid": rec["uid"],
        "context": rec["context"],
        "question": rec["question"],
        "options": new_opts,
        "label": new_label,
        "unknown_idx": new_unk,
        "image_ref": rec.get("image_ref"),
        "source": rec.get("source"),
        "license": rec.get("license"),
        "axis": rec["axis"],
        "polarity": rec["polarity"],
        "ambig": rec["ambig"],
    }


def to_csv_row(rec: dict, sample_id: str) -> dict:
    """대회 CSV 한 행으로 직렬화 (answers=JSON 문자열)."""
    return {
        "sample_id": sample_id,
        "image_path": rec.get("image_ref") or "",
        "context": rec["context"],
        "question": rec["question"],
        "answers": json.dumps(rec["options"], ensure_ascii=False),
        "label": int(rec["label"]),
    }


def measure_unknown_distribution(test_csv: Path, lexicon: list[str]) -> dict:
    """test.csv에서 Unknown 표현 10종의 관측 빈도(가중치)를 측정."""
    import pandas as pd

    surf2canon = {normalize_surface(t): t for t in lexicon}
    counter: Counter = Counter()
    df = pd.read_csv(test_csv)
    for a in df["answers"]:
        for o in json.loads(a):
            canon = surf2canon.get(normalize_surface(o))
            if canon:
                counter[canon] += 1
    return {t: counter.get(t, 0) for t in lexicon}


# --- I/O ---

def run(config: dict | None = None) -> None:
    import pandas as pd

    cfg = config or load_config()
    seed = cfg["seed"]
    lexicon = cfg["unknown_lexicon"]
    mapped_path = resolve_path(cfg, "mapped")
    train_csv = resolve_path(cfg, "train_csv")

    # Unknown 목표 분포
    dist = None
    if cfg.get("unknown_distribution", "proportional") == "proportional":
        test_csv = resolve_path(cfg, "test_csv")
        if test_csv.exists():
            dist = measure_unknown_distribution(test_csv, lexicon)

    records = [json.loads(line) for line in mapped_path.read_text(encoding="utf-8").splitlines() if line.strip()]

    sel_rng = derive_rng(seed, "select")
    selected = select_by_ratio(records, cfg["target_per_cell"], sel_rng)

    pos_rng = derive_rng(seed, "positions")
    positions = assign_unknown_positions(len(selected), pos_rng)

    comp_rng = derive_rng(seed, "compose")
    rows = []
    for i, (rec, pos) in enumerate(zip(selected, positions)):
        composed = compose_record(rec, pos, comp_rng, lexicon, dist)
        rows.append(to_csv_row(composed, f"TRAIN_{i:06d}"))

    train_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(TRAIN_COLUMNS)).to_csv(train_csv, index=False)
    print(f"[compose] {len(rows)} rows -> {train_csv}")


if __name__ == "__main__":
    run()
