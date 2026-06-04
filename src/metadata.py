"""T4 — test 누수 제거 + 메타데이터.

제공 test.csv의 (context+question) 정규화 해시와 대조해 누수 학습샘플을 제거하고,
전 샘플의 출처/라이선스(NC 플래그)를 metadata.jsonl에 기록한다.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from src.common import LICENSE_SB, leak_key, load_config, resolve_path


# --- 순수 함수 ---

def build_test_keyset(test_rows: list[dict]) -> set[str]:
    """test.csv 모든 (context, question)의 누수 키 집합."""
    return {leak_key(r.get("context", ""), r.get("question", "")) for r in test_rows}


def filter_leaks(records: list[dict], test_keyset: set[str]) -> tuple[list[dict], int]:
    """test와 (context+question)이 겹치는 레코드 제거. (kept, removed_count) 반환."""
    kept: list[dict] = []
    removed = 0
    for r in records:
        if leak_key(r.get("context", ""), r.get("question", "")) in test_keyset:
            removed += 1
        else:
            kept.append(r)
    return kept, removed


def make_metadata_row(rec: dict, sample_id: str) -> dict:
    """한 샘플의 출처/라이선스 메타 레코드."""
    license_ = rec.get("license", "")
    return {
        "sample_id": sample_id,
        "uid": rec.get("uid"),
        "text_source": rec.get("source"),
        "image_source": rec.get("source") if rec.get("image_ref") else None,
        "license": license_,
        "is_nc": license_ == LICENSE_SB,
        "axis": rec.get("axis"),
        "ambig": rec.get("ambig"),
        "polarity": rec.get("polarity"),
    }


# --- I/O ---

def run(config: dict | None = None) -> None:
    import pandas as pd

    cfg = config or load_config()
    train_csv = resolve_path(cfg, "train_csv")
    mapped_path = resolve_path(cfg, "mapped")
    test_csv = resolve_path(cfg, "test_csv")
    meta_path = resolve_path(cfg, "metadata")

    df = pd.read_csv(train_csv)

    # uid → mapped 레코드 (출처/라이선스 회수용). train.csv엔 uid가 없으므로
    # mapped를 context+question 키로 역참조.
    mapped = {}
    if mapped_path.exists():
        for line in mapped_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                m = json.loads(line)
                mapped[leak_key(m["context"], m["question"])] = m

    # test 누수 제거
    removed = 0
    if test_csv.exists():
        test_rows = pd.read_csv(test_csv).to_dict("records")
        keyset = build_test_keyset(test_rows)
        mask = ~df.apply(lambda r: leak_key(r["context"], r["question"]) in keyset, axis=1)
        removed = int((~mask).sum())
        df = df[mask].reset_index(drop=True)
        # sample_id 재부여 (연속성 유지)
        df["sample_id"] = [f"TRAIN_{i:06d}" for i in range(len(df))]
        df.to_csv(train_csv, index=False)

    # 메타데이터 작성
    lic_cnt: Counter = Counter()
    src_cnt: Counter = Counter()
    with open(meta_path, "w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            m = mapped.get(leak_key(r["context"], r["question"]), {})
            row = make_metadata_row(m, r["sample_id"])
            lic_cnt[row["license"]] += 1
            src_cnt[row["text_source"]] += 1
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[metadata] train rows={len(df)} | test 누수 제거={removed}")
    print(f"[metadata] source={dict(src_cnt)} | license={dict(lic_cnt)}")
    print(f"[metadata] -> {meta_path}")


if __name__ == "__main__":
    run()
