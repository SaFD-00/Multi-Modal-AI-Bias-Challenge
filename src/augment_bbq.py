"""T2 — BBQ 텍스트로 부족 셀 보강.

9축 × ambig/disambig × polarity 셀이 target_per_cell 미달일 때만 BBQ(CC-BY-4.0) 텍스트를
추출해 보강한다. 이미지는 동일 axis SB-Bench 이미지를 재사용해 분포 정합을 유지한다.
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.common import (
    LICENSE_BBQ,
    derive_rng,
    leak_key,
    load_config,
    resolve_path,
)
from src.map_sbbench import canon_axis


# --- 순수 함수 ---

def bbq_unknown_idx(answer_info: dict, options: list[str]) -> int | None:
    """answer_info에서 'unknown' 태그가 붙은 옵션 인덱스.

    answer_info = {"ans0": [...,"unknown"], "ans1": [...], ...} 형태(BBQ 표준).
    """
    if isinstance(answer_info, str):
        try:
            answer_info = json.loads(answer_info)
        except Exception:
            answer_info = {}
    if isinstance(answer_info, dict):
        for i in range(len(options)):
            tags = answer_info.get(f"ans{i}", [])
            tags = tags if isinstance(tags, (list, tuple)) else [tags]
            if any(str(t).strip().lower() == "unknown" for t in tags):
                return i
    return None


def map_bbq_row(row: dict) -> dict | None:
    """BBQ 한 행 → MappedRecord dict (image_ref=None; 이후 결합). 매핑 불가 시 None."""
    options = [row.get("ans0"), row.get("ans1"), row.get("ans2")]
    if any(o is None for o in options):
        return None
    options = [str(o) for o in options]
    try:
        label = int(row["label"])
    except Exception:
        return None
    if label not in (0, 1, 2):
        return None
    unk_idx = bbq_unknown_idx(row.get("answer_info", {}), options)
    if unk_idx is None:
        return None
    cc = str(row.get("context_condition", "")).lower()
    ambig = cc.startswith("ambig")
    polarity = "neg" if str(row.get("question_polarity", "")).lower().startswith("neg") else "nonneg"
    context = str(row.get("context", ""))
    question = str(row.get("question", ""))
    qid = f"{row.get('category','')}-{row.get('example_id','')}-{row.get('question_index','')}-{label}"
    return {
        "uid": f"bbq-{qid}-{leak_key(context, question)[:8]}",
        "source": "bbq",
        "license": LICENSE_BBQ,
        "axis": canon_axis(row.get("category", "")),
        "polarity": polarity,
        "ambig": ambig,
        "context": context,
        "question": question,
        "options": options,
        "label": label,
        "unknown_idx": unk_idx,
        "unknown_text": options[unk_idx],
        "image_ref": None,
        "norm_key": leak_key(context, question),
        "meta": {"orig_id": qid, "bbq_category": row.get("category")},
    }


def cell_key(rec: dict) -> tuple[str, bool, str]:
    return (rec["axis"], bool(rec["ambig"]), rec["polarity"])


def cells_needed(current_counts: dict, target_per_cell: int) -> dict:
    """셀별 부족분. target 이상인 셀은 제외."""
    return {k: target_per_cell - v for k, v in current_counts.items() if v < target_per_cell}


def build_image_pool(records: list[dict]) -> dict:
    """기존(SB-Bench) 레코드에서 axis별 이미지 경로 풀 구성."""
    pool: dict[str, list[str]] = defaultdict(list)
    for r in records:
        if r.get("image_ref"):
            pool[r["axis"]].append(r["image_ref"])
    return pool


def pair_image_for_bbq(record: dict, image_pool: dict, rng: random.Random) -> str | None:
    """동일 axis SB-Bench 이미지 재사용 (없으면 전체 풀에서, 그것도 없으면 None)."""
    candidates = image_pool.get(record["axis"])
    if not candidates:
        flat = [p for ps in image_pool.values() for p in ps]
        candidates = flat or None
    if not candidates:
        return None
    return rng.choice(candidates)


# --- I/O ---

# BBQ 원본 카테고리 파일 (GitHub, CC-BY-4.0). 9축 + 2 교차축.
BBQ_CATEGORIES = (
    "Age", "Disability_status", "Gender_identity", "Nationality",
    "Physical_appearance", "Race_ethnicity", "Religion", "SES",
    "Sexual_orientation", "Race_x_SES", "Race_x_gender",
)
BBQ_RAW_BASE = "https://raw.githubusercontent.com/nyu-mll/BBQ/main/data"


def load_bbq(cfg: dict) -> list[dict]:
    """BBQ JSONL을 GitHub raw에서 로드(캐시). 실패한 카테고리는 건너뛴다."""
    import urllib.request

    cache_dir = resolve_path(cfg, "bbq_dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for cat in BBQ_CATEGORIES:
        local = cache_dir / f"{cat}.jsonl"
        if not local.exists() or local.stat().st_size == 0:
            url = f"{BBQ_RAW_BASE}/{cat}.jsonl"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    local.write_bytes(r.read())
            except Exception as e:
                print(f"[augment_bbq] {cat} 다운로드 실패: {str(e)[:80]}")
                continue
        for line in local.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    print(f"[augment_bbq] BBQ 로드: {len(rows)}행 ({cache_dir})")
    return rows


def run(config: dict | None = None) -> None:
    cfg = config or load_config()
    target = cfg["target_per_cell"]
    mapped_path = resolve_path(cfg, "mapped")
    mapped_path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if mapped_path.exists() and mapped_path.stat().st_size:
        existing = [json.loads(l) for l in mapped_path.read_text(encoding="utf-8").splitlines() if l.strip()]

    counts: Counter = Counter(cell_key(r) for r in existing)
    image_pool = build_image_pool(existing)
    safe_only = cfg.get("safe_only", False)

    bbq_rows = load_bbq(cfg)
    if not bbq_rows:
        print("[augment_bbq] BBQ 데이터 없음 → 보강 생략")
        return

    rng = derive_rng(cfg["seed"], "augment")
    added = 0
    seen_keys = {r["norm_key"] for r in existing}
    with open(mapped_path, "a", encoding="utf-8") as f:
        for row in bbq_rows:
            rec = map_bbq_row(row)
            if rec is None or rec["norm_key"] in seen_keys:
                continue
            ck = cell_key(rec)
            if counts[ck] >= target:  # 셀 충족 → 스킵
                continue
            # 이미지 결합 (safe_only면 검색 경로지만 풀이 없으면 None 허용)
            if not safe_only:
                rec["image_ref"] = pair_image_for_bbq(rec, image_pool, rng)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[ck] += 1
            seen_keys.add(rec["norm_key"])
            added += 1
    print(f"[augment_bbq] BBQ 보강 +{added}행 (target_per_cell={target})")


if __name__ == "__main__":
    run()
