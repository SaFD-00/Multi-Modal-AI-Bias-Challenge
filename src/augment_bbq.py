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
    LICENSE_SB,
    derive_rng,
    leak_key,
    load_config,
    resolve_path,
)
from src.external_images import build_external_pool
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
        "image_source": None,
        "image_license": None,
        "norm_key": leak_key(context, question),
        "meta": {"orig_id": qid, "bbq_category": row.get("category")},
    }


def cell_key(rec: dict) -> tuple[str, bool, str]:
    return (rec["axis"], bool(rec["ambig"]), rec["polarity"])


def cells_needed(current_counts: dict, target_per_cell: int) -> dict:
    """셀별 부족분. target 이상인 셀은 제외."""
    return {k: target_per_cell - v for k, v in current_counts.items() if v < target_per_cell}


def build_image_pool(records: list[dict]) -> dict:
    """기존(SB-Bench) 레코드에서 axis별 이미지 풀 구성. 항목=(image_ref, source, license)."""
    pool: dict[str, list] = defaultdict(list)
    for r in records:
        if r.get("image_ref"):
            pool[r["axis"]].append((
                r["image_ref"],
                r.get("image_source", "sb-bench"),
                r.get("image_license", LICENSE_SB),
            ))
    return pool


def pair_image_for_bbq(
    record: dict, ext_pool: dict, sb_pool: dict, rng: random.Random
) -> tuple | None:
    """BBQ 행에 이미지 결합. (image_ref, source, license) 또는 None 반환.

    우선순위: 동일 axis 외부(FairFace/MMBias) → 동일 axis SB-Bench →
    외부 전체 → SB-Bench 전체. 외부를 앞세워 다양성·라이선스 정화를 동시 달성.
    """
    axis = record["axis"]
    for pool in (ext_pool, sb_pool):
        cands = pool.get(axis)
        if cands:
            return rng.choice(cands)
    for pool in (ext_pool, sb_pool):
        flat = [c for cs in pool.values() for c in cs]
        if flat:
            return rng.choice(flat)
    return None


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
    sb_pool = build_image_pool(existing)
    safe_only = cfg.get("safe_only", False)
    # 외부 이미지 풀(FairFace CC-BY + MMBias MIT)을 한 번만 받아 BBQ 행에 결합.
    ext_pool = {} if safe_only else build_external_pool(cfg, resolve_path(cfg, "train_images"))

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
            # 이미지 결합: 외부(FairFace/MMBias) 우선, SB-Bench 폴백 (safe_only면 생략)
            if not safe_only:
                ref = pair_image_for_bbq(rec, ext_pool, sb_pool, rng)
                if ref:
                    rec["image_ref"], rec["image_source"], rec["image_license"] = ref
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts[ck] += 1
            seen_keys.add(rec["norm_key"])
            added += 1
    print(f"[augment_bbq] BBQ 보강 +{added}행 (target_per_cell={target})")


if __name__ == "__main__":
    run()
