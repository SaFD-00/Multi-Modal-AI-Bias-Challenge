"""T1 — SB-Bench MCQ → 중간 스키마 매핑 + 이미지 저장.

HF 게이트 데이터셋 ucf-crcv/SB-Bench(CC BY-NC 4.0)를 로드해 mapped.jsonl로 정규화한다.
context_condition이 노출되지 않으므로 휴리스틱으로 ambig 여부를 도출한다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from src.common import (
    LICENSE_SB,
    derive_rng,
    leak_key,
    load_config,
    normalize_surface,
    resolve_path,
)

# SB-Bench/BBQ category → 9축 정규화
_AXIS_MAP = {
    "age": "Age",
    "disability": "Disability_status", "disability_status": "Disability_status",
    "gender": "Gender_identity", "gender_identity": "Gender_identity",
    "nationality": "Nationality",
    "appearance": "Physical_appearance", "physical_appearance": "Physical_appearance",
    "race": "Race_ethnicity", "race_ethnicity": "Race_ethnicity", "ethnicity": "Race_ethnicity",
    "religion": "Religion",
    "ses": "SES", "socioeconomic": "SES", "socioeconomic_status": "SES",
    "sexual_orientation": "Sexual_orientation", "orientation": "Sexual_orientation",
}


def canon_axis(category: str) -> str:
    """category 문자열을 9축 정규 라벨로. 교차축/미상은 'Other'."""
    if not category:
        return "Other"
    key = re.sub(r"[^a-z_]", "", str(category).strip().lower().replace(" ", "_"))
    if key in _AXIS_MAP:
        return _AXIS_MAP[key]
    if "_x_" in key or "cross" in key:  # 교차축
        return "Intersectional"
    for k, v in _AXIS_MAP.items():
        if k in key:
            return v
    return "Other"


def detect_unknown_idx(options: list[str], unknown_lexicon: list[str]) -> int | None:
    """선택지 중 Unknown류 슬롯 인덱스. 표면형 정규화 매칭."""
    surf = {normalize_surface(t) for t in unknown_lexicon}
    for i, o in enumerate(options):
        if normalize_surface(o) in surf:
            return i
    return None


def derive_context_condition(row: dict, label: int, unknown_idx: int) -> tuple[bool, bool]:
    """ambig 여부 도출. 반환: (ambig, inferred).

    우선순위: additional_metadata.context_condition → id 패턴 → label==unknown_idx 추정.
    """
    meta = row.get("additional_metadata") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except Exception:
            meta = {}
    cc = meta.get("context_condition")
    if cc in ("ambig", "ambiguous"):
        return True, False
    if cc in ("disambig", "disambiguated"):
        return False, False
    rid = str(row.get("id", "")).lower()
    if "ambig" in rid and "disambig" not in rid:
        return True, False
    if "disambig" in rid:
        return False, False
    # 최후 추정: 정답이 Unknown이면 ambiguous
    return (label == unknown_idx), True


def map_sbbench_row(row: dict, unknown_lexicon: list[str]) -> dict | None:
    """SB-Bench 한 행 → MappedRecord dict. 매핑 불가 시 None."""
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
    unk_idx = detect_unknown_idx(options, unknown_lexicon)
    if unk_idx is None:
        return None
    ambig, inferred = derive_context_condition(row, label, unk_idx)
    polarity = "neg" if str(row.get("question_polarity", "")).lower().startswith("neg") else "nonneg"
    context = str(row.get("context", ""))
    question = str(row.get("question", ""))
    orig_id = str(row.get("id", "")) or leak_key(context, question)[:12]
    return {
        "uid": f"sb-bench-{orig_id}",
        "source": "sb-bench",
        "license": LICENSE_SB,
        "axis": canon_axis(row.get("category", "")),
        "polarity": polarity,
        "ambig": ambig,
        "context": context,
        "question": question,
        "options": options,
        "label": label,
        "unknown_idx": unk_idx,
        "unknown_text": options[unk_idx],
        "image_ref": None,  # run()에서 이미지 저장 후 채움
        "image_source": None,
        "image_license": None,
        "norm_key": leak_key(context, question),
        "meta": {"orig_id": orig_id, "ambig_inferred": inferred,
                 "sb_category": row.get("category")},
    }


# --- I/O ---

def load_sbbench(cfg: dict):
    """HF 게이트 로그인 후 SB-Bench 'real' split 로드. 실패 시 None.

    'real'(실사 검색 이미지)이 대회 test 분포와 정합. 'synthetic'은 제외.
    category/question_polarity는 ClassLabel 정수이므로 features로 디코딩한다.
    """
    try:
        from huggingface_hub import login
        from datasets import load_dataset

        token = cfg.get("hf_token")
        if token:
            login(token=token, add_to_git_credential=False)
        ds = load_dataset(
            cfg["datasets"]["sb_bench"],
            cache_dir=str(resolve_path(cfg, "sb_bench_dir")),
            token=token,
        )
        split = cfg["datasets"].get("sb_split", "real")
        if split not in ds:
            split = list(ds.keys())[0]
        return ds[split]
    except Exception as e:
        print(f"[map_sbbench] SB-Bench 로드 실패 → 폴백: {e}")
        return None


def decode_sb_row(row: dict, features) -> dict:
    """ClassLabel 정수(category, question_polarity)를 문자열로 디코딩."""
    out = dict(row)
    cat = row.get("category")
    if isinstance(cat, int) and features is not None and "category" in features:
        out["category"] = features["category"].int2str(cat)
    pol = row.get("question_polarity")
    if isinstance(pol, int) and features is not None and "question_polarity" in features:
        out["question_polarity"] = features["question_polarity"].int2str(pol)
    lab = row.get("label")
    if isinstance(lab, int):
        out["label"] = lab  # ClassLabel names ['0','1','2'] → 정수값이 곧 인덱스
    return out


def save_image_pil(img, dst_dir: Path, idx: int) -> str:
    """PIL 이미지를 RGB JPEG로 저장, 상대경로 반환."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    name = f"train_img_{idx:06d}.jpg"
    img.convert("RGB").save(dst_dir / name, format="JPEG", quality=92)
    return f"./images/{name}"


def run(config: dict | None = None) -> None:
    cfg = config or load_config()
    lexicon = cfg["unknown_lexicon"]
    mapped_path = resolve_path(cfg, "mapped")
    images_dir = resolve_path(cfg, "train_images")

    if cfg.get("safe_only"):
        print("[map_sbbench] safe_only=true → SB-Bench 건너뜀")
        mapped_path.parent.mkdir(parents=True, exist_ok=True)
        mapped_path.write_text("", encoding="utf-8")
        return

    ds = load_sbbench(cfg)
    mapped_path.parent.mkdir(parents=True, exist_ok=True)
    features = getattr(ds, "features", None)
    n_ok = n_skip = 0
    with open(mapped_path, "w", encoding="utf-8") as f:
        if ds is None:
            print("[map_sbbench] 데이터 없음 → 빈 mapped.jsonl (augment_bbq가 보강)")
            return
        for i, row in enumerate(ds):
            row = decode_sb_row(row, features)
            rec = map_sbbench_row(row, lexicon)
            if rec is None:
                n_skip += 1
                continue
            img = row.get("file_name")
            if img is not None:
                try:
                    rec["image_ref"] = save_image_pil(img, images_dir, n_ok)
                    rec["image_source"] = "sb-bench"
                    rec["image_license"] = LICENSE_SB
                except Exception as e:
                    print(f"[map_sbbench] 이미지 저장 실패 idx={i}: {e}")
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_ok += 1
    print(f"[map_sbbench] mapped={n_ok}, skipped={n_skip} -> {mapped_path}")


if __name__ == "__main__":
    run()
