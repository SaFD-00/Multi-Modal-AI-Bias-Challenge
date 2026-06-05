"""외부 이미지 소스 — BBQ 행 이미지 다양화 + 라이선스 정화.

FairFace(CC-BY-4.0, age/gender/race 균형 얼굴)와 MMBias(MIT, religion/nationality/
disability/sexual-orientation 실사)를 축별 이미지 풀로 변환한다. 기존엔 BBQ 텍스트 행에
동일 axis SB-Bench(CC-BY-NC) 이미지를 재사용(14,578장→수만 행, 중복 큼)했는데, 외부 풀로
대체하면 (1) 이미지 다양성↑ (2) OOD 축(Religion/Sexual_orientation) 이미지 확보
(3) BBQ 행을 CC-BY/MIT만으로 구성(NC 의존 제거 = 라이선스 정화)한다.

반환 풀 형식: {axis: [(image_ref, source, license), ...]}.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

from src.common import LICENSE_FAIRFACE, LICENSE_MMBIAS

# FairFace 1장은 age+gender+race를 모두 가지므로 아래 4축 공용 얼굴 풀로 쓴다.
FAIRFACE_AXES = ("Age", "Gender_identity", "Race_ethnicity", "Intersectional")

# MMBias zip 이름(공백 포함) → 9축 정규 라벨. "Valence"는 인물/개념 자극이라 제외.
MMBIAS_ZIPS = {
    "Religion": "Religion",
    "Sexual Orientation": "Sexual_orientation",
    "Nationality": "Nationality",
    "Disability": "Disability_status",
}
MMBIAS_RAW_BASE = "https://raw.githubusercontent.com/sepehrjng92/MMBias/main/data/Images"
_IMG_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def _save_jpeg(img, images_dir: Path, name: str) -> str:
    """PIL 이미지를 RGB JPEG로 저장하고 train.csv 상대경로(./images/...) 반환."""
    images_dir.mkdir(parents=True, exist_ok=True)
    img.convert("RGB").save(images_dir / name, format="JPEG", quality=92)
    return f"./images/{name}"


def load_mmbias(cfg: dict, images_dir: Path) -> dict:
    """MMBias 축별 zip을 받아 이미지 저장. {axis: [(rel, 'mmbias', MIT)]} 반환."""
    from PIL import Image

    cache = Path(cfg["_root"]) / cfg["paths"]["mmbias_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    pool: dict[str, list] = {}
    idx = 0
    for zip_stem, axis in MMBIAS_ZIPS.items():
        local = cache / f"{zip_stem}.zip"
        if not local.exists() or local.stat().st_size == 0:
            url = f"{MMBIAS_RAW_BASE}/{zip_stem.replace(' ', '%20')}.zip"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                local.write_bytes(urllib.request.urlopen(req, timeout=180).read())
            except Exception as e:
                print(f"[external] MMBias {zip_stem} 다운로드 실패: {str(e)[:80]}")
                continue
        entries: list = []
        with zipfile.ZipFile(local) as z:
            for n in z.namelist():
                if n.startswith("__MACOSX") or n.endswith("/"):
                    continue
                if not n.lower().endswith(_IMG_EXT):
                    continue
                try:
                    with z.open(n) as fh:
                        img = Image.open(io.BytesIO(fh.read()))
                        rel = _save_jpeg(img, images_dir, f"mmbias_{idx:06d}.jpg")
                except Exception:
                    continue
                entries.append((rel, "mmbias", LICENSE_MMBIAS))
                idx += 1
        if entries:
            pool[axis] = entries
        print(f"[external] MMBias {axis}: {len(entries)}장")
    return pool


def load_fairface(cfg: dict, images_dir: Path) -> dict:
    """FairFace 얼굴을 저장해 age/gender/race/intersectional 공용 풀 구성.

    {axis: [(rel, 'fairface', CC-BY)]} 반환 (FAIRFACE_AXES가 동일 리스트 공유).
    """
    from datasets import load_dataset

    ex = cfg.get("external_images", {}) or {}
    cap = int(ex.get("fairface_max", 10000))
    conf = str(ex.get("fairface_config", "0.25"))
    cache = Path(cfg["_root"]) / cfg["paths"]["fairface_dir"]
    try:
        ds = load_dataset(
            cfg["datasets"]["fairface"], conf, split="validation",
            cache_dir=str(cache),
        )
    except Exception as e:
        print(f"[external] FairFace 로드 실패: {str(e)[:120]}")
        return {}
    entries: list = []
    n = min(cap, len(ds))
    for i in range(n):
        try:
            rel = _save_jpeg(ds[i]["image"], images_dir, f"fairface_{i:06d}.jpg")
        except Exception:
            continue
        entries.append((rel, "fairface", LICENSE_FAIRFACE))
    print(f"[external] FairFace: {len(entries)}장 ({conf} validation)")
    return {axis: list(entries) for axis in FAIRFACE_AXES}


def build_external_pool(cfg: dict, images_dir: Path) -> dict:
    """FairFace + MMBias 통합 축별 풀. external_images.enabled=false면 {}."""
    if not (cfg.get("external_images", {}) or {}).get("enabled", False):
        return {}
    pool: dict[str, list] = {}
    for sub in (load_fairface(cfg, images_dir), load_mmbias(cfg, images_dir)):
        for axis, entries in sub.items():
            pool.setdefault(axis, []).extend(entries)
    total = sum(len(v) for v in pool.values())
    print(f"[external] 통합 풀: {total}장, 축={ {k: len(v) for k, v in pool.items()} }")
    return pool
