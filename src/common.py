"""공통 유틸: config 로드, 결정적 RNG 파생, 텍스트 정규화, 중간 스키마 상수.

순수 함수(부수효과 없음)와 I/O(config/.env 로드)를 한곳에 모은다. RNG는 항상
인자로 주입해 테스트에서 시드를 고정할 수 있게 한다.
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import unicodedata
from pathlib import Path

import yaml

# 중간 스키마(mapped.jsonl) 필드명 — T1/T2가 쓰고 T3가 읽는 단일 정규형
MAPPED_FIELDS = (
    "uid", "source", "license", "axis", "polarity", "ambig",
    "context", "question", "options", "label", "unknown_idx",
    "unknown_text", "image_ref", "norm_key", "meta",
)

# 대회 제출 CSV 스키마 (test.csv와 동일 + label)
TRAIN_COLUMNS = ("sample_id", "image_path", "context", "question", "answers", "label")

LICENSE_SB = "CC-BY-NC-4.0"
LICENSE_BBQ = "CC-BY-4.0"


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> dict:
    """config.yaml 로드. HF_TOKEN은 .env → 환경변수에서 채운다."""
    root = project_root()
    cfg_path = Path(path) if path else root / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # .env 로드 (있으면). 프로젝트 .env가 정본이므로 쉘 환경변수를 덮어쓴다.
    env_path = root / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=True)
        except Exception:
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip()
    cfg["hf_token"] = os.environ.get("HF_TOKEN")
    cfg["_root"] = str(root)
    return cfg


def resolve_path(cfg: dict, key: str) -> Path:
    """config.paths[key]를 프로젝트 루트 기준 절대경로로 변환."""
    return Path(cfg["_root"]) / cfg["paths"][key]


def _stable_hash(s: str) -> int:
    """프로세스 독립적인 안정 해시(파이썬 hash()는 시드별로 달라 사용 불가)."""
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:4], "big")


def derive_rng(seed: int, stage: str) -> random.Random:
    """마스터 시드에서 단계별 독립 RNG 스트림을 파생.

    한 단계의 변경이 다른 단계 출력에 영향을 주지 않도록 stage 이름으로 분기한다.
    """
    return random.Random((seed ^ _stable_hash(stage)) & 0xFFFFFFFF)


# --- 텍스트 정규화 ---

_WS_RE = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    """누수 제거용 정규화: NFKC → casefold → 공백 단일화 → strip.

    대소문자/공백/유니코드 변종 차이만 있는 문자열을 동일하게 만든다.
    """
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.casefold()
    s = _WS_RE.sub(" ", s).strip()
    return s


def leak_key(context: str, question: str) -> str:
    """(context, question) 조합의 누수 비교 키."""
    payload = normalize_text(context) + "\x1f" + normalize_text(question)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def normalize_surface(s: str) -> str:
    """Unknown 표면형 매칭용 정규화: 아포스트로피 변종 통일 + 소문자 + strip."""
    if s is None:
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = s.replace("’", "'").replace("‘", "'")
    return _WS_RE.sub(" ", s).strip().casefold()
