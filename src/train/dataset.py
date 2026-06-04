"""train.csv 로드 + 결정적 95/5 split + Dataset.

torch가 없는 환경에서도 split 로직을 단위테스트할 수 있도록, torch.utils.data.Dataset
의존부는 지연 import한다(load_rows/split_train_val은 순수 파이썬).
"""

import csv
import json
from pathlib import Path

from ..common import derive_rng


def load_rows(train_csv) -> list[dict]:
    """train.csv → dict 리스트. 컬럼: sample_id,image_path,context,question,answers,label."""
    rows = []
    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def load_metadata(meta_path) -> dict[str, dict]:
    """metadata.jsonl → {sample_id: meta}. axis/polarity/source 등 분할 기준 회수용."""
    meta = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                meta[d["sample_id"]] = d
    return meta


def split_train_val(rows, seed, val_ratio=0.05):
    """시드 고정 셔플 후 (train, val) 분할. 동일 시드 → 동일 분할.

    sample_id 정렬로 입력 순서 영향을 제거한 뒤 derive_rng로 셔플한다.
    """
    ordered = sorted(rows, key=lambda r: r["sample_id"])
    rng = derive_rng(seed, "train_split")
    rng.shuffle(ordered)
    n_val = int(len(ordered) * val_ratio)
    val = ordered[:n_val]
    train = ordered[n_val:]
    return train, val


def split_train_val_ood(rows, meta, seed, val_ratio=0.05, ood_axes=()):
    """leave-axis-out 3분할. (train, in_domain_val, ood_val) 반환.

    ood_axes에 속한 bias 축의 행을 통째로 OOD 검증셋으로 hold-out하고, 나머지에서
    기존 split_train_val로 train / in-domain-val을 나눈다. 학습에서 안 본 편향 축에 대한
    일반화를 측정해 Private(Shake-up) 위험을 진단한다.

    ood_axes가 비면 ood_val=[]이고 (train, in_domain_val)은 split_train_val과 동일(하위호환).
    """
    ood_set = set(ood_axes)
    ood_val, rest = [], []
    for r in rows:
        axis = meta.get(r["sample_id"], {}).get("axis")
        (ood_val if axis in ood_set else rest).append(r)
    ood_val.sort(key=lambda r: r["sample_id"])  # 입력 순서 무관하게 결정적
    train, in_val = split_train_val(rest, seed, val_ratio)
    return train, in_val, ood_val


def resolve_image_path(images_dir, image_path) -> Path:
    """image_path('./images/xxx.jpg')를 images_dir 기준 절대경로로 변환."""
    rel = str(image_path).lstrip("./")
    return Path(images_dir) / rel


class BiasVQADataset:
    """행 리스트를 감싸는 Dataset. __getitem__은 원본 행 dict + 해석된 이미지 경로 반환.

    이미지 로드/토크나이즈는 collator에서 수행(지연 로드)한다.
    torch.utils.data.Dataset을 상속하지만 import 실패 시에도 동작하도록 동적 처리.
    """

    def __init__(self, rows, images_dir):
        self.rows = rows
        self.images_dir = images_dir

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        return {
            "context": r["context"],
            "question": r["question"],
            "answers": r["answers"],
            "label": int(r["label"]),
            "image_path": resolve_image_path(self.images_dir, r["image_path"]),
        }
