"""train.csv 로드 + 결정적 95/5 split + Dataset.

torch가 없는 환경에서도 split 로직을 단위테스트할 수 있도록, torch.utils.data.Dataset
의존부는 지연 import한다(load_rows/split_train_val은 순수 파이썬).
"""

import csv
from pathlib import Path

from ..common import derive_rng


def load_rows(train_csv) -> list[dict]:
    """train.csv → dict 리스트. 컬럼: sample_id,image_path,context,question,answers,label."""
    rows = []
    with open(train_csv, "r", encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


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
