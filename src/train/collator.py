"""LLaVA-OneVision SFT용 멀티모달 collator.

배치마다 PIL 이미지를 로드하고, processor.apply_chat_template로 (user 프롬프트 + assistant 타깃)
전체 시퀀스를 토크나이즈한 뒤, **assistant 응답 이전 토큰을 -100으로 마스킹**해 프롬프트가 아닌
정답(JSON)만 학습한다. 프롬프트 토큰 수는 동일 입력으로 prompt-only를 한 번 더 토크나이즈해 구한다.

torch/transformers 의존부는 클래스 안에서만 쓰이며, 마스킹 로직(mask_prompt_tokens)과
이미지 로더(load_image)는 그 바깥에서 단위테스트 가능하다.
"""

import base64
from io import BytesIO
from pathlib import Path

from PIL import Image

from .prompt import build_conversation, build_target_json

IGNORE_INDEX = -100


def load_image(image, img_size=224, base_64=False):
    """베이스라인 노트북 load_image 재현(width 기준 리사이즈, RGB, LANCZOS).

    추론과 동일한 전처리를 학습에도 적용해 정합을 유지한다.
    """
    try:
        if isinstance(image, (str, Path)):
            img = Image.open(str(image))
        else:
            img = Image.open(BytesIO(image["bytes"]))
        img = img.convert("RGB")
        width_percent = img_size / float(img.size[0])
        new_height = int((float(img.size[1]) * width_percent))
        img_resized = img.resize((img_size, new_height), Image.LANCZOS)
        if base_64:
            buf = BytesIO()
            img_resized.save(buf, format="JPEG")
            return base64.b64encode(buf.getvalue()).decode("utf-8")
        return img_resized
    except Exception as e:  # noqa: BLE001 - 베이스라인 동작(경로 오류 시 None) 보존
        print(e)
        return None


def mask_prompt_tokens(input_ids, prompt_len, pad_id):
    """labels 생성: 프롬프트 구간([:prompt_len])과 패딩 토큰을 IGNORE_INDEX로 마스킹.

    input_ids: 1D 시퀀스(list 또는 1D tensor-like). 반환은 동일 길이 list.
    assistant 응답 토큰만 손실 계산 대상이 된다.
    """
    labels = [int(t) for t in input_ids]
    for i in range(len(labels)):
        if i < prompt_len or labels[i] == pad_id:
            labels[i] = IGNORE_INDEX
    return labels


class LlavaOVCollator:
    """processor 기반 멀티모달 SFT collator (GPU/transformers 환경에서 동작)."""

    def __init__(self, processor, img_size=224, unknown_lexicon=None, max_length=4096):
        self.processor = processor
        self.img_size = img_size
        self.unknown_lexicon = unknown_lexicon or []
        self.max_length = max_length

    def _encode_one(self, item):
        import torch  # 지연 import (테스트 환경 보호)

        image = load_image(item["image_path"], img_size=self.img_size)
        conv = build_conversation(item["context"], item["question"], item["answers"])
        target = build_target_json(item["answers"], item["label"], self.unknown_lexicon)

        # prompt-only (assistant 생성 직전까지) → 프롬프트 토큰 길이 산출.
        # transformers 4.5x: apply_chat_template은 텍스트만 렌더(tokenize=False)하고,
        # 이미지 바인딩은 processor(text=, images=)로 분리한다(images 인자 중복 회피).
        prompt_text = self.processor.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=False)
        prompt_ids = self.processor(
            text=prompt_text, images=[image], return_tensors="pt")["input_ids"][0]
        prompt_len = int(prompt_ids.shape[0])

        # full (assistant 타깃 포함)
        full_conv = conv + [{"role": "assistant",
                             "content": [{"type": "text", "text": target}]}]
        full_text = self.processor.apply_chat_template(
            full_conv, add_generation_prompt=False, tokenize=False)
        enc = self.processor(text=full_text, images=[image], return_tensors="pt")
        input_ids = enc["input_ids"][0]
        pad_id = self.processor.tokenizer.pad_token_id
        labels = torch.tensor(
            mask_prompt_tokens(input_ids.tolist(), prompt_len, pad_id),
            dtype=torch.long,
        )
        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"][0],
            "labels": labels,
            "pixel_values": enc["pixel_values"][0],
            "image_sizes": enc["image_sizes"][0],
        }

    def __call__(self, batch):
        import torch
        from torch.nn.utils.rnn import pad_sequence

        encoded = [self._encode_one(it) for it in batch]
        pad_id = self.processor.tokenizer.pad_token_id
        input_ids = pad_sequence(
            [e["input_ids"] for e in encoded], batch_first=True, padding_value=pad_id)
        attention_mask = pad_sequence(
            [e["attention_mask"] for e in encoded], batch_first=True, padding_value=0)
        labels = pad_sequence(
            [e["labels"] for e in encoded], batch_first=True, padding_value=IGNORE_INDEX)
        # anyres: 이미지마다 patch 수가 달라(num_patches) 그대로 stack 불가.
        # 프로세서 배치 동작과 동일하게 patch 차원을 배치 최대값으로 zero-padding 후 stack.
        # 모델은 image_sizes로 실제 patch 수를 계산해 패딩 patch를 무시한다.
        pvs = [e["pixel_values"] for e in encoded]
        max_patches = max(p.shape[0] for p in pvs)
        padded_pvs = []
        for p in pvs:
            if p.shape[0] < max_patches:
                pad = torch.zeros((max_patches - p.shape[0], *p.shape[1:]), dtype=p.dtype)
                p = torch.cat([p, pad], dim=0)
            padded_pvs.append(p)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.stack(padded_pvs),
            "image_sizes": torch.stack([e["image_sizes"] for e in encoded]),
        }
