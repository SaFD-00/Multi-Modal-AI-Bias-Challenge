"""test.csv → submission.csv 추론 (대회 1차 제출물 생성).

vLLM 오프라인 추론 + guided JSON 디코딩. 베이스라인 노트북 main() 로직을 CLI로
이식하되, 프롬프트/이미지 전처리는 학습 모듈을 그대로 재사용해 학습-추론 정합을 보장한다
(prompt.build_prompt_text, collator.load_image). vLLM은 멀티모달 LoRA 직접 로드가
어려우므로 src.train.merge로 병합한 체크포인트를 로드한다.

규칙 준수: 최종 답변은 LLM이 JSON을 생성하고 거기서 answer_id만 파싱한다(룰/argmax 아님).
오프라인 실행(HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE)으로 외부 통신을 차단한다.

실행:
    python -m src.predict --model outputs/llava_ov_merged \
        --test-csv data/raw/test/test.csv --images-dir data/raw/test \
        --out output/submission.csv --img-size 224
"""

import argparse
import json
import os
from pathlib import Path

from .common import project_root
from .train.prompt import build_prompt_text

# 베이스라인 run_llava_onevision의 chat 래핑 — 변경 금지(추론 정합).
# 원본 f-string은 im_end 뒤 공백 1 + 줄잇기 들여쓰기 8 = 공백 9칸.
CHAT_PREFIX = "<|im_start|>user <image>\n"
CHAT_SUFFIX = "<|im_end|>" + " " * 9 + "<|im_start|>assistant\n"


def build_chat_prompt(context, question, answers) -> str:
    """학습 프롬프트(build_prompt_text)를 LLaVA-OV chat 템플릿으로 감싼다."""
    return CHAT_PREFIX + build_prompt_text(context, question, answers) + CHAT_SUFFIX


def normalize_answer_id(value) -> str:
    """0/1/2 중 하나만 허용. 그 외/None → "0" (베이스라인 동일)."""
    if value is None:
        return "0"
    text = str(value).strip()
    return text if text in {"0", "1", "2"} else "0"


def extract_answer_id(generated_text) -> str:
    """생성 텍스트에서 JSON을 추출해 answer_id를 파싱. 실패 시 "0"."""
    if not generated_text:
        return "0"
    try:
        start = generated_text.find("{")
        end = generated_text.rfind("}")
        if start >= 0 and end > start:
            parsed = json.loads(generated_text[start:end + 1])
        else:
            parsed = json.loads(generated_text)
        return normalize_answer_id(parsed.get("answer_id"))
    except Exception:  # noqa: BLE001 - 깨진 JSON 등은 안전하게 "0"
        return "0"


def parse_args():
    root = project_root()
    ap = argparse.ArgumentParser(description="vLLM 추론으로 sample_id,label 제출 CSV 생성")
    ap.add_argument("--model", default=str(root / "outputs/llava_ov_merged"),
                    help="병합 체크포인트 디렉터리(src.train.merge 산출물)")
    ap.add_argument("--test-csv", default=str(root / "data/raw/test/test.csv"))
    ap.add_argument("--images-dir", default=str(root / "data/raw/test"),
                    help="image_path가 가리키는 이미지 루트")
    ap.add_argument("--out", default=str(root / "output/submission.csv"))
    ap.add_argument("--img-size", type=int, default=224, help="학습과 동일(224) 유지")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None, help="스모크용 샘플 제한")
    return ap.parse_args()


def main():
    args = parse_args()

    # 오프라인 강제(외부 API/다운로드 차단) — 규칙 준수.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    # 무거운 의존성은 여기서만 import(테스트 환경 보호).
    from typing import Literal

    import pandas as pd
    import torch
    from pydantic import BaseModel
    from tqdm.auto import tqdm
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    from .train.collator import load_image

    class ReasonAnswer(BaseModel):
        reason: str
        answer_id: Literal["0", "1", "2"]

    model_path = Path(args.model)
    if not model_path.exists():
        raise SystemExit(
            f"[predict] 병합 모델 경로 없음: {model_path}. "
            "먼저 `python -m src.train.merge --adapter outputs/llava_ov_lora "
            "--out outputs/llava_ov_merged` 를 실행하세요.")

    df = pd.read_csv(args.test_csv)
    if args.max_samples is not None:
        df = df.head(args.max_samples).copy()
    df["label"] = None

    llm = LLM(
        model=str(model_path),
        max_model_len=16384,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        gpu_memory_utilization=0.9,
        disable_mm_preprocessor_cache=True,
        seed=args.seed,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=128,
        guided_decoding=GuidedDecodingParams(json=ReasonAnswer.model_json_schema()),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inputs, batch_indices = [], []

    def flush():
        if not inputs:
            return
        outputs = llm.generate(inputs, sampling_params=sampling_params, use_tqdm=False)
        for idx, o in zip(batch_indices, outputs):
            df.at[idx, "label"] = extract_answer_id(o.outputs[0].text)
        df[["sample_id", "label"]].to_csv(out_path, index=False, encoding="utf-8")
        inputs.clear()
        batch_indices.clear()

    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="Inference", unit="sample"):
        image = load_image(Path(args.images_dir) / str(row["image_path"]), img_size=args.img_size)
        if image is None:  # 이미지 로드 실패 → 안전하게 "0"
            df.at[row_idx, "label"] = "0"
            continue
        inputs.append({
            "prompt": build_chat_prompt(row["context"], row["question"], row["answers"]),
            "multi_modal_data": {"image": image},
        })
        batch_indices.append(row_idx)
        if len(inputs) >= args.batch_size:
            flush()
    flush()

    # 이미지 누락 행만 있던 경우에도 최종 저장 보장.
    df[["sample_id", "label"]].to_csv(out_path, index=False, encoding="utf-8")
    print(f"[predict] 제출 파일 저장: {out_path} ({len(df)}행, sample_id,label)")


if __name__ == "__main__":
    main()
