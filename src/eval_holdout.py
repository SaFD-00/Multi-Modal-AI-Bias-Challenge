"""학습 모델을 held-out(in-domain val / OOD)에 대해 실제 accuracy로 평가.

train.py와 동일한 leave-axis-out 분할(같은 seed/val_ratio/ood_axes)을 재현해, 학습에
쓰지 않은 in_val·ood 행의 정답(label)을 vLLM 추론과 비교한다. 대회 지표(accuracy)로
in-domain vs OOD 일반화 갭을 보고, epoch1/epoch2 체크포인트를 직접 비교한다.

프롬프트·이미지·JSON 파싱은 predict.py를 재사용(학습-추론 정합). merge된 모델이 필요하다.

실행:
    # epoch별 adapter를 먼저 병합한 뒤(merge는 base에 LoRA를 합쳐 vLLM 로드용 체크포인트 생성)
    python -m src.train.merge --adapter outputs/llava_ov_lora/checkpoint-1215 --out outputs/eval_ep1
    CUDA_VISIBLE_DEVICES=1 python -m src.eval_holdout --model outputs/eval_ep1 --split both
"""

import argparse
import os
from pathlib import Path

import yaml

from .common import project_root
from .predict import build_chat_prompt, extract_answer_id
from .train.dataset import (
    load_metadata,
    load_rows,
    resolve_image_path,
    split_train_val_ood,
)


def _abs(root, p):
    p = Path(p)
    return p if p.is_absolute() else root / p


def load_splits(root):
    """train.py와 동일 분할 재현 → ({"in": in_val, "ood": ood}, images_dir, img_size, meta)."""
    train_cfg = yaml.safe_load(open(_abs(root, "configs/train_lora.yaml")))
    data_cfg = yaml.safe_load(open(root / "config.yaml"))
    ood_axes = data_cfg.get("ood_axes") or []
    meta = load_metadata(_abs(root, data_cfg["paths"]["metadata"]))
    rows = load_rows(_abs(root, train_cfg["train_csv"]))
    _, in_val, ood = split_train_val_ood(
        rows, meta, seed=train_cfg["seed"],
        val_ratio=train_cfg["val_ratio"], ood_axes=ood_axes,
    )
    images_dir = _abs(root, train_cfg["images_dir"])
    return {"in": in_val, "ood": ood}, images_dir, train_cfg["img_size"], meta


def build_engine(model_path, seed):
    """merge된 체크포인트로 vLLM 엔진 + guided-JSON SamplingParams 생성(predict.py와 동일 설정)."""
    from typing import Literal

    import torch
    from pydantic import BaseModel
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    class ReasonAnswer(BaseModel):
        reason: str
        answer_id: Literal["0", "1", "2"]

    if not Path(model_path).exists():
        raise SystemExit(
            f"[eval] 병합 모델 경로 없음: {model_path}. "
            "src.train.merge로 체크포인트를 먼저 병합하세요.")
    llm = LLM(
        model=str(model_path), max_model_len=16384,
        limit_mm_per_prompt={"image": 1},
        tensor_parallel_size=max(1, torch.cuda.device_count()),
        gpu_memory_utilization=0.9, disable_mm_preprocessor_cache=True, seed=seed,
    )
    sp = SamplingParams(
        temperature=0.0, max_tokens=128,
        guided_decoding=GuidedDecodingParams(json=ReasonAnswer.model_json_schema()),
    )
    return llm, sp


def infer(llm, sp, rows, images_dir, img_size, batch_size):
    """rows에 대해 답 인덱스("0"/"1"/"2") 예측 리스트 반환. 이미지 로드 실패 행은 "0"."""
    from tqdm.auto import tqdm

    from .train.collator import load_image

    preds = ["0"] * len(rows)
    inputs, idxs = [], []

    def flush():
        if not inputs:
            return
        outs = llm.generate(inputs, sampling_params=sp, use_tqdm=False)
        for i, o in zip(idxs, outs):
            preds[i] = extract_answer_id(o.outputs[0].text)
        inputs.clear()
        idxs.clear()

    for i, r in enumerate(tqdm(rows, desc="eval", unit="sample")):
        img = load_image(resolve_image_path(images_dir, r["image_path"]), img_size=img_size)
        if img is None:
            continue  # preds[i]는 기본값 "0"
        inputs.append({
            "prompt": build_chat_prompt(r["context"], r["question"], r["answers"]),
            "multi_modal_data": {"image": img},
        })
        idxs.append(i)
        if len(inputs) >= batch_size:
            flush()
    flush()
    return preds


def report(name, rows, preds, meta):
    """전체 accuracy + ambig(모호/명확)별 accuracy 출력.

    ambig=True 행의 정답률은 모호 context에서 'unknown' 선택지를 고른 비율(unknown 회수율)에
    해당 — bias 회피 성능의 핵심 신호. 전체 accuracy만 보면 편향 회귀를 놓친다(AGENTS 규칙3).
    """
    n = len(rows)
    correct = 0
    amb = {True: [0, 0], False: [0, 0]}
    for r, p in zip(rows, preds):
        ok = int(p == str(r["label"]).strip())
        correct += ok
        a = meta.get(r["sample_id"], {}).get("ambig")
        if a in amb:
            amb[a][0] += ok
            amb[a][1] += 1
    acc = correct / n if n else 0.0
    print(f"[eval] split={name:3s} n={n:5d}  accuracy={acc:.4f}")
    for a, label in [(True, "ambiguous"), (False, "disambiguated")]:
        c, t = amb[a]
        if t:
            print(f"         {label:13s} n={t:5d}  acc={c / t:.4f}")
    return acc


def main():
    ap = argparse.ArgumentParser(description="held-out(in/OOD) accuracy 평가")
    ap.add_argument("--model", required=True, help="병합 체크포인트(src.train.merge 산출물)")
    ap.add_argument("--split", choices=["in", "ood", "both"], default="both")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-samples", type=int, default=None, help="스모크용 샘플 제한")
    args = ap.parse_args()

    # 오프라인 강제(외부 다운로드 차단) — predict.py와 동일.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    root = project_root()
    splits, images_dir, img_size, meta = load_splits(root)
    targets = ["in", "ood"] if args.split == "both" else [args.split]

    llm, sp = build_engine(args.model, args.seed)
    results = {}
    for name in targets:
        rows = splits[name]
        if args.max_samples:
            rows = rows[: args.max_samples]
        preds = infer(llm, sp, rows, images_dir, img_size, args.batch_size)
        results[name] = report(name, rows, preds, meta)

    if "in" in results and "ood" in results:
        print(f"[eval] 일반화 갭 (in - ood) = {results['in'] - results['ood']:+.4f}")


if __name__ == "__main__":
    main()
