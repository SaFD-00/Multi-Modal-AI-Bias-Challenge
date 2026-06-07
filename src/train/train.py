"""다중 VLM(family) SFT 학습 진입점.

지원 family: llava_ov / qwen2_5_vl / mimo_vl (src.train.models.MODEL_REGISTRY). 모델 클래스·
processor·비전 freeze·LoRA 타깃은 레지스트리가 정본이며, 이 스크립트는 --model {family}로 선택한다.
configs/train.yaml(공통)을 base로 로드한 뒤 --config(모드별 파일)로 덮어쓴다. 비전타워/멀티모달
projector는 모드와 무관하게 freeze하고 LLM만 학습한다. finetune_type=lora면 LLM에 adapter를,
full이면 LLM 전체를 학습한다. WANDB로 train/eval loss를 모니터링한다(키 없으면 tensorboard 폴백).

출력(family/모드별 자동 산출, src.train.paths):
    lora → outputs/{family}/adapters/lora    (merge로 outputs/{family}/merged/lora 생성)
    full → outputs/{family}/merged/full       (완결 모델, 병합 불필요)

실행:
    python -m src.train.train --model qwen2_5_vl --config configs/train_lora.yaml   # LoRA
    python -m src.train.train --model llava_ov  --config configs/train_full.yaml    # full
    python -m src.train.train --config configs/train_lora.yaml --max-samples 64 --no-wandb  # 스모크
"""

import argparse
import os
from pathlib import Path

import yaml

from ..common import project_root
from . import paths
from .collator import VLMCollator
from .dataset import (
    BiasVQADataset,
    load_metadata,
    load_rows,
    split_train_val,
    split_train_val_ood,
)
from .models import DEFAULT_FAMILY


def load_train_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _abs(root, p):
    p = Path(p)
    return p if p.is_absolute() else root / p


def setup_wandb(cfg, use_wandb):
    """WANDB 사용 가능하면 report_to=['wandb'], 아니면 tensorboard로 폴백.

    WANDB_API_KEY/WANDB_PROJECT는 .env 또는 환경변수에서 읽는다(load_config 패턴).
    """
    if not use_wandb:
        return ["tensorboard"]
    root = project_root()
    env_path = root / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path, override=False)
        except Exception:
            pass
    if not os.environ.get("WANDB_API_KEY"):
        print("INFO: WANDB_API_KEY 없음 → tensorboard로 폴백. wandb를 쓰려면 .env에 키를 넣으세요.")
        return ["tensorboard"]
    os.environ.setdefault("WANDB_PROJECT", cfg.get("wandb_project", "skku-bias"))
    return ["wandb"]


def build_model_and_processor(cfg, family):
    import torch

    from . import models

    # family별 모델/processor 로드 + 비전타워/projector freeze (LoRA/full 모드 모두 LLM만 학습).
    model, processor = models.load_model_and_processor(family, bf16=cfg.get("bf16", True))

    if cfg.get("finetune_type", "lora") == "lora":
        from peft import LoraConfig, get_peft_model
        # lora_target_modules는 config override가 있으면 우선, 없으면 레지스트리 기본값.
        targets = cfg.get("lora_target_modules") or models.lora_targets(family)
        lora = LoraConfig(
            r=cfg["lora_r"],
            lora_alpha=cfg["lora_alpha"],
            lora_dropout=cfg["lora_dropout"],
            target_modules=targets,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    # full 모드: LoRA 미적용 → freeze 안 된 LLM 파라미터가 requires_grad=True로 학습된다.

    # gradient_checkpointing + 동결 임베딩 조합에서 입력이 grad를 전파하도록 활성화
    # (없으면 "element 0 of tensors does not require grad" 발생).
    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
    # NaN-guard: 특정 배치가 bf16 overflow로 grad에 nan/inf를 내도 그 값만 유한값으로 치환한다
    # (해당 step을 사실상 skip). max_grad_norm은 nan>1.0=False라 발산을 못 막으므로, 첫 오염이
    # weight로 전파돼 loss=0/grad_norm=nan으로 영구 고착되는 연쇄를 여기서 끊는다(full 모드에서 더 중요).
    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4))
    # print_trainable_parameters는 PEFT 전용 → full 모드는 수동 카운트로 폴백.
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
    else:
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {n_tr:,} || all params: {n_all:,} || trainable%: {100 * n_tr / n_all:.4f}")
    return model, processor


def main():
    ap = argparse.ArgumentParser(description="다중 VLM SFT (LoRA/full)")
    ap.add_argument("--model", default=None,
                    help="모델 family (llava_ov|qwen2_5_vl|mimo_vl). 미지정 시 config의 model, 없으면 llava_ov")
    ap.add_argument("--config", default="configs/train_lora.yaml",
                    help="모드별 override (configs/train.yaml 공통 base 위에 덮어씀)")
    ap.add_argument("--max-samples", type=int, default=None, help="스모크용 샘플 제한")
    ap.add_argument("--no-wandb", action="store_true", help="wandb 비활성(tensorboard 폴백)")
    args = ap.parse_args()

    from transformers import Trainer, TrainingArguments

    root = project_root()
    # 공통 base(train.yaml)를 로드한 뒤 모드별 파일(--config)로 얕은 override.
    cfg = load_train_config(root / "configs" / "train.yaml")
    cfg.update(load_train_config(_abs(root, args.config)))

    # 모델 family 결정: CLI > config(model) > 기본(llava_ov).
    family = args.model or cfg.get("model") or DEFAULT_FAMILY
    finetune_type = cfg.get("finetune_type", "lora")
    # 출력 경로는 family + 모드로 자동 산출(lora=adapters/lora, full=merged/full).
    out_dir = paths.adapter_dir(family) if finetune_type == "lora" else paths.merged_dir(family, "full")
    run_name = cfg.get("run_name") or f"{family}-{finetune_type}"

    # GPU 프로파일 런처(src.train.launch)가 .env(GPU_TYPE/GPU_COUNT) 기반으로 주입한 override.
    # 직접 실행 시엔 env 미설정 → yaml 값 그대로 사용(하위호환). global batch는 런처가 32로 맞춘다.
    if os.environ.get("TRAIN_PER_DEVICE_BATCH"):
        b = int(os.environ["TRAIN_PER_DEVICE_BATCH"])
        cfg["per_device_train_batch_size"] = b
        cfg["per_device_eval_batch_size"] = b
    if os.environ.get("TRAIN_ACCUM"):
        cfg["gradient_accumulation_steps"] = int(os.environ["TRAIN_ACCUM"])
    if os.environ.get("TRAIN_BF16"):
        cfg["bf16"] = os.environ["TRAIN_BF16"] == "1"

    # 데이터 레벨 설정(unknown_lexicon, ood_axes, metadata 경로)은 configs/data.yaml이 정본.
    unknown_lexicon, ood_axes, meta_rel = [], [], None
    cfg_yaml = root / "configs" / "data.yaml"
    if cfg_yaml.exists():
        data_cfg = yaml.safe_load(open(cfg_yaml))
        unknown_lexicon = data_cfg.get("unknown_lexicon", [])
        ood_axes = data_cfg.get("ood_axes") or []
        meta_rel = data_cfg.get("paths", {}).get("metadata")

    rows = load_rows(_abs(root, cfg["train_csv"]))
    if args.max_samples:
        rows = rows[: args.max_samples]
    images_dir = _abs(root, cfg["images_dir"])

    # OOD 검증셋: ood_axes 지정 + metadata 존재 시 leave-axis-out 3분할.
    # eval_dataset을 {"in": ..., "ood": ...} dict로 주면 Trainer가 eval_in_loss/eval_ood_loss를
    # 각각 로깅하고, metric_for_best_model="eval_ood_loss"로 OOD 기준 best를 고른다.
    meta_path = _abs(root, meta_rel) if meta_rel else None
    if ood_axes and meta_path and meta_path.exists():
        meta = load_metadata(meta_path)
        train_rows, in_rows, ood_rows = split_train_val_ood(
            rows, meta, seed=cfg["seed"], val_ratio=cfg["val_ratio"], ood_axes=ood_axes
        )
        train_ds = BiasVQADataset(train_rows, images_dir)
        eval_dataset = {
            "in": BiasVQADataset(in_rows, images_dir),
            "ood": BiasVQADataset(ood_rows, images_dir),
        }
        best_metric = "eval_ood_loss"
        print(f"train={len(train_rows)} in_val={len(in_rows)} ood_val={len(ood_rows)} (ood_axes={ood_axes})")
    else:
        train_rows, val_rows = split_train_val(rows, seed=cfg["seed"], val_ratio=cfg["val_ratio"])
        train_ds = BiasVQADataset(train_rows, images_dir)
        eval_dataset = BiasVQADataset(val_rows, images_dir)
        best_metric = "eval_loss"
        print(f"train={len(train_rows)} val={len(val_rows)}")

    model, processor = build_model_and_processor(cfg, family)
    collator = VLMCollator(
        processor, family=family, img_size=cfg["img_size"],
        unknown_lexicon=unknown_lexicon, max_length=cfg["max_length"],
    )

    report_to = setup_wandb(cfg, use_wandb=not args.no_wandb)

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=cfg["num_train_epochs"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        max_grad_norm=cfg.get("max_grad_norm", 1.0),  # gradient clipping (발산 완화)
        weight_decay=cfg["weight_decay"],
        lr_scheduler_type=cfg["lr_scheduler_type"],
        logging_steps=cfg["logging_steps"],
        bf16=cfg.get("bf16", True),
        gradient_checkpointing=cfg.get("gradient_checkpointing", True),
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=best_metric,   # OOD 활성 시 eval_ood_loss 기준 best
        greater_is_better=False,
        seed=cfg["seed"],
        report_to=report_to,
        run_name=run_name,
        remove_unused_columns=False,
        dataloader_num_workers=4,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()
    trainer.save_model(str(out_dir))
    processor.save_pretrained(str(out_dir))
    if finetune_type == "lora":
        merged = paths.merged_dir(family, "lora")
        print(f"LoRA adapter 저장: {out_dir}  → `python -m src.train.merge --family {family}`로 "
              f"{merged}에 병합하세요.")
    else:
        print(f"Full 모델 저장: {out_dir}  → 병합 불필요. 바로 추론/eval에 사용하세요.")


if __name__ == "__main__":
    main()
