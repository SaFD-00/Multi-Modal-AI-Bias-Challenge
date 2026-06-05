"""LLaVA-OneVision LoRA 학습 진입점 (A100 80GB x 1).

베이스라인과 동일한 모델을 LoRA로 fine-tuning한다. 비전타워/멀티모달 projector는 freeze하고
LLM에만 adapter를 붙인다. WANDB로 train/eval loss를 실시간 모니터링한다(키 없으면 tensorboard 폴백).

실행:
    python -m src.train.train --config configs/train_lora.yaml
    python -m src.train.train --config configs/train_lora.yaml --max-samples 64 --no-wandb  # 스모크
학습 후 LoRA adapter는 output_dir에 저장 → src.train.merge로 base에 병합해 추론에 사용.
"""

import argparse
import os
from pathlib import Path

import yaml

from ..common import project_root
from .collator import LlavaOVCollator
from .dataset import (
    BiasVQADataset,
    load_metadata,
    load_rows,
    split_train_val,
    split_train_val_ood,
)


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


def build_model_and_processor(cfg):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    processor = AutoProcessor.from_pretrained(cfg["model_id"])
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        cfg["model_id"],
        torch_dtype=torch.bfloat16 if cfg.get("bf16", True) else torch.float32,
    )

    # 비전타워 + 멀티모달 projector freeze (LoRA는 LLM에만)
    for name, p in model.named_parameters():
        if "vision_tower" in name or "multi_modal_projector" in name:
            p.requires_grad = False

    lora = LoraConfig(
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["lora_target_modules"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # gradient_checkpointing + 동결 임베딩 조합에서 입력이 grad를 전파하도록 활성화
    # (없으면 "element 0 of tensors does not require grad" 발생).
    if cfg.get("gradient_checkpointing", True):
        model.enable_input_require_grads()
    # NaN-guard: 특정 배치가 bf16 overflow로 grad에 nan/inf를 내도 그 값만 유한값으로 치환한다
    # (해당 step을 사실상 skip). max_grad_norm은 nan>1.0=False라 발산을 못 막으므로, 첫 오염이
    # weight로 전파돼 loss=0/grad_norm=nan으로 영구 고착되는 연쇄를 여기서 끊는다.
    for p in model.parameters():
        if p.requires_grad:
            p.register_hook(lambda g: torch.nan_to_num(g, nan=0.0, posinf=1e4, neginf=-1e4))
    model.print_trainable_parameters()
    return model, processor


def main():
    ap = argparse.ArgumentParser(description="LLaVA-OneVision LoRA SFT")
    ap.add_argument("--config", default="configs/train_lora.yaml")
    ap.add_argument("--max-samples", type=int, default=None, help="스모크용 샘플 제한")
    ap.add_argument("--no-wandb", action="store_true", help="wandb 비활성(tensorboard 폴백)")
    args = ap.parse_args()

    from transformers import Trainer, TrainingArguments

    root = project_root()
    cfg = load_train_config(_abs(root, args.config))

    # 데이터 레벨 설정(unknown_lexicon, ood_axes, metadata 경로)은 config.yaml이 정본.
    unknown_lexicon, ood_axes, meta_rel = [], [], None
    cfg_yaml = root / "config.yaml"
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

    model, processor = build_model_and_processor(cfg)
    collator = LlavaOVCollator(
        processor, img_size=cfg["img_size"],
        unknown_lexicon=unknown_lexicon, max_length=cfg["max_length"],
    )

    report_to = setup_wandb(cfg, use_wandb=not args.no_wandb)

    targs = TrainingArguments(
        output_dir=str(_abs(root, cfg["output_dir"])),
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
        run_name=cfg.get("run_name"),
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
    trainer.save_model(str(_abs(root, cfg["output_dir"])))
    processor.save_pretrained(str(_abs(root, cfg["output_dir"])))
    print(f"LoRA adapter 저장: {cfg['output_dir']}  → src.train.merge로 병합하세요.")


if __name__ == "__main__":
    main()
