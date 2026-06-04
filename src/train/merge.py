"""LoRA adapter를 base 모델에 병합해 완결 HF 체크포인트를 저장한다.

vLLM은 멀티모달 LoRA를 직접 로드하기 어려우므로, merge_and_unload로 가중치를 합친 뒤
processor까지 함께 저장한다. 결과 디렉터리를 베이스라인 추론 노트북의
EngineArgs(model="<out>")에 그대로 넣으면 추론이 동작한다.

실행:
    python -m src.train.merge --adapter outputs/llava_ov_lora --out outputs/llava_ov_merged
"""

import argparse

from ..common import project_root


def merge(adapter_dir, out_dir, base_model_id=None):
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

    # adapter에 base 경로가 기록돼 있으면 그대로, 아니면 인자 사용
    if base_model_id is None:
        from peft import PeftConfig
        base_model_id = PeftConfig.from_pretrained(adapter_dir).base_model_name_or_path

    base = LlavaOnevisionForConditionalGeneration.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    model.save_pretrained(out_dir)

    # processor: adapter_dir에 저장돼 있으면 그걸, 없으면 base에서
    try:
        processor = AutoProcessor.from_pretrained(adapter_dir)
    except Exception:
        processor = AutoProcessor.from_pretrained(base_model_id)
    processor.save_pretrained(out_dir)
    print(f"병합 완료 → {out_dir} (베이스라인 노트북 EngineArgs(model='{out_dir}')에 사용)")


def main():
    ap = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    ap.add_argument("--adapter", required=True, help="LoRA adapter 디렉터리(output_dir)")
    ap.add_argument("--out", required=True, help="병합 체크포인트 출력 디렉터리")
    ap.add_argument("--base-model", default=None, help="base 모델 ID(미지정 시 adapter 설정에서 추론)")
    args = ap.parse_args()

    root = project_root()

    def _abs(p):
        from pathlib import Path
        p = Path(p)
        return str(p if p.is_absolute() else root / p)

    merge(_abs(args.adapter), _abs(args.out), args.base_model)


if __name__ == "__main__":
    main()
