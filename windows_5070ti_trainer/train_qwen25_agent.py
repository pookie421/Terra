"""QLoRA fine-tuning and GGUF export for the RTX 5070 Ti trainer."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping

ROOT = Path(__file__).resolve().parent


def validate_chatml(path: Path, expected_rows: int = 600) -> int:
    if not path.exists():
        raise FileNotFoundError("source-linked ChatML trajectory file is missing: {}".format(path))
    seen = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_id = str(row.get("source_event_id") or "")
            messages = row.get("messages")
            if not source_id or not isinstance(messages, list) or not messages:
                raise ValueError("ChatML row {} lacks a source ID or messages".format(line_number))
            if source_id in seen:
                raise ValueError("duplicate source-linked ChatML trajectory: {}".format(source_id))
            if not any(message.get("role") == "tool" for message in messages):
                raise ValueError("ChatML row {} does not contain a source observation tool result".format(line_number))
            seen.add(source_id)
            count += 1
    if count != expected_rows:
        raise ValueError("expected {} distinct trajectories, found {}".format(expected_rows, count))
    return count


def export_trained_model(model: Any, tokenizer: Any, output_dir: Path, gguf_path: Path) -> Dict[str, Any]:
    """Preserve trained weights before optional conversions; record actual outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter = output_dir / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    manifest = {"adapter": {"status": "saved", "path": "adapter"}}
    for kind in ("merged_16bit", "gguf"):
        try:
            if kind == "merged_16bit":
                model.save_pretrained_merged(str(output_dir / kind), tokenizer, save_method="merged_16bit")
                manifest[kind] = {"status": "saved", "path": kind}
            else:
                # A fresh staging directory prevents stale GGUF files masquerading as a new export.
                import tempfile
                gguf_path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix="gguf_export_", dir=str(output_dir)) as temporary:
                    model.save_pretrained_gguf(temporary, tokenizer, quantization_method="q4_k_m")
                    exports = [p for p in Path(temporary).glob("*.gguf") if "q4_k_m" in p.name.lower()]
                    if not exports:
                        raise RuntimeError("No Q4_K_M GGUF produced")
                    shutil.move(str(exports[0]), str(gguf_path))
                (gguf_path.parent / "Modelfile").write_text("FROM ./{}\nPARAMETER temperature 0.2\n".format(gguf_path.name), encoding="utf-8")
                manifest[kind] = {"status": "saved", "path": gguf_path.name}
        except Exception as exc:
            manifest[kind] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}
            print("{} export failed; trained adapter retained: {}".format(kind, exc))
        (output_dir / "export_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def train_qwen25(dataset_path: Path, output_dir: Path, gguf_path: Path) -> Dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen2.5 QLoRA training requires a visible CUDA GPU; use the Windows RTX trainer.")
    validate_chatml(dataset_path)

    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    dataset = load_dataset("json", data_files={"train": str(dataset_path)}, split="train")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    def render(example: Mapping[str, Any]) -> Dict[str, str]:
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)}

    dataset = dataset.map(render, num_proc=1)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = SFTConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=10,
        num_train_epochs=2,
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        bf16=bool(torch.cuda.is_bf16_supported()),
        fp16=not bool(torch.cuda.is_bf16_supported()),
        max_length=2048,
        dataset_text_field="text",
        dataset_num_proc=1,
        dataloader_num_workers=0,
        packing=False,
        report_to="none",
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=config)
    trainer.train()

    return export_trained_model(model, tokenizer, output_dir, gguf_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(ROOT / "data" / "chatml_trajectories.jsonl"))
    parser.add_argument("--output-dir", default=str(ROOT / "artifacts" / "qwen2_5_7b_lora"))
    parser.add_argument("--gguf", default=str(ROOT / "artifacts" / "terra-shield-qwen-q4_k_m.gguf"))
    args = parser.parse_args()
    train_qwen25(Path(args.dataset), Path(args.output_dir), Path(args.gguf))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

