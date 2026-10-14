"""Opt-in QLoRA training. Fails closed on unsupported hardware or missing evidence data."""

import argparse
import json
import logging
from pathlib import Path


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("train_qlora")


def check_hardware(min_vram_gb: float = 16.0) -> tuple[object, object]:
    try:
        import torch
        import bitsandbytes  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("Install the optional training dependencies from requirements-ml.txt in Linux/WSL.") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA needs a supported CUDA GPU; CPU training is intentionally disabled for this 8B workflow.")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024 ** 3)
    if vram_gb < min_vram_gb:
        raise RuntimeError(f"GPU has {vram_gb:.1f} GB VRAM; this script requires at least {min_vram_gb:.0f} GB for a conservative Qwen3-8B run.")
    if props.major < 7:
        raise RuntimeError(f"GPU compute capability {props.major}.{props.minor} is below the supported baseline for this 4-bit training setup.")
    return torch, props


def load_rows(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not all(row.get(key) for key in ("question", "context", "answer")):
            raise ValueError(f"{path}:{line_no}: question, context, and answer are required.")
        if not (row.get("source_url") or row.get("source_document") or row.get("abstention_reason")):
            raise ValueError(f"{path}:{line_no}: add source provenance or abstention_reason.")
        rows.append(row)
    if not rows:
        raise ValueError(f"No training records found in {path}.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("training/data/splits"))
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--output-dir", type=Path, default=Path("training/output/qwen3-8b-legal-adapter"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--min-vram-gb", type=float, default=16.0)
    args = parser.parse_args()

    torch, props = check_hardware(args.min_vram_gb)
    train_rows = load_rows(args.data_dir / "train.jsonl")
    validation_rows = load_rows(args.data_dir / "validation.jsonl")
    test_rows = load_rows(args.data_dir / "test.jsonl")
    logger.info("GPU=%s VRAM_GB=%.1f train=%s validation=%s test=%s", props.name, props.total_memory / (1024**3), len(train_rows), len(validation_rows), len(test_rows))

    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, DataCollatorForLanguageModeling, Trainer, TrainingArguments

    def as_text(row: dict) -> dict:
        evidence = row.get("evidence") or ""
        return {"text": f"### Context\n{row['context']}\n\n### Evidence provenance\n{evidence}\n\n### Question\n{row['question']}\n\n### Answer\n{row['answer']}"}

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    quantization = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quantization, device_map="auto", trust_remote_code=True)
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    max_length = 2048

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_length, padding="max_length")

    datasets = {}
    for name, rows in (("train", train_rows), ("validation", validation_rows), ("test", test_rows)):
        dataset = Dataset.from_list([as_text(row) for row in rows])
        datasets[name] = dataset.map(tokenize, batched=True, remove_columns=["text"])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=str(args.output_dir), num_train_epochs=args.epochs, learning_rate=2e-4,
        per_device_train_batch_size=1, per_device_eval_batch_size=1, gradient_accumulation_steps=8,
        gradient_checkpointing=True, fp16=True, logging_steps=10, save_strategy="epoch",
        eval_strategy="epoch", report_to=[], load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )
    train_result = trainer.train()
    validation_metrics = trainer.evaluate(datasets["validation"], metric_key_prefix="validation")
    test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    (args.output_dir / "training_metrics.json").write_text(json.dumps({"train": train_result.metrics, "validation": validation_metrics, "test": test_metrics}, indent=2), encoding="utf-8")
    logger.info("Adapter and measured loss metrics saved to %s", args.output_dir)
    logger.warning("Loss metrics are not answer correctness or legal accuracy. Review predictions with a source-checked evaluation set.")


if __name__ == "__main__":
    main()
