import json
import os
from datasets import Dataset
from PIL import Image

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


MODEL_NAME = "../models/Qwen3.5-0.8B"
TRAIN_JSONL = "sample_imgs/train.jsonl"
VAL_JSONL = "sample_imgs/val.jsonl"
OUTPUT_DIR = "./outputs/qwen35_08b_lora_cls"

IMAGE_SIZE = 512

PROMPT = (
    "This is a simulated image, not a real photograph. "
    "Please do not judge it too strictly by photo realism. "
    "Evaluate the semantic and spatial consistency of the image. "
    "Output only one label: good, medium, or low."
)


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_rgb_image(image_path: str, image_size: int = 512) -> Image.Image:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB").resize((image_size, image_size))


print("start")
print("loading train/val jsonl...")
train_raw = load_jsonl(TRAIN_JSONL)
val_raw = load_jsonl(VAL_JSONL)

train_dataset = Dataset.from_list(train_raw)
val_dataset = Dataset.from_list(val_raw)

print(f"train samples: {len(train_dataset)}")
print(f"val samples: {len(val_dataset)}")

print("loading processor...")
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
)

print("loading model...")
model = AutoModelForImageTextToText.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

'''
def collate_fn(examples):
    texts = []
    images = []

    for ex in examples:
        messages = ex["messages"]
        if len(messages) < 2:
            raise ValueError(f"Invalid sample, expected 2 messages, got: {len(messages)}")

        user_msg = messages[0]
        assistant_msg = messages[1]

        image_path = None
        other_user_items = []

        for item in user_msg["content"]:
            if item["type"] == "image":
                image_path = item["image"]
            else:
                other_user_items.append(item)

        if image_path is None:
            raise ValueError("No image found in user message.")

        image = load_rgb_image(image_path, IMAGE_SIZE)

        # 只让 user 部分进入 chat template，避免 assistant 部分重复引入图像占位
        user_only_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    *other_user_items,
                ],
            }
        ]

        prompt_text = processor.apply_chat_template(
            user_only_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        label_text = assistant_msg["content"][0]["text"].strip()
        full_text = prompt_text + label_text

        texts.append(full_text)
        images.append(image)

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch
'''
def collate_fn(examples):
    texts = []
    images = []

    for ex in examples:
        messages = ex["messages"]
        if len(messages) < 2:
            raise ValueError(f"Invalid sample, expected 2 messages, got {len(messages)}")

        user_msg = messages[0]
        assistant_msg = messages[1]

        image_path = None
        user_texts = []

        for item in user_msg["content"]:
            if item["type"] == "image":
                image_path = item["image"]
            elif item["type"] == "text":
                user_texts.append(item["text"])

        if image_path is None:
            raise ValueError("No image found in user message.")

        image = load_rgb_image(image_path, IMAGE_SIZE)
        label_text = assistant_msg["content"][0]["text"].strip()

        # 这里不要把真实图片对象塞进 messages，只保留 image placeholder
        template_messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": " ".join(user_texts)},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": label_text},
                ],
            },
        ]

        text = processor.apply_chat_template(
            template_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        texts.append(text)
        images.append(image)

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        return_tensors="pt",
    )

    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=1e-4,
    num_train_epochs=3,
    warmup_steps=20,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    report_to="none",
    remove_unused_columns=False,
    dataset_text_field="",
    dataset_kwargs={"skip_prepare_dataset": True},
    dataloader_num_workers=0,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
)

print("start train")
trainer.train()

print("saving model...")
trainer.model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)
print("done")
