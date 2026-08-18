import argparse
import json
import re
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


DEFAULT_PROMPT = (
    "你是图像合成质量评估助手。请根据以下维度给图片打分："
    "1) 内容自洽程度；2) 物体相对大小是否失真；3) 是否出现不规律或不合理物体。"
    "输出 JSON，字段必须包含：score(0-100整数), level(perfect/good/medium/low), "
    "reason(不超过60字)。"
)


def load_image(path: str, image_size: int) -> Image.Image:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    return Image.open(image_path).convert("RGB").resize((image_size, image_size))


def extract_json_block(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\\n", "", text)
        text = re.sub(r"\\n```$", "", text)

    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError(f"No JSON object found in model output: {text}")

    return json.loads(match.group(0))


def normalize_result(result: dict) -> dict:
    score = int(result.get("score", 0))
    score = max(0, min(100, score))
    level = str(result.get("level", "medium")).lower()
    if level not in {"perfect", "good", "medium", "low"}:
        level = "medium"
    reason = str(result.get("reason", ""))

    return {
        "score": score,
        "level": level,
        "reason": reason,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Use Qwen3.5-2B to score synthetic-image consistency.")
    parser.add_argument("--model", default="../models/Qwen3.5-2B", help="HF model path or local model dir")
    parser.add_argument("--image", required=True, help="Path to image")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Scoring prompt")
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map=device_map,
    )

    image = load_image(args.image, args.image_size)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(text=[text], images=[image], return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
        )

    gen_ids = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    try:
        result = normalize_result(extract_json_block(output_text))
    except Exception:
        result = {
            "score": None,
            "level": None,
            "reason": output_text.strip(),
        }

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
