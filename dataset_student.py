import json
import os
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset
from PIL import Image

from torchvision import transforms


def build_transforms(img_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])


class StudentDistillDataset(Dataset):
    """
    Reads jsonl annotations.

    Expected fields per line:
        image: str
        teacher_probs: list[float], length C
        teacher_score: float
        teacher_conf: float
        is_human_labeled: bool (optional, default False)
        gt_class: int (optional, default -1)
    """

    def __init__(self, ann_path: str, img_size: int = 224, is_train: bool = True):
        super().__init__()
        self.ann_path = ann_path
        self.samples = self._load_jsonl(ann_path)
        self.transform = build_transforms(img_size=img_size, is_train=is_train)

    def _load_jsonl(self, path: str) -> List[Dict[str, Any]]:
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                samples.append(obj)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        img_path = item["image"]
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        teacher_probs = item.get("teacher_probs", None)
        if teacher_probs is None:
            raise ValueError(f"Missing teacher_probs in sample: {img_path}")

        teacher_score = float(item.get("teacher_score", 0.0))
        teacher_conf = float(item.get("teacher_conf", max(teacher_probs)))

        is_human_labeled = bool(item.get("is_human_labeled", False))
        gt_class = int(item.get("gt_class", -1))

        sample = {
            "image": image,
            "teacher_probs": torch.tensor(teacher_probs, dtype=torch.float32),
            "teacher_score": torch.tensor(teacher_score, dtype=torch.float32),
            "teacher_conf": torch.tensor(teacher_conf, dtype=torch.float32),
            "is_human_labeled": torch.tensor(1 if is_human_labeled else 0, dtype=torch.float32),
            "gt_class": torch.tensor(gt_class, dtype=torch.long),
            "image_path": img_path,
        }
        return sample
