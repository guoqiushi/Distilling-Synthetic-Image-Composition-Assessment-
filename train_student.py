import os
import time
import argparse
from typing import Dict

import torch
from torch.utils.data import DataLoader

from dataset_student import StudentDistillDataset
from student_net import MobileNetV3Student
from distill_loss import DistillLoss


def parse_args():
    parser = argparse.ArgumentParser("Train MobileNetV3 student with distillation")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--val_jsonl", type=str, default="")
    parser.add_argument("--save_dir", type=str, default="./student_ckpt")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--lambda_kd", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--lambda_ce", type=float, default=1.0)
    parser.add_argument("--lambda_rank", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--rank_margin", type=float, default=0.05)
    parser.add_argument("--rank_delta", type=float, default=0.10)
    parser.add_argument("--no_conf_weight", action="store_true")

    parser.add_argument("--save_freq", type=int, default=1)
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


def build_dataloader(jsonl_path, img_size, batch_size, num_workers, is_train=True):
    dataset = StudentDistillDataset(
        ann_path=jsonl_path,
        img_size=img_size,
        is_train=is_train,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train,
    )
    return loader


def move_batch_to_device(batch: Dict, device):
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


@torch.no_grad()
def evaluate(model, criterion, val_loader, device):
    model.eval()

    total_loss = 0.0
    total_kd = 0.0
    total_reg = 0.0
    total_ce = 0.0
    total_rank = 0.0
    num_batches = 0

    for batch in val_loader:
        batch = move_batch_to_device(batch, device)
        outputs = model(batch["image"])
        loss_dict = criterion(outputs, batch)

        total_loss += loss_dict["loss"].item()
        total_kd += loss_dict["loss_kd"].item()
        total_reg += loss_dict["loss_reg"].item()
        total_ce += loss_dict["loss_ce"].item()
        total_rank += loss_dict["loss_rank"].item()
        num_batches += 1

    metrics = {
        "val_loss": total_loss / max(1, num_batches),
        "val_kd": total_kd / max(1, num_batches),
        "val_reg": total_reg / max(1, num_batches),
        "val_ce": total_ce / max(1, num_batches),
        "val_rank": total_rank / max(1, num_batches),
    }
    return metrics


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_loader = build_dataloader(
        args.train_jsonl,
        args.img_size,
        args.batch_size,
        args.num_workers,
        is_train=True,
    )

    val_loader = None
    if args.val_jsonl and os.path.exists(args.val_jsonl):
        val_loader = build_dataloader(
            args.val_jsonl,
            args.img_size,
            args.batch_size,
            args.num_workers,
            is_train=False,
        )

    model = MobileNetV3Student(
        num_classes=4,
        embed_dim=256,
        dropout=0.2,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
    )
    model = model.to(device)

    criterion = DistillLoss(
        temperature=args.temperature,
        lambda_kd=args.lambda_kd,
        lambda_reg=args.lambda_reg,
        lambda_ce=args.lambda_ce,
        lambda_rank=args.lambda_rank,
        rank_margin=args.rank_margin,
        rank_delta=args.rank_delta,
        use_conf_weight=not args.no_conf_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()

        epoch_loss = 0.0
        epoch_kd = 0.0
        epoch_reg = 0.0
        epoch_ce = 0.0
        epoch_rank = 0.0
        num_batches = 0

        start = time.time()

        for batch in train_loader:
            batch = move_batch_to_device(batch, device)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                outputs = model(batch["image"])
                loss_dict = criterion(outputs, batch)
                loss = loss_dict["loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss_dict["loss"].item()
            epoch_kd += loss_dict["loss_kd"].item()
            epoch_reg += loss_dict["loss_reg"].item()
            epoch_ce += loss_dict["loss_ce"].item()
            epoch_rank += loss_dict["loss_rank"].item()
            num_batches += 1

        scheduler.step()

        log_msg = (
            f"Epoch [{epoch+1}/{args.epochs}] "
            f"loss={epoch_loss/max(1, num_batches):.4f} "
            f"kd={epoch_kd/max(1, num_batches):.4f} "
            f"reg={epoch_reg/max(1, num_batches):.4f} "
            f"ce={epoch_ce/max(1, num_batches):.4f} "
            f"rank={epoch_rank/max(1, num_batches):.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"time={time.time()-start:.1f}s"
        )

        if val_loader is not None:
            metrics = evaluate(model, criterion, val_loader, device)
            log_msg += (
                f" | val_loss={metrics['val_loss']:.4f}"
                f" val_kd={metrics['val_kd']:.4f}"
                f" val_reg={metrics['val_reg']:.4f}"
                f" val_ce={metrics['val_ce']:.4f}"
                f" val_rank={metrics['val_rank']:.4f}"
            )

            if metrics["val_loss"] < best_val:
                best_val = metrics["val_loss"]
                ckpt_path = os.path.join(args.save_dir, "best_student.pth")
                torch.save({
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "args": vars(args),
                    "best_val": best_val,
                }, ckpt_path)

        print(log_msg)

        if (epoch + 1) % args.save_freq == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pth")
            torch.save({
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "args": vars(args),
            }, ckpt_path)

    final_path = os.path.join(args.save_dir, "last_student.pth")
    torch.save({
        "epoch": args.epochs,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args),
    }, final_path)

    print(f"Training finished. Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
