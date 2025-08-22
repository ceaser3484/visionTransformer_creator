
import json
from model import ViTClassifier
from dataloader import get_dataloaders
from train import run_train

if __name__ == '__main__':
    cfg = json.load(open("config.json"))

    train_loader, valid_loader = get_dataloaders(
        data_dir=cfg["data_dir"],
        batch_size=cfg.get("batch_size", 16),
        num_workers=cfg.get("num_workers", 8),
        pin_memory=cfg.get("pin_memory", True),
    )

    num_classes = len(getattr(train_loader.dataset, "classes", [])) or cfg.get("num_classes", 1000)
    model = ViTClassifier(
        img_size=cfg.get("img_size", 224),
        patch_size=cfg.get("patch_size", 16),
        num_classes=num_classes,
    )

    best_ckpt, last_ckpt = run_train(
        model,
        train_loader,
        valid_loader,
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 1e-2),
        epochs=cfg.get("epochs", 90),
        early_stop_patience=cfg.get("early_stop_patience", 10),
        grad_clip_norm=cfg.get("grad_clip_norm", None),
        ckpt_dir=cfg.get("ckpt_dir", "./checkpoints"),
        logging=cfg.get("logging", "none"),             # <-- 변경
        log_dir=cfg.get("log_dir", "runs/vit-exp1"),    # <-- 선택
        seed=cfg.get("seed", 42),
    )

