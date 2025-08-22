# train.py (FP32, 모델/로더 외부 주입 버전)

import os, time, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


# -----------------------------
# 유틸: 시드/체크포인트
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_ckpt(path, epoch, model, optimizer, best_val_loss):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }, path)

def load_ckpt_if_exists(path, model, optimizer=None, map_location="cpu"):
    if os.path.isfile(path):
        ckpt = torch.load(path, map_location=map_location)
        model.load_state_dict(ckpt["model_state"])
        if optimizer is not None and "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", np.inf)
        print(f"[Resume] Loaded {path} (epoch={start_epoch-1})")
        return start_epoch, best_val_loss
    return 0, np.inf


# -----------------------------
# 학습/검증 루프 (FP32)
# -----------------------------
def train_one_epoch(model, loader, optimizer, device, criterion, grad_clip_norm=None):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Train", leave=False)

    for X, y in pbar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        correct      += (logits.argmax(1) == y).sum().item()
        total        += y.size(0)
        pbar.set_postfix(loss=running_loss/max(total,1), acc=correct/max(total,1))

    return running_loss/total, correct/total


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="Valid", leave=False)

    for X, y in pbar:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(X)
        loss = criterion(logits, y)

        running_loss += loss.item() * X.size(0)
        correct      += (logits.argmax(1) == y).sum().item()
        total        += y.size(0)
        pbar.set_postfix(loss=running_loss/max(total,1), acc=correct/max(total,1))

    return running_loss/total, correct/total


# -----------------------------
# 메인: 학습 실행 (모델/로더를 외부에서 주입)
# -----------------------------
def run_train(
    model: torch.nn.Module,
    train_loader,
    valid_loader,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    epochs: int = 90,
    early_stop_patience: int = 10,
    grad_clip_norm: float | None = None,
    ckpt_dir: str = "./checkpoints",
    use_tb: bool = False,
    seed: int = 42,
):
    set_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    # (옵션) 스케줄러
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scheduler = None

    # 체크포인트/로그
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt = os.path.join(ckpt_dir, "best.pt")
    last_ckpt = os.path.join(ckpt_dir, "last.pt")
    start_epoch, best_val_loss = load_ckpt_if_exists(last_ckpt, model, optimizer, map_location="cpu")

    writer = None
    if use_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="runs/vit-exp1")

    # 에폭 루프
    patience = 0
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, criterion, grad_clip_norm=grad_clip_norm
        )
        val_loss, val_acc = evaluate(model, valid_loader, device, criterion)

        if scheduler is not None:
            scheduler.step()

        print(f"[{epoch+1:03d}/{epochs}] "
              f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"{time.time()-t0:.1f}s")

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val",   val_loss,   epoch)
            writer.add_scalar("Acc/train",  train_acc,  epoch)
            writer.add_scalar("Acc/val",    val_acc,    epoch)

        save_ckpt(last_ckpt, epoch, model, optimizer, best_val_loss)  # 항상 저장

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            save_ckpt(best_ckpt, epoch, model, optimizer, best_val_loss)
            patience = 0
        else:
            patience += 1

        if patience >= early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} (best val_loss={best_val_loss:.4f})")
            break

    if writer:
        writer.close()

    print("학습 종료. 베스트 체크포인트:", best_ckpt)
    return best_ckpt, last_ckpt
