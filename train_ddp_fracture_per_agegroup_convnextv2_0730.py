import os
import pandas as pd
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, WeightedRandomSampler, DistributedSampler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as transforms
from timm import create_model
import mlflow
from unified_dataset_0704 import UnifiedDataset

# ✅ DDP 초기화
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="nccl")
DEVICE = torch.device("cuda", local_rank)

# ✅ 경로 설정
BASE_DIR = os.path.dirname(__file__)

# ✅ 설정
BATCH_SIZE = 12
EPOCHS = 100
IMG_SIZE = 384
LR = 1e-4
MODEL_NAME = "convnextv2_large.fcmae_ft_in22k_in1k_384"

def AGE_GROUP_FN(age):
    age = float(age)
    if age < 5: return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else: return 3

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

for proj in ["AP", "Lat"]:
    if local_rank == 0:
        print(f"\n🚀 Start Training Projection: {proj}")

    train_csv = os.path.join(BASE_DIR, f"age_train_tmp_{proj}.csv")
    val_csv = os.path.join(BASE_DIR, f"age_val_tmp_{proj}.csv")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_all = pd.concat([df_train, df_val]).reset_index(drop=True)

    df_all["label"] = df_all["fracture_visible"].fillna(0).astype(float)
    df_all["age_group_label"] = df_all["age"].astype(float).apply(AGE_GROUP_FN)

    for age_group in [0, 1, 2, 3]:
        if local_rank == 0:
            print(f"\n🔹 Training {proj}_{age_group}...")

        df_g = df_all[df_all["age_group_label"] == age_group].copy()

        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        for train_idx, val_idx in splitter.split(df_g, df_g["label"]):
            df_train_g = df_g.iloc[train_idx]
            df_val_g = df_g.iloc[val_idx]

        train_dataset = UnifiedDataset(df_train_g, transform=transform, task="fracture_only")
        val_dataset = UnifiedDataset(df_val_g, transform=transform, task="fracture_only")

        train_labels = df_train_g["label"].tolist()
        count_0, count_1 = train_labels.count(0.0), train_labels.count(1.0)
        weight_0, weight_1 = 1.0 / (count_0 + 1e-6), 1.0 / (count_1 + 1e-6)
        sample_weights = [weight_0 if l == 0.0 else weight_1 for l in train_labels]

        weighted_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=weighted_sampler, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

        model = create_model(MODEL_NAME, pretrained=True, num_classes=1).to(DEVICE)
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

        # Loss
        if count_1 == 0:
            criterion = nn.BCEWithLogitsLoss()
        else:
            imbalance_ratio = max(count_0, count_1) / (min(count_0, count_1) + 1e-6)
            if imbalance_ratio > 5.0 and min(count_0, count_1) >= 50:
                criterion = FocalLoss(alpha=0.25, gamma=2.0)
            else:
                pos_weight = torch.tensor([count_0 / (count_1 + 1e-6)]).to(DEVICE)
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        if local_rank == 0:
            run_name = f"ConvNeXtV2_{proj}_Age{age_group}"
            mlflow.start_run(run_name=run_name)
            best_f1 = 0.0

        for epoch in range(EPOCHS):
            model.train()
            total_loss, preds, labels = 0, [], []

            for images, targets in train_loader:
                images, targets = images.to(DEVICE), targets.to(DEVICE).float().view(-1)
                optimizer.zero_grad()
                outputs = model(images).squeeze(dim=-1)
                loss = criterion(outputs, targets)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                total_loss += loss.item()
                probs = torch.sigmoid(outputs).detach().cpu().numpy()
                preds += (probs > 0.5).astype(int).tolist()
                labels += targets.cpu().numpy().astype(int).tolist()

            if local_rank == 0:
                acc = accuracy_score(labels, preds)
                f1 = f1_score(labels, preds)
                print(f"[Train] {proj}_{age_group} Epoch {epoch} | Loss: {total_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f}")
                mlflow.log_metric("train_acc", acc, step=epoch)
                mlflow.log_metric("train_f1", f1, step=epoch)

                # Validation
                model.eval()
                val_preds, val_labels = [], []
                with torch.no_grad():
                    for images, targets in val_loader:
                        images = images.to(DEVICE)
                        targets = targets.to(DEVICE).float().view(-1)
                        outputs = model(images).squeeze(dim=-1)
                        probs = torch.sigmoid(outputs).cpu().numpy()
                        val_preds += (probs > 0.5).astype(int).tolist()
                        val_labels += targets.cpu().numpy().astype(int).tolist()

                val_acc = accuracy_score(val_labels, val_preds)
                val_f1 = f1_score(val_labels, val_preds)
                print(f"[Val] {proj}_{age_group} Epoch {epoch} | Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("val_f1", val_f1, step=epoch)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    model_path = os.path.join(BASE_DIR, f"best_ddp_convnextv2_{proj}_{age_group}.pt")
                    torch.save(model.module.state_dict(), model_path)
                    mlflow.log_artifact(model_path)
                    print(f"✅ Saved: {model_path}")

        if local_rank == 0:
            mlflow.end_run()

# ✅ 종료
dist.destroy_process_group()
