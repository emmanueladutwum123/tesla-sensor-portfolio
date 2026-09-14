"""
PointNet LiDAR Point Cloud Classifier — Tesla Sensor Portfolio.

Architecture: Qi et al., "PointNet: Deep Learning on Point Sets for 3D
Classification and Segmentation" (CVPR 2017).

Tesla-relevant classes:
  0 car  1 pedestrian  2 cyclist  3 truck  4 cone  5 background
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

TESLA_LIDAR_CLASSES = ["car", "pedestrian", "cyclist", "truck", "cone", "background"]


# ---------------------------------------------------------------------------
# Sub-networks
# ---------------------------------------------------------------------------


class TNet(nn.Module):
    """Spatial transformer network: predicts a k×k alignment matrix."""

    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: (B, k, N)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.max(dim=2)[0]  # global max pool → (B, 1024)
        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)  # (B, k*k)

        # initialize to identity
        eye = torch.eye(self.k, device=x.device).view(1, self.k * self.k).repeat(x.size(0), 1)
        x = x + eye
        return x.view(-1, self.k, self.k)


class PointNetEncoder(nn.Module):
    """Shared MLP encoder with two T-Nets; returns global feature (B, 1024)."""

    def __init__(self, global_feat: bool = True):
        super().__init__()
        self.global_feat = global_feat
        self.tnet3 = TNet(k=3)
        self.tnet64 = TNet(k=64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)

    def forward(self, x):
        # x: (B, 3, N)
        _B, _, N = x.shape

        # Input transform
        t3 = self.tnet3(x)
        x = torch.bmm(t3, x)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        # Feature transform
        t64 = self.tnet64(x)
        x = torch.bmm(t64, x)
        local_feat = x  # (B, 64, N) — kept for seg head

        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.bn5(self.conv5(x))  # (B, 1024, N)

        global_feat = x.max(dim=2)[0]  # (B, 1024)

        if self.global_feat:
            return global_feat, t64
        return torch.cat([local_feat, global_feat.unsqueeze(2).repeat(1, 1, N)], dim=1), t64


class PointNetClassifier(nn.Module):
    """PointNet for point cloud classification."""

    def __init__(self, num_classes: int = len(TESLA_LIDAR_CLASSES), dropout: float = 0.3):
        super().__init__()
        self.encoder = PointNetEncoder(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x):
        # x: (B, N, 3) → transpose to (B, 3, N)
        x = x.transpose(2, 1)
        feat, tmat = self.encoder(x)
        x = F.relu(self.bn1(self.fc1(feat)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1), tmat


# ---------------------------------------------------------------------------
# Regularization loss
# ---------------------------------------------------------------------------


def feature_transform_regularizer(tmat: torch.Tensor) -> torch.Tensor:
    """||I - A·Aᵀ||_F loss to keep the feature transform close to orthogonal."""
    B, K, _ = tmat.shape
    I = torch.eye(K, device=tmat.device).unsqueeze(0).expand(B, -1, -1)
    diff = I - torch.bmm(tmat, tmat.transpose(2, 1))
    return (diff**2).sum(dim=(1, 2)).mean()


# ---------------------------------------------------------------------------
# Dataset helper
# ---------------------------------------------------------------------------


class PointCloudDataset(torch.utils.data.Dataset):
    """Expects .npy files, each shaped (N, 3) with a corresponding _label.txt.

    Directory layout:
        root/
          class_name/
            sample_0001.npy
            ...
    """

    def __init__(self, root: str, num_points: int = 1024, augment: bool = True):
        self.root = Path(root)
        self.num_points = num_points
        self.augment = augment
        self.class_map = {c: i for i, c in enumerate(TESLA_LIDAR_CLASSES)}
        self.samples = []

        for class_dir in self.root.iterdir():
            if class_dir.is_dir() and class_dir.name in self.class_map:
                label = self.class_map[class_dir.name]
                for f in class_dir.glob("*.npy"):
                    self.samples.append((f, label))

    def __len__(self):
        return len(self.samples)

    def _sample_points(self, pts: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        if n >= self.num_points:
            idx = np.random.choice(n, self.num_points, replace=False)
        else:
            idx = np.random.choice(n, self.num_points, replace=True)
        return pts[idx]

    def _augment(self, pts: np.ndarray) -> np.ndarray:
        # Random rotation about the up-axis (z)
        theta = np.random.uniform(0, 2 * np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float32)
        pts = pts @ R.T
        # Jitter
        pts += np.random.randn(*pts.shape).astype(np.float32) * 0.02
        return pts

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        pts = np.load(path).astype(np.float32)
        # Normalize to unit sphere
        pts -= pts.mean(axis=0)
        dist = np.linalg.norm(pts, axis=1).max()
        if dist > 0:
            pts /= dist
        pts = self._sample_points(pts)
        if self.augment:
            pts = self._augment(pts)
        return torch.from_numpy(pts), label


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    data_root: str = "data/lidar",
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    num_points: int = 1024,
    save_path: str = "pointnet_tesla.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = PointCloudDataset(data_root, num_points=num_points, augment=True)
    n_val = max(1, int(0.15 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = PointNetClassifier().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        # -- train --
        model.train()
        total_loss = correct = total = 0
        for pts, labels in train_loader:
            pts, labels = pts.to(device), labels.to(device)
            optimizer.zero_grad()
            preds, tmat = model(pts)
            loss = F.nll_loss(preds, labels) + 0.001 * feature_transform_regularizer(tmat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (preds.argmax(1) == labels).sum().item()
            total += len(labels)
        scheduler.step()

        # -- validate --
        model.eval()
        val_correct = val_total = 0
        with torch.no_grad():
            for pts, labels in val_loader:
                pts, labels = pts.to(device), labels.to(device)
                preds, _ = model(pts)
                val_correct += (preds.argmax(1) == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total if val_total else 0.0
        print(
            f"Epoch {epoch:3d}/{epochs}  loss={total_loss / total:.4f}  "
            f"train_acc={correct / total:.3f}  val_acc={val_acc:.3f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  → saved (val_acc={val_acc:.3f})")

    print(f"\nBest validation accuracy: {best_acc:.3f}")
    return model


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------


class LiDARClassifier:
    """Thin inference wrapper around a trained PointNetClassifier. Clusters an incoming point cloud (numpy N×3) into
    segments and returns per-segment (class_name, confidence, centroid) triples.
    """

    def __init__(self, weights: str = "pointnet_tesla.pt", num_points: int = 1024):
        self.num_points = num_points
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = PointNetClassifier().to(self.device)
        if Path(weights).exists():
            self.model.load_state_dict(torch.load(weights, map_location=self.device))
            print(f"Loaded PointNet weights from {weights}")
        else:
            print(f"[LiDARClassifier] No weights at {weights}; running with random init")
        self.model.eval()

    def _cluster(self, pts: np.ndarray, voxel_size: float = 2.0):
        """Naive voxel-based clustering — one voxel, one segment for simplicity."""
        if len(pts) == 0:
            return []
        # Quantise into voxels
        keys = (pts[:, :3] / voxel_size).astype(int)
        voxels: dict = {}
        for i, k in enumerate(map(tuple, keys)):
            voxels.setdefault(k, []).append(i)
        return [pts[idxs] for idxs in voxels.values() if len(idxs) >= 10]

    @torch.no_grad()
    def classify_cloud(self, pts: np.ndarray, voxel_size: float = 2.0):
        """
        Args:
            pts: (N, 3+) array of LiDAR points (only first 3 columns used).

        Returns:
            list of dicts: {class_name, class_id, confidence, centroid, num_points}.
        """
        segments = self._cluster(pts[:, :3], voxel_size)
        results = []
        for seg in segments:
            centroid = seg.mean(axis=0)
            # Normalize
            seg = seg - seg.mean(axis=0)
            d = np.linalg.norm(seg, axis=1).max()
            if d > 0:
                seg /= d
            # Sample
            n = seg.shape[0]
            if n >= self.num_points:
                idx = np.random.choice(n, self.num_points, replace=False)
            else:
                idx = np.random.choice(n, self.num_points, replace=True)
            seg = seg[idx].astype(np.float32)

            tensor = torch.from_numpy(seg).unsqueeze(0).to(self.device)
            logits, _ = self.model(tensor)
            probs = logits.exp().squeeze()
            cls_id = int(probs.argmax())
            conf = float(probs[cls_id])
            results.append(
                {
                    "class_name": TESLA_LIDAR_CLASSES[cls_id],
                    "class_id": cls_id,
                    "confidence": conf,
                    "centroid": centroid,
                    "num_points": n,
                }
            )
        return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PointNet LiDAR classifier — Tesla Sensor Portfolio")
    sub = parser.add_subparsers(dest="cmd")

    tr = sub.add_parser("train", help="Train on a labeled point cloud dataset")
    tr.add_argument("--data", default="data/lidar", help="Dataset root")
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--batch-size", type=int, default=32)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--num-points", type=int, default=1024)
    tr.add_argument("--save", default="pointnet_tesla.pt")

    inf = sub.add_parser("infer", help="Classify a .npy point cloud file")
    inf.add_argument("file", help="Path to .npy point cloud (N×3+)")
    inf.add_argument("--weights", default="pointnet_tesla.pt")

    demo = sub.add_parser("demo", help="Run on a synthetic point cloud")

    args = parser.parse_args()

    if args.cmd == "train":
        train(
            data_root=args.data,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_points=args.num_points,
            save_path=args.save,
        )

    elif args.cmd == "infer":
        clf = LiDARClassifier(weights=args.weights)
        pts = np.load(args.file)
        dets = clf.classify_cloud(pts)
        print(f"\nDetections in {args.file}:")
        for d in dets:
            print(
                f"  {d['class_name']:12s}  conf={d['confidence']:.3f}  "
                f"centroid=({d['centroid'][0]:.1f}, {d['centroid'][1]:.1f}, {d['centroid'][2]:.1f})  "
                f"pts={d['num_points']}"
            )

    else:  # demo / no command
        print("Demo: classifying a synthetic 'car-like' point cloud (no trained weights).")
        # Box-shaped point cloud roughly resembling a car
        pts = np.random.uniform([-2, -1, 0], [2, 1, 1.5], size=(2048, 3)).astype(np.float32)
        clf = LiDARClassifier(weights="pointnet_tesla.pt")
        dets = clf.classify_cloud(pts, voxel_size=10.0)
        for d in dets:
            print(f"  {d['class_name']:12s}  conf={d['confidence']:.3f}")
