"""
tesla_evaluate.py — End-to-End Evaluation Script.

Runs each component (camera detector, LiDAR classifier, fusion pipeline)
on test data, prints a structured report, and optionally saves JSON results.

Usage:
  python tesla_evaluate.py --help
  python tesla_evaluate.py --camera-only  --images data/images
  python tesla_evaluate.py --lidar-only   --lidar-dir data/lidar_test
  python tesla_evaluate.py --fusion       --images data/images --lidar-dir data/lidar_test
  python tesla_evaluate.py --benchmark    --frames 100
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _banner(title: str, width: int = 56) -> None:
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_latency(stats: dict, label: str = "") -> None:
    prefix = f"  [{label}] " if label else "  "
    print(
        f"{prefix}mean={stats['mean_ms']:.1f}ms  "
        f"p50={stats['p50_ms']:.1f}  "
        f"p90={stats['p90_ms']:.1f}  "
        f"p99={stats['p99_ms']:.1f}  "
        f"fps={stats['fps']:.1f}"
    )


# ---------------------------------------------------------------------------
# Camera-only evaluation
# ---------------------------------------------------------------------------


def evaluate_camera(
    images_dir: str = "data/images",
    model_path: str = "yolov5s.pt",
    num_frames: int = 30,
    save_json: str | None = None,
) -> dict:
    from tesla_detector import TeslaStyleDetector

    _banner("Camera Detection Evaluation (YOLOv5)")
    detector = TeslaStyleDetector(model_path=model_path)

    img_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    if not img_paths:
        print(f"  No images found in {images_dir}")
        return {}

    import cv2

    all_dets: list[dict] = []
    class_counts: dict[str, int] = {}

    for p in img_paths[:10]:  # cap for quick eval
        img = cv2.imread(str(p))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        summary = detector.detection_summary(img_rgb)
        all_dets.extend(summary["detections"])
        for d in summary["detections"]:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

    lat_stats = detector.benchmark_latency(
        image_paths=[str(p) for p in img_paths[:2]],
        num_frames=num_frames,
    )

    print("\n  Class distribution across evaluated frames:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:16s}: {cnt}")

    result = {"latency": lat_stats, "class_counts": class_counts, "total_detections": len(all_dets)}

    if save_json:
        Path(save_json).write_text(json.dumps(result, indent=2))
        print(f"\n  Results saved to {save_json}")

    return result


# ---------------------------------------------------------------------------
# LiDAR-only evaluation
# ---------------------------------------------------------------------------


def evaluate_lidar(
    lidar_dir: str = "data/lidar_test",
    weights: str = "pointnet_tesla.pt",
    num_points: int = 1024,
    save_json: str | None = None,
) -> dict:
    from pointnet_classifier import LiDARClassifier

    _banner("LiDAR Classification Evaluation (PointNet)")
    clf = LiDARClassifier(weights=weights, num_points=num_points)

    npy_files = list(Path(lidar_dir).glob("*.npy")) if Path(lidar_dir).exists() else []
    if not npy_files:
        print(f"  No .npy files in {lidar_dir}; running on synthetic data.")
        npy_files = []

    latencies: list[float] = []
    class_counts: dict[str, int] = {}
    all_confs: list[float] = []

    def _eval_pts(pts: np.ndarray) -> None:
        t0 = time.perf_counter()
        dets = clf.classify_cloud(pts)
        latencies.append((time.perf_counter() - t0) * 1000)
        for d in dets:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1
            all_confs.append(d["confidence"])

    if npy_files:
        for p in npy_files:
            pts = np.load(str(p))
            _eval_pts(pts)
    else:
        # Synthetic: 10 random clouds
        for _ in range(10):
            pts = np.random.randn(2048, 3).astype(np.float32)
            _eval_pts(pts)

    arr = np.array(latencies) if latencies else np.array([0.0])
    lat_stats = {
        "frames": len(arr),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "fps": float(1000.0 / max(np.mean(arr), 1e-3)),
    }

    _print_latency(lat_stats, "LiDAR classify")

    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
    print(f"\n  Mean detection confidence: {avg_conf:.3f}")
    print("  Class distribution:")
    for cls, cnt in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"    {cls:14s}: {cnt}")

    result = {
        "latency": lat_stats,
        "class_counts": class_counts,
        "mean_confidence": avg_conf,
    }

    if save_json:
        Path(save_json).write_text(json.dumps(result, indent=2))
        print(f"\n  Results saved to {save_json}")

    return result


# ---------------------------------------------------------------------------
# Fusion evaluation
# ---------------------------------------------------------------------------


def evaluate_fusion(
    images_dir: str = "data/images",
    lidar_dir: str = "data/lidar_test",
    model_path: str = "yolov5s.pt",
    weights: str = "pointnet_tesla.pt",
    num_frames: int = 20,
    save_json: str | None = None,
) -> dict:
    import cv2

    from pointnet_classifier import LiDARClassifier
    from sensor_fusion import CameraDetection, CameraLiDARCalibration, LiDARDetection, SensorFusion
    from tesla_detector import TESLA_COCO_CLASSES, TeslaStyleDetector

    _banner("Sensor Fusion Evaluation (Camera + LiDAR)")

    detector = TeslaStyleDetector(model_path=model_path)
    clf = LiDARClassifier(weights=weights)
    calib = CameraLiDARCalibration()
    fusion = SensorFusion(calib=calib)

    img_paths = list(Path(images_dir).glob("*.jpg")) + list(Path(images_dir).glob("*.png"))
    npy_files = list(Path(lidar_dir).glob("*.npy")) if Path(lidar_dir).exists() else []

    latencies: list[float] = []
    source_counts = {"fused": 0, "camera_only": 0, "lidar_only": 0}
    total_dets = 0

    for i in range(min(num_frames, max(len(img_paths), 5))):
        # Camera
        cam_dets: list[CameraDetection] = []
        if img_paths:
            img = cv2.imread(str(img_paths[i % len(img_paths)]))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                df = detector.process_frame(img_rgb)
                for _, row in df.iterrows():
                    cid = int(row["class"])
                    cam_dets.append(
                        CameraDetection(
                            class_name=TESLA_COCO_CLASSES.get(cid, "unknown"),
                            class_id=cid,
                            confidence=float(row["confidence"]),
                            bbox=np.array([row["xmin"], row["ymin"], row["xmax"], row["ymax"]]),
                        )
                    )

        # LiDAR
        lidar_dets: list[LiDARDetection] = []
        if npy_files:
            pts = np.load(str(npy_files[i % len(npy_files)]))
            raw = clf.classify_cloud(pts)
        else:
            pts = np.random.randn(2048, 3).astype(np.float32)
            raw = clf.classify_cloud(pts)

        for d in raw:
            if d["class_name"] == "background":
                continue
            lidar_dets.append(
                LiDARDetection(
                    class_name=d["class_name"],
                    class_id=d["class_id"],
                    confidence=d["confidence"],
                    centroid=np.asarray(d["centroid"]),
                    num_points=d.get("num_points", 0),
                )
            )

        t0 = time.perf_counter()
        result = fusion.fuse(cam_dets, lidar_dets)
        elapsed = (time.perf_counter() - t0) * 1000
        latencies.append(elapsed)

        total_dets += len(result)
        for d in result:
            source_counts[d.source] = source_counts.get(d.source, 0) + 1

    arr = np.array(latencies) if latencies else np.array([0.0])
    lat_stats = {
        "frames": len(arr),
        "mean_ms": float(np.mean(arr)),
        "p50_ms": float(np.percentile(arr, 50)),
        "p90_ms": float(np.percentile(arr, 90)),
        "p99_ms": float(np.percentile(arr, 99)),
        "fps": float(1000.0 / max(np.mean(arr), 1e-3)),
    }

    _print_latency(lat_stats, "fusion step")
    print(f"\n  Total fused detections   : {total_dets}")
    print("  Source breakdown:")
    for src, cnt in source_counts.items():
        print(f"    {src:<16}: {cnt}")

    result_dict = {"latency": lat_stats, "source_counts": source_counts, "total_detections": total_dets}

    if save_json:
        Path(save_json).write_text(json.dumps(result_dict, indent=2))
        print(f"\n  Results saved to {save_json}")

    return result_dict


# ---------------------------------------------------------------------------
# Full benchmark report
# ---------------------------------------------------------------------------


def run_full_benchmark(
    images_dir: str = "data/images",
    lidar_dir: str = "data/lidar_test",
    frames: int = 50,
    save_dir: str | None = None,
) -> None:
    _banner("Tesla Sensor Portfolio — Full Benchmark", width=60)
    print(f"  Camera images : {images_dir}")
    print(f"  LiDAR data    : {lidar_dir}")
    print(f"  Frames        : {frames}")

    results: dict = {}

    try:
        results["camera"] = evaluate_camera(
            images_dir=images_dir,
            num_frames=frames,
            save_json=f"{save_dir}/camera.json" if save_dir else None,
        )
    except Exception as e:
        print(f"  Camera eval failed: {e}")

    try:
        results["lidar"] = evaluate_lidar(
            lidar_dir=lidar_dir,
            save_json=f"{save_dir}/lidar.json" if save_dir else None,
        )
    except Exception as e:
        print(f"  LiDAR eval failed: {e}")

    _banner("Summary", width=56)
    for component, stats in results.items():
        lat = stats.get("latency", {})
        if lat:
            _print_latency(lat, component)

    if save_dir and results:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        (Path(save_dir) / "full_report.json").write_text(json.dumps(results, indent=2))
        print(f"\n  Full report → {save_dir}/full_report.json")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Tesla Sensor Portfolio — Evaluation")
    p.add_argument("--camera-only", action="store_true")
    p.add_argument("--lidar-only", action="store_true")
    p.add_argument("--fusion", action="store_true")
    p.add_argument("--benchmark", action="store_true", help="Full benchmark of all components")
    p.add_argument("--images", default="data/images")
    p.add_argument("--lidar-dir", default="data/lidar_test")
    p.add_argument("--model", default="yolov5s.pt")
    p.add_argument("--weights", default="pointnet_tesla.pt")
    p.add_argument("--frames", type=int, default=30)
    p.add_argument("--save-json", default=None)
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()

    if args.camera_only:
        evaluate_camera(args.images, args.model, args.frames, args.save_json)
    elif args.lidar_only:
        evaluate_lidar(args.lidar_dir, args.weights, save_json=args.save_json)
    elif args.fusion:
        evaluate_fusion(args.images, args.lidar_dir, args.model, args.weights, args.frames, args.save_json)
    else:
        run_full_benchmark(args.images, args.lidar_dir, args.frames, args.save_dir)
