# tesla_detector.py - Tesla-Style YOLOv5 Object Detector
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

# Tesla-relevant COCO class IDs and their names
TESLA_COCO_CLASSES: dict[int, str] = {
    0:  "person",
    2:  "car",
    3:  "motorcycle",
    5:  "bus",
    7:  "truck",
    9:  "traffic_light",
    11: "stop_sign",
}


class TeslaStyleDetector:
    def __init__(self, model_path: str = "yolov5s.pt", conf: float = 0.25, iou: float = 0.45):
        print("Loading YOLOv5 model for Tesla-style detection...")
        self.model      = torch.hub.load(".", "custom", path=model_path, source="local")
        self.model.conf = conf
        self.model.iou  = iou
        self._latencies: list[float] = []
        print("Model loaded successfully!")

    def process_frame(self, frame: np.ndarray):
        """
        Run inference on a single RGB frame.
        Returns a pandas DataFrame of Tesla-relevant detections.
        """
        results    = self.model(frame)
        detections = results.pandas().xyxy[0]
        if len(detections) == 0:
            return detections
        return detections[detections["class"].isin(TESLA_COCO_CLASSES)]

    def benchmark_latency(
        self,
        image_paths: Optional[list[str]] = None,
        num_frames: int = 50,
        warmup: int = 5,
    ) -> dict:
        """
        Measure end-to-end inference latency over N frames.

        Includes a warmup phase (excluded from stats) to avoid JIT cold-start
        inflation. Reports p50/p90/p99 in addition to mean/min/max — averages
        alone mask tail latency spikes that matter for real-time systems.

        Returns a dict with all latency stats in milliseconds.
        """
        if image_paths is None:
            image_paths = ["data/images/bus.jpg", "data/images/zidane.jpg"]

        # Load images once
        frames: list[np.ndarray] = []
        for p in image_paths:
            img = cv2.imread(p)
            if img is not None:
                frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                print(f"  Warning: could not load {p}")
        if not frames:
            raise RuntimeError("No valid images found for benchmark.")

        total_needed = warmup + num_frames
        frame_cycle  = [frames[i % len(frames)] for i in range(total_needed)]

        print(f"\nRunning latency benchmark ({warmup} warmup + {num_frames} measured frames)...")
        latencies: list[float] = []

        for i, frame in enumerate(frame_cycle):
            t0      = time.perf_counter()
            self.process_frame(frame)
            elapsed = (time.perf_counter() - t0) * 1000  # → ms

            if i >= warmup:
                latencies.append(elapsed)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1 - warmup:>3}/{num_frames} measured frames")

        arr = np.array(latencies)
        stats = {
            "frames":   len(arr),
            "mean_ms":  float(np.mean(arr)),
            "p50_ms":   float(np.percentile(arr, 50)),
            "p90_ms":   float(np.percentile(arr, 90)),
            "p99_ms":   float(np.percentile(arr, 99)),
            "min_ms":   float(np.min(arr)),
            "max_ms":   float(np.max(arr)),
            "std_ms":   float(np.std(arr)),
            "fps":      float(1000.0 / np.mean(arr)),
        }

        self._latencies.extend(latencies)

        width = 48
        print("\n" + "=" * width)
        print("  TESLA CAMERA LATENCY BENCHMARK")
        print("=" * width)
        print(f"  Frames measured : {stats['frames']}")
        print(f"  Mean            : {stats['mean_ms']:6.2f} ms")
        print(f"  p50             : {stats['p50_ms']:6.2f} ms")
        print(f"  p90             : {stats['p90_ms']:6.2f} ms")
        print(f"  p99             : {stats['p99_ms']:6.2f} ms   ← tail")
        print(f"  Min / Max       : {stats['min_ms']:.2f} / {stats['max_ms']:.2f} ms")
        print(f"  Estimated FPS   : {stats['fps']:.1f}")
        print("=" * width)

        target_ms = 50.0
        if stats["p99_ms"] < target_ms:
            print(f"  ✓ p99 < {target_ms:.0f}ms  — meets Tesla real-time target")
        elif stats["mean_ms"] < target_ms:
            print(f"  ~ mean < {target_ms:.0f}ms but p99={stats['p99_ms']:.1f}ms — tail needs work")
        else:
            print(f"  ✗ mean {stats['mean_ms']:.1f}ms > {target_ms:.0f}ms — optimisation required")

        return stats

    def visualize_detections(
        self,
        image_path: str,
        save_path: str = "tesla_detection_result.jpg",
    ) -> Optional[np.ndarray]:
        """Run detection on a single image, render bounding boxes, and save."""
        img = cv2.imread(image_path)
        if img is None:
            print(f"Warning: could not load {image_path}")
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results  = self.model(img_rgb)

        rendered     = results.render()[0]
        rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, rendered_bgr)

        relevant = self.process_frame(img_rgb)
        print(f"Saved {save_path}  ({len(relevant)} Tesla-relevant detections)")

        for _, row in relevant.iterrows():
            cid  = int(row["class"])
            name = TESLA_COCO_CLASSES.get(cid, "unknown")
            print(f"  {name:14s}  conf={row['confidence']:.2f}  "
                  f"bbox=[{int(row['xmin'])},{int(row['ymin'])},{int(row['xmax'])},{int(row['ymax'])}]")

        return rendered_bgr

    def detection_summary(self, frame: np.ndarray) -> dict:
        """
        Return a structured summary of detections in one frame.
        Useful for the fusion pipeline.
        """
        df   = self.process_frame(frame)
        dets = []
        for _, row in df.iterrows():
            dets.append({
                "class_id":   int(row["class"]),
                "class_name": TESLA_COCO_CLASSES.get(int(row["class"]), "unknown"),
                "confidence": float(row["confidence"]),
                "bbox":       [float(row["xmin"]), float(row["ymin"]),
                               float(row["xmax"]), float(row["ymax"])],
            })
        return {"detections": dets, "count": len(dets)}


if __name__ == "__main__":
    detector = TeslaStyleDetector()

    print("\n--- Detection on sample images ---")
    for img_path, out_path in [
        ("data/images/bus.jpg",    "tesla_bus.jpg"),
        ("data/images/zidane.jpg", "tesla_zidane.jpg"),
    ]:
        if Path(img_path).exists():
            detector.visualize_detections(img_path, out_path)

    print("\n--- Latency benchmark ---")
    detector.benchmark_latency(num_frames=30)
