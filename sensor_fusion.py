"""
Sensor Fusion — Tesla Sensor Portfolio

Late-fusion strategy:
  1. Camera stream  → YOLO 2D bounding boxes (class, confidence, [x1,y1,x2,y2])
  2. LiDAR stream   → PointNet 3D detections  (class, confidence, centroid_xyz)
  3. Association    → project LiDAR centroids into the image plane via a
                      calibration matrix; match to YOLO boxes with IoU; merge
                      confidences for co-detected objects.

The module is self-contained and usable without a running YOLO or PointNet
model — pass pre-computed detection dicts from either source.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CameraDetection:
    """A single 2D detection from the camera pipeline."""
    class_name:  str
    class_id:    int
    confidence:  float
    bbox:        np.ndarray          # [x1, y1, x2, y2] in pixel coords
    timestamp_ms: float = 0.0


@dataclass
class LiDARDetection:
    """A single 3D detection from the LiDAR/PointNet pipeline."""
    class_name:  str
    class_id:    int
    confidence:  float
    centroid:    np.ndarray          # [X, Y, Z] in LiDAR frame (metres)
    num_points:  int = 0
    timestamp_ms: float = 0.0


@dataclass
class FusedDetection:
    """A detection that fuses camera + LiDAR evidence."""
    class_name:      str
    class_id:        int
    fused_confidence: float
    bbox:            Optional[np.ndarray] = None   # from camera, if matched
    centroid:        Optional[np.ndarray] = None   # from LiDAR, if matched
    camera_conf:     float = 0.0
    lidar_conf:      float = 0.0
    source:          str = "fused"   # "camera_only" | "lidar_only" | "fused"
    timestamp_ms:    float = 0.0


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

class CameraLiDARCalibration:
    """
    Holds the extrinsic and intrinsic matrices needed to project a LiDAR
    point into the image plane.

    For real deployment load these from the vehicle calibration file.
    The defaults here are plausible for a forward-facing camera at ~2 m height.
    """

    def __init__(
        self,
        K:    Optional[np.ndarray] = None,
        R:    Optional[np.ndarray] = None,
        t:    Optional[np.ndarray] = None,
        image_wh: tuple[int, int] = (1280, 720),
    ):
        # Intrinsic (3×3)
        if K is None:
            fx = fy = 800.0
            cx, cy  = image_wh[0] / 2, image_wh[1] / 2
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
        # Rotation LiDAR→camera (3×3)
        if R is None:
            R = np.eye(3, dtype=np.float64)
        # Translation LiDAR→camera (3,)
        if t is None:
            t = np.array([0.0, -2.0, 0.0], dtype=np.float64)

        self.K        = K
        self.R        = R
        self.t        = t
        self.image_wh = image_wh

    def project(self, pts_lidar: np.ndarray) -> np.ndarray:
        """
        Project N×3 LiDAR points into the image plane.
        Returns N×2 pixel coords (may include points behind the camera).
        """
        pts_cam = (self.R @ pts_lidar.T).T + self.t   # N×3
        # Filter behind camera
        behind = pts_cam[:, 2] <= 0
        uv     = (self.K @ pts_cam.T).T               # N×3
        uv     = uv[:, :2] / np.where(uv[:, 2:3] == 0, 1e-6, uv[:, 2:3])
        uv[behind] = -1                                # mark as invalid
        return uv  # N×2


# ---------------------------------------------------------------------------
# IoU helpers
# ---------------------------------------------------------------------------

def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    """2D axis-aligned bounding box IoU. a, b are [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def point_in_bbox(uv: np.ndarray, bbox: np.ndarray) -> bool:
    """Check if a 2D pixel point falls inside a bounding box."""
    return (bbox[0] <= uv[0] <= bbox[2]) and (bbox[1] <= uv[1] <= bbox[3])


# ---------------------------------------------------------------------------
# Class ID mapping between camera (COCO) and LiDAR label spaces
# ---------------------------------------------------------------------------

# COCO class IDs → Tesla LiDAR class IDs
_COCO_TO_LIDAR: dict[int, int] = {
    2:  0,   # car       → car
    3:  2,   # motorcycle→ cyclist
    5:  0,   # bus       → car   (closest available)
    7:  3,   # truck     → truck
    0:  1,   # person    → pedestrian
    9:  -1,  # traffic light — no LiDAR class
    11: -1,  # stop sign     — no LiDAR class
}

LIDAR_CLASSES = ["car", "pedestrian", "cyclist", "truck", "cone", "background"]


def _lidar_class_for_coco(coco_id: int) -> int:
    return _COCO_TO_LIDAR.get(coco_id, -1)


# ---------------------------------------------------------------------------
# Fusion engine
# ---------------------------------------------------------------------------

class SensorFusion:
    """
    Fuses camera (YOLO) and LiDAR (PointNet) detections.

    Confidence fusion: weighted average of camera and LiDAR confidences,
    boosted by a co-detection bonus when both sensors agree on class.
    """

    def __init__(
        self,
        calib:            Optional[CameraLiDARCalibration] = None,
        iou_threshold:    float = 0.3,
        camera_weight:    float = 0.6,
        lidar_weight:     float = 0.4,
        codetect_bonus:   float = 0.1,
        max_time_diff_ms: float = 100.0,
    ):
        self.calib            = calib or CameraLiDARCalibration()
        self.iou_threshold    = iou_threshold
        self.camera_weight    = camera_weight
        self.lidar_weight     = lidar_weight
        self.codetect_bonus   = codetect_bonus
        self.max_time_diff_ms = max_time_diff_ms

    # ------------------------------------------------------------------

    def _lidar_to_image_bbox(self, det: LiDARDetection, box_size_px: float = 50.0) -> Optional[np.ndarray]:
        """Project a LiDAR centroid to a rough 2D bounding box estimate."""
        uv = self.calib.project(det.centroid.reshape(1, 3))[0]
        if uv[0] < 0:  # behind camera
            return None
        h, w = self.calib.image_wh[1], self.calib.image_wh[0]
        if not (0 <= uv[0] < w and 0 <= uv[1] < h):
            return None
        half = box_size_px / 2
        return np.array([uv[0] - half, uv[1] - half, uv[0] + half, uv[1] + half])

    def _classes_compatible(self, cam_id: int, lidar_id: int) -> bool:
        expected = _lidar_class_for_coco(cam_id)
        return expected == lidar_id

    def _fuse_pair(
        self,
        cam: CameraDetection,
        lid: LiDARDetection,
        bonus: bool,
    ) -> FusedDetection:
        fused_conf = (
            self.camera_weight * cam.confidence
            + self.lidar_weight * lid.confidence
            + (self.codetect_bonus if bonus else 0.0)
        )
        fused_conf = min(1.0, fused_conf)
        return FusedDetection(
            class_name       = cam.class_name,
            class_id         = cam.class_id,
            fused_confidence = fused_conf,
            bbox             = cam.bbox,
            centroid         = lid.centroid,
            camera_conf      = cam.confidence,
            lidar_conf       = lid.confidence,
            source           = "fused",
            timestamp_ms     = cam.timestamp_ms,
        )

    # ------------------------------------------------------------------

    def fuse(
        self,
        camera_dets: List[CameraDetection],
        lidar_dets:  List[LiDARDetection],
    ) -> List[FusedDetection]:
        """
        Associate and fuse camera + LiDAR detections.
        Unmatched detections are kept as single-sensor outputs.
        """
        matched_cam   = set()
        matched_lidar = set()
        fused: List[FusedDetection] = []

        # Build projected bboxes for LiDAR detections
        lidar_bboxes = [self._lidar_to_image_bbox(d) for d in lidar_dets]

        # Greedy matching: highest-IoU pair first
        iou_pairs = []
        for ci, cam in enumerate(camera_dets):
            for li, (lid, lb) in enumerate(zip(lidar_dets, lidar_bboxes)):
                if lb is None:
                    continue
                iou = bbox_iou(cam.bbox, lb)
                if iou >= self.iou_threshold:
                    iou_pairs.append((iou, ci, li))

        iou_pairs.sort(reverse=True)

        for iou, ci, li in iou_pairs:
            if ci in matched_cam or li in matched_lidar:
                continue
            cam = camera_dets[ci]
            lid = lidar_dets[li]
            matched_cam.add(ci)
            matched_lidar.add(li)
            bonus = self._classes_compatible(cam.class_id, lid.class_id)
            fused.append(self._fuse_pair(cam, lid, bonus))

        # Camera-only detections
        for ci, cam in enumerate(camera_dets):
            if ci not in matched_cam:
                fused.append(FusedDetection(
                    class_name       = cam.class_name,
                    class_id         = cam.class_id,
                    fused_confidence = cam.confidence * self.camera_weight,
                    bbox             = cam.bbox,
                    camera_conf      = cam.confidence,
                    source           = "camera_only",
                    timestamp_ms     = cam.timestamp_ms,
                ))

        # LiDAR-only detections
        for li, lid in enumerate(lidar_dets):
            if li not in matched_lidar:
                fused.append(FusedDetection(
                    class_name       = lid.class_name,
                    class_id         = lid.class_id,
                    fused_confidence = lid.confidence * self.lidar_weight,
                    centroid         = lid.centroid,
                    lidar_conf       = lid.confidence,
                    source           = "lidar_only",
                    timestamp_ms     = lid.timestamp_ms,
                ))

        # Sort by confidence descending
        fused.sort(key=lambda d: d.fused_confidence, reverse=True)
        return fused


# ---------------------------------------------------------------------------
# Live pipeline (ties YOLO + PointNet together)
# ---------------------------------------------------------------------------

class TeslaPerceptionPipeline:
    """
    End-to-end perception: camera frame + LiDAR scan → fused detections.

    Pass in pre-built detector objects to avoid circular imports:
      camera_detector : TeslaStyleDetector  (from tesla_detector.py)
      lidar_classifier: LiDARClassifier     (from pointnet_classifier.py)
    """

    TESLA_COCO_CLASSES = {2: "car", 3: "motorcycle", 5: "bus",
                          7: "truck", 9: "traffic_light", 11: "stop_sign", 0: "person"}

    def __init__(
        self,
        camera_detector=None,
        lidar_classifier=None,
        calib: Optional[CameraLiDARCalibration] = None,
    ):
        self.camera  = camera_detector
        self.lidar   = lidar_classifier
        self.fusion  = SensorFusion(calib=calib)
        self._latencies: list[float] = []

    def _yolo_to_camera_dets(self, df, ts: float) -> List[CameraDetection]:
        dets = []
        for _, row in df.iterrows():
            cid = int(row["class"])
            if cid not in self.TESLA_COCO_CLASSES:
                continue
            dets.append(CameraDetection(
                class_name   = self.TESLA_COCO_CLASSES[cid],
                class_id     = cid,
                confidence   = float(row["confidence"]),
                bbox         = np.array([row["xmin"], row["ymin"],
                                         row["xmax"], row["ymax"]]),
                timestamp_ms = ts,
            ))
        return dets

    def _pointnet_to_lidar_dets(self, raw: list, ts: float) -> List[LiDARDetection]:
        return [
            LiDARDetection(
                class_name   = d["class_name"],
                class_id     = d["class_id"],
                confidence   = d["confidence"],
                centroid     = np.asarray(d["centroid"], dtype=np.float64),
                num_points   = d.get("num_points", 0),
                timestamp_ms = ts,
            )
            for d in raw
            if d["class_name"] != "background"
        ]

    def run(self, frame, lidar_pts: np.ndarray) -> List[FusedDetection]:
        """
        Args:
            frame      : BGR image (H×W×3 numpy array)
            lidar_pts  : (N, 3+) LiDAR point cloud
        Returns:
            List of FusedDetection sorted by confidence.
        """
        ts = time.time() * 1000
        t0 = time.perf_counter()

        camera_dets: List[CameraDetection] = []
        lidar_dets:  List[LiDARDetection]  = []

        if self.camera is not None:
            import cv2
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            df = self.camera.process_frame(frame_rgb)
            camera_dets = self._yolo_to_camera_dets(df, ts)

        if self.lidar is not None and len(lidar_pts) > 0:
            raw = self.lidar.classify_cloud(lidar_pts)
            lidar_dets = self._pointnet_to_lidar_dets(raw, ts)

        result = self.fusion.fuse(camera_dets, lidar_dets)
        self._latencies.append((time.perf_counter() - t0) * 1000)
        return result

    def latency_report(self) -> dict:
        if not self._latencies:
            return {}
        arr = np.array(self._latencies)
        return {
            "frames":    len(arr),
            "mean_ms":   float(np.mean(arr)),
            "p50_ms":    float(np.percentile(arr, 50)),
            "p90_ms":    float(np.percentile(arr, 90)),
            "p99_ms":    float(np.percentile(arr, 99)),
            "min_ms":    float(np.min(arr)),
            "max_ms":    float(np.max(arr)),
        }


# ---------------------------------------------------------------------------
# CLI demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Sensor Fusion Demo — synthetic data (no model weights needed)\n")

    # Synthetic camera detections
    cam_dets = [
        CameraDetection("car",    2, 0.91, np.array([300, 200, 600, 450])),
        CameraDetection("truck",  7, 0.78, np.array([700, 180, 1100, 500])),
        CameraDetection("person", 0, 0.65, np.array([100, 300, 200, 600])),
    ]

    # Synthetic LiDAR detections
    lid_dets = [
        LiDARDetection("car",        0, 0.88, np.array([15.0,  2.0, 0.5]), 512),
        LiDARDetection("truck",      3, 0.82, np.array([20.0, -5.0, 1.2]), 1024),
        LiDARDetection("pedestrian", 1, 0.70, np.array([ 8.0,  8.0, 0.9]),  128),
    ]

    # Default calibration (identity rotation, 2m offset)
    calib  = CameraLiDARCalibration(image_wh=(1280, 720))
    fusion = SensorFusion(calib=calib)
    result = fusion.fuse(cam_dets, lid_dets)

    print(f"{'Source':<14} {'Class':<14} {'Fused Conf':>10}  {'Cam':>6}  {'LiDAR':>6}")
    print("-" * 58)
    for d in result:
        bbox_str  = f"bbox={d.bbox.astype(int).tolist()}" if d.bbox is not None else ""
        cent_str  = (f"centroid=({d.centroid[0]:.1f},{d.centroid[1]:.1f},{d.centroid[2]:.1f})"
                     if d.centroid is not None else "")
        print(f"{d.source:<14} {d.class_name:<14} {d.fused_confidence:>10.3f}  "
              f"{d.camera_conf:>6.3f}  {d.lidar_conf:>6.3f}  {bbox_str} {cent_str}")
