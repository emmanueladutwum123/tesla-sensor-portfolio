# tesla_detector.py - Tesla-Style Object Detector
import time

import cv2
import numpy as np
import torch


class TeslaStyleDetector:
    def __init__(self, model_path="yolov5s.pt"):
        """Initialize detector with YOLOv5 model."""
        print("Loading YOLOv5 model for Tesla-style detection...")
        self.model = torch.hub.load(".", "custom", path=model_path, source="local")
        self.model.conf = 0.25  # confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        print("Model loaded successfully!")

    def process_frame(self, frame):
        """Process single frame and return Tesla-relevant detections."""
        # Run inference
        results = self.model(frame)

        # Get detections as pandas DataFrame
        detections = results.pandas().xyxy[0]

        # Tesla-relevant COCO classes:
        # 2: car, 3: motorcycle, 5: bus, 7: truck, 9: traffic light, 11: stop sign
        tesla_classes = [2, 3, 5, 7, 9, 11]

        # Filter for Tesla-relevant objects
        if len(detections) > 0:
            relevant = detections[detections["class"].isin(tesla_classes)]
            return relevant
        return detections

    def benchmark_latency(self, num_frames=30):
        """Simple latency benchmark using sample images instead of webcam."""
        print(f"\n⏱️ Running benchmark on sample images for {num_frames} frames...")

        # Use the sample images we already have
        test_images = ["data/images/bus.jpg", "data/images/zidane.jpg"]
        latencies = []
        frame_count = 0

        # Loop through images until we reach desired frame count
        for img_path in test_images * (num_frames // 2 + 1):
            if frame_count >= num_frames:
                break

            # Load and process image
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Could not load image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Measure inference time
            start = time.time()
            self.process_frame(img_rgb)
            inference_time = (time.time() - start) * 1000  # convert to ms

            latencies.append(inference_time)
            frame_count += 1

            if frame_count % 10 == 0 or frame_count == num_frames:
                print(f"  Processed {frame_count}/{num_frames} frames")

        # Calculate statistics
        latencies = np.array(latencies)
        mean_latency = np.mean(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        std_latency = np.std(latencies)

        print("\n" + "=" * 50)
        print("TESLA LATENCY BENCHMARK RESULTS")
        print("=" * 50)
        print(f"Mean latency: {mean_latency:.2f}ms")
        print(f"Min latency: {min_latency:.2f}ms")
        print(f"Max latency: {max_latency:.2f}ms")
        print(f"Std deviation: {std_latency:.2f}ms")
        print(f"Estimated FPS: {1000 / mean_latency:.1f}")
        print("=" * 50)

        # Tesla target check
        if mean_latency < 50:
            print("✅ MEETS TESLA TARGET: <50ms latency!")
        else:
            print("⚠️ Needs optimization to meet Tesla's 50ms target")

        return mean_latency

    def visualize_detections(self, image_path, save_path="tesla_detection_result.jpg"):
        """Run detection on an image and save visualization."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"⚠️ Could not load image: {image_path}")
            return None

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Run detection
        results = self.model(img_rgb)

        # Render results on image
        rendered = results.render()[0]
        rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR)

        # Save
        cv2.imwrite(save_path, rendered_bgr)
        print(f"✅ Saved: {save_path}")

        # Print detections summary
        results.pandas().xyxy[0]
        tesla_relevant = self.process_frame(img_rgb)
        print(f"  Found {len(tesla_relevant)} Tesla-relevant objects")

        return rendered_bgr


# Quick test if run directly
if __name__ == "__main__":
    detector = TeslaStyleDetector()

    # Test on sample images
    print("\n📸 Testing on sample images...")
    detector.visualize_detections("data/images/bus.jpg", "tesla_bus.jpg")
    detector.visualize_detections("data/images/zidane.jpg", "tesla_zidane.jpg")

    # Run latency benchmark
    detector.benchmark_latency(num_frames=30)
