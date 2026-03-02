<div align="center">                                                                              File: README.md                                                                                    Modified

# Tesla-Style YOLOv5 Object Detector

## 🚗 Project Overview
This project implements a **real-time object detector** optimized for Tesla-relevant classes (cars, trucks, traffic lights, etc.) using YOLOv5. Built for Tesla's Audio & Sensor Software Engineering internsh$

## 📊 Key Features
- **Tesla-Relevant Detection**: Filters COCO classes to focus on:
  - Cars (class 2)
  - Motorcycles (class 3)
  - Buses (class 5)
  - Trucks (class 7)
  - Traffic lights (class 9)
  - Stop signs (class 11)

- **Latency Benchmarking**: Measures inference speed to meet Tesla's <50ms target
- **Visualization**: Saves images with bounding boxes for portfolio proof

## 📸 Results
### Sample Detections
| Input | Output |
|-------|--------|
| Bus Image | ![Bus Detection](tesla_bus.jpg) |
| Zidane Image | ![Zidane Detection](tesla_zidane.jpg) |

### Performance Metrics
| Metric | Value | Tesla Target |
|--------|-------|--------------|
| Mean Latency | 62.40ms | <50ms |
| FPS | 16.0 | >20 |

## 🛠<️ Technical Implementation
- **Framework**: PyTorch, YOLOv5
- **Languages**: Python
- **Key Libraries**: OpenCV, NumPy, Torch Hub

## 🎯 Why This Matters for Tesla
This project demonstrates:
- ✅ Experience with CNN-based object detection
- ✅ Understanding of autonomous vehicle perception needs
- ✅ Ability to benchmark and optimize for real-time systems
- ✅ Clean code architecture with Tesla-specific requirements

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/emmanueladutwum123/tesla-sensor-portfolio
cd tesla-sensor-portfolio/yolo_detection

# Install requirements
pip install -r requirements.txt

# Run detector
python tesla_detector.py<div align="center">
  <p>
    <a href="https://platform.ultralytics.com/ultralytics/yolo26" target="_blank">
      <img width="100%" src="https://raw.githubusercontent.com/ultralytics/assets/main/yolov8/banner-yolov8.png" alt="Ultralytics YOLO banner"></a>
  </p>
 


