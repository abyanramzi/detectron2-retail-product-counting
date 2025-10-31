# 🎯 Detectron2 + SORT Tracking for Retail Product Counting

## 🧩 Overview
This project focuses on **object counting and tracking for retail products** using a combination of **Detectron2** for object detection and a **Kalman Filter + Hungarian Algorithm (SORT)** for multi-object tracking.  

The system is designed to accurately count products that cross a defined line — distinguishing between **incoming (left → right)** and **outgoing (right → left)** movements.  
It was developed as part of a **research and development initiative** to improve an existing client solution where **hand movements were mistakenly counted** as product transitions.

---

## 🚀 Objectives
- Develop a robust retail product counting system.
- Accurately track products crossing a counting line in both directions.
- Prevent false counts caused by hand movements near the line.
- Improve detection consistency under varying lighting and camera angles.
- Demonstrate the integration of **Detectron2 (amodal detection)** with **SORT tracking** for real-time analysis.

---

## 🧠 Methodology & Architecture
### Detection and Tracking Flow
The system operates in a continuous pipeline:

📹 Video Stream → 🧭 Detectron2 Object Detection → 🔄 Kalman Filter Prediction → 
🔗 Hungarian Matching → 🎯 Object ID Assignment → 🚦 Line Crossing Logic → 📊 Count Update

### Counting Rule
- A **polygon line area** is defined between `x = 600` and `x = 610`.
- If a tracked object's **centroid** moves **from left to right**, it counts as **IN (+1)**.
- If it moves **from right to left**, it counts as **OUT (-1)**.
- The Kalman filter smooths motion, and Hungarian matching ensures consistent object IDs, even during occlusion.

---

## ⚙️ Dataset & Training
- **Model:** Faster R-CNN (from Detectron2 Model Zoo)
- **Training Type:** Custom dataset fine-tuning
- **Images:** 372 base images augmented to 830 (brightness ±20%)
- **Classes:**  
  1. Nextar  
  2. Steam Cake  
- **Annotation Type:** Amodal annotation (object bounding boxes include partially occluded objects)
- **Hardware:** NVIDIA RTX 4050 GPU

---

## 🎞️ Demonstration
Two main results were achieved:

1. ✅ **Accurate counting** of retail products moving across the line.  
2. 🚫 **No false detection** for hand movements near the line.

| Scenario | Example GIF |
|-----------|--------------|
| Product Counting | ![Counting Example](assets/product_counting.gif) |
| Hand Exclusion | ![Hand Filtering Example](assets/hand_exclusion.gif) |

---

## 🧩 System Components

| Component | Description |
|------------|--------------|
| **Detectron2** | Performs object detection for each frame. |
| **Kalman Filter** | Predicts future object positions to handle motion and temporary occlusion. |
| **Hungarian Algorithm** | Associates detections to tracked objects with minimal ID switching. |
| **Counting Logic** | Detects centroid crossing through the polygon line area and updates count. |
| **OpenCV** | Handles real-time video stream display. |

---

