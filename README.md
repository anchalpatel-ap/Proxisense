# 🚗 ProxiSense

> **Real-Time Pedestrian & Obstacle Intent Prediction for ADAS**

ProxiSense is an edge-ready, real-time perception pipeline designed for Advanced Driver Assistance Systems (ADAS). It fuses **YOLOv8 object detection**, **ByteTrack multi-object tracking**, and an **LSTM intent classifier** to predict the behavior of pedestrians and road agents — and raise actionable alerts before a collision occurs.

---

## ✨ Features

| Capability | Details |
|---|---|
| 🔍 **Detection** | YOLOv8-nano — fast, lightweight, ONNX-exportable |
| 🎯 **Tracking** | ByteTrack with Kalman-filter trajectory smoothing |
| 🧠 **Intent Prediction** | LSTM model classifying 4 intent classes |
| ⏱️ **TTC Alerts** | Time-To-Collision zones: GREEN / AMBER / RED |
| 📊 **Dashboard** | Live Streamlit dashboard with charts and alert table |
| ☁️ **Cloud Sync** | Optional AWS Greengrass telemetry (toggle in config) |
| 🏎️ **Edge-Ready** | FP16 inference, ONNX Runtime, targets 30 FPS |

### Intent Classes

| Label | Meaning |
|---|---|
| `STATIONARY` | Agent is standing still |
| `WILL_CROSS` | Pedestrian about to cross the road |
| `ERRATIC` | Unpredictable / sudden movement |
| `LANE_CHANGE` | Vehicle changing lanes |

---

## 🗂️ Project Structure

```
ProxiSense/
├── main.py                     # CLI entry point (OpenCV window)
├── config/
│   └── config.yaml             # All tunable parameters
├── src/
│   ├── pipeline.py             # Orchestrates the full Perception→Prediction→Alert pipeline
│   ├── perception/
│   │   ├── detector.py         # YOLOv8 wrapper (Ultralytics)
│   │   └── tracker.py          # ByteTrack multi-object tracker
│   ├── prediction/
│   │   ├── lstm_model.py       # LSTM inference (PyTorch / ONNX Runtime)
│   │   ├── trajectory.py       # Trajectory encoding & heuristic classifier
│   │   ├── train.py            # Training loop (synthetic + real data)
│   │   └── export_onnx.py      # Export trained model to ONNX
│   ├── alert/
│   │   ├── ttc.py              # Time-To-Collision calculator
│   │   └── alert_engine.py     # Alert zone evaluation & braking signal
│   └── dashboard/
│       └── app.py              # Streamlit real-time dashboard
├── scripts/
│   ├── train_on_jaad.py        # Fine-tune on the JAAD pedestrian dataset
│   ├── test_pipeline.py        # Smoke-test the full pipeline
│   └── download_models.py      # Helper to fetch pre-trained weights
├── models/
│   ├── detection/              # YOLO weights (.pt / .onnx)
│   └── prediction/             # LSTM weights (lstm_intent_best.pt / .onnx)
├── data/
│   ├── datasets/               # Training datasets (e.g. JAAD)
│   └── sample_videos/          # Test video clips
├── notebooks/
│   └── proxisense_training.ipynb  # Interactive training notebook
├── yolov8n.pt                  # Default YOLOv8-nano checkpoint
└── requirements.txt
```

---

## ⚙️ Pipeline Architecture

```
Video / Webcam
      │
      ▼
┌─────────────┐
│  YOLOv8     │  ← Detects pedestrians, vehicles, cyclists
│  Detector   │
└──────┬──────┘
       │ bounding boxes + class labels
       ▼
┌─────────────┐
│  ByteTrack  │  ← Assigns persistent track IDs, maintains trajectories
│  Tracker    │
└──────┬──────┘
       │ per-track trajectory history
       ▼
┌─────────────────┐
│  LSTM Intent    │  ← Classifies: STATIONARY / WILL_CROSS / ERRATIC / LANE_CHANGE
│  Classifier     │     (falls back to heuristics if no trained model present)
└──────┬──────────┘
       │ intent label + confidence
       ▼
┌─────────────┐
│  TTC Alert  │  ← Computes Time-To-Collision, raises GREEN / AMBER / RED alert
│  Engine     │     Emits BRAKE signal when TTC < 2 s
└──────┬──────┘
       │
       ▼
  OpenCV Overlay  OR  Streamlit Dashboard
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **GPU (recommended):** Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/) before running `pip install -r requirements.txt`.

### 2. Run on Webcam

```bash
python main.py
```

### 3. Run on a Video File

```bash
python main.py --video data/sample_videos/test.mp4
```

### 4. Launch the Streamlit Dashboard

```bash
streamlit run src/dashboard/app.py
```

Open your browser at **http://localhost:8501** and select an input source (Webcam / Video File / CARLA Simulator).

---

## 🖥️ CLI Reference

```
python main.py [OPTIONS]

Options:
  --video PATH        Path to a video file (omit to use webcam)
  --config PATH       Path to config YAML (default: config/config.yaml)
  --no-display        Headless mode — no OpenCV window
  --record PATH       Write output video to PATH
  --benchmark         Print per-frame FPS and inference time
```

---

## 🧠 Training the LSTM Model

### Option A — Synthetic Data (no dataset required)

```bash
python src/prediction/train.py --epochs 50 --output-dir models/prediction
```

### Option B — JAAD Dataset

1. Download the [JAAD dataset](http://data.nvision2.eecs.yorku.ca/JAAD_dataset/) and place annotations in `data/datasets/jaad/`.
2. Run the training script:

```bash
python scripts/train_on_jaad.py --data-dir data/datasets/jaad --epochs 50
```

### Export to ONNX (for edge deployment)

```bash
python src/prediction/export_onnx.py --model models/prediction/lstm_intent_best.pt
```

The resulting `lstm_intent_best.onnx` file will be picked up automatically by the pipeline.

---

## 🔧 Configuration

All parameters are in [`config/config.yaml`](config/config.yaml):

```yaml
model:
  detection:
    name: "yolov8n"           # YOLOv8 variant (n/s/m/l/x)
    conf_threshold: 0.5       # Detection confidence cutoff
    half_precision: true      # FP16 for faster GPU inference

  prediction:
    type: "lstm"
    input_len: 15             # Past frames fed to LSTM
    predict_len: 10           # Future frames to reason over
    confidence_threshold: 0.75

tracking:
  method: "bytetrack"
  max_lost: 30                # Frames to keep a lost track alive

alert:
  ttc_thresholds:
    green: 8.0                # Safe   (seconds)
    amber: 4.0                # Caution
    red: 2.0                  # Immediate danger — BRAKE signal

cloud:
  enabled: false              # Toggle AWS Greengrass sync
  greengrass:
    thing_name: "proxisense_01"
    region: "ap-south-1"
    topic: "proxisense/anomalies"
```

---

## 📋 Requirements

| Package | Purpose |
|---|---|
| `ultralytics >= 8.0` | YOLOv8 detection |
| `torch >= 1.13` | LSTM training & inference |
| `onnxruntime >= 1.15` | Edge ONNX inference |
| `opencv-python >= 4.8` | Video I/O & rendering |
| `streamlit >= 1.28` | Real-time web dashboard |
| `filterpy >= 1.4.5` | Kalman filters for ByteTrack |
| `lap >= 0.4` | Linear assignment for ByteTrack |
| `scipy >= 1.10` | TTC calculations |
| `boto3 >= 1.28` | AWS Greengrass (optional) |

---

## 🧪 Testing

```bash
# Smoke-test the full pipeline on a sample video
python scripts/test_pipeline.py
```

---

## 🗺️ Roadmap

- [ ] Transformer-based intent predictor
- [ ] CARLA Simulator integration (live telemetry)
- [ ] Multi-camera / BEV fusion
- [ ] TensorRT optimization for NVIDIA Jetson
- [ ] AWS Greengrass cloud anomaly sync

---

## 📄 License

This project is for research and educational purposes. See [LICENSE](LICENSE) for details.
