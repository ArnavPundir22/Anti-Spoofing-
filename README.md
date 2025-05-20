# 🛡️ Anti-Spoofing System

A deep learning-based Anti-Spoofing System that uses YOLOv8 to distinguish between real and fake faces, helping prevent spoofing attacks in face recognition systems. This project includes tools for dataset collection, preprocessing, training, and real-time detection.

---

## 📁 Project Structure

```
.
├── DataCollection.py        # Script to collect and classify face data
├── Dataset/                 # Organized dataset folders
│   ├── DataCollect/         # Raw collected data
│   │   ├── All/             # All collected samples
│   │   ├── Fake/            # Fake face samples (photos/videos)
│   │   └── Real/            # Real face samples (live)
│   └── SplitData/           # YOLO-formatted dataset
│       ├── data.yaml
│       ├── train/val/test/  # Images and YOLO labels
├── main.py                  # Run real-time detection
├── model/                   # YOLOv8 pretrained weights
│   └── yolov8n.pt
├── requirements.txt         # Python dependencies
├── runs/                    # Training results
│   └── detect/train4/
├── SplitData.py             # Convert raw data into YOLO format
├── train.py                 # Train the YOLO model
└── yolov8n.pt               # YOLOv8 checkpoint file
```

---

## 🚀 Features

- Real vs Fake face classification
- YOLOv8-based detection
- Automatic dataset collection and splitting
- Visual metrics and performance graphs
- Real-time camera-based spoof detection

---

## 🔧 Setup Instructions

### 1️⃣ Clone the repository

```bash
git clone https://github.com/yourusername/antispoofing-system.git
cd antispoofing-system
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 📸 Step-by-Step Workflow

### ✅ Step 1: Collect real and fake face data

Run the following script and choose either "Real" or "Fake" to store images from the webcam:

```bash
python DataCollection.py
```

- Stores images in `Dataset/DataCollect/Real` or `Fake` based on user input.
- All images are also stored in `All` for backup.

---

### 🔄 Step 2: Convert collected data into YOLO format

```bash
python SplitData.py
```

- Converts images and labels into YOLO format
- Outputs organized data in `Dataset/SplitData/{train, val, test}` with `data.yaml`

---

### 🏋️ Step 3: Train the YOLOv8 model

```bash
python train.py
```

- Uses `Dataset/SplitData/data.yaml` as the training config
- Training logs and weights saved in `runs/detect/train4/`

---

### 👁️ Step 4: Run detection on live camera or images

```bash
python main.py
```

- Loads best model from training and runs detection
- Can be used for real-time spoof detection

---

## 📊 Output Examples

You’ll find the following results in `runs/detect/train4/`:

- `results.png`: Accuracy and loss graph
- `confusion_matrix.png`: Confusion matrix
- `PR_curve.png`, `F1_curve.png`, etc.: Model performance metrics
- `weights/best.pt`: Best performing model saved here

---

## 📦 Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- NumPy
- tqdm
- PyYAML

(Full list available in `requirements.txt`)

---

## 📝 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV for video capture
- Contributors to open-source anti-spoofing datasets and tools

---

## 🔐 License

This project is licensed under the MIT License.
