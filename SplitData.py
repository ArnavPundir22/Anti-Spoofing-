import os
import random
import shutil

# Input and Output Paths
inputFolderPath = "Dataset/DataCollect/All"
outputFolderPath = "Dataset/SplitData"

# Class labels
classes = ["fake", "real"]

# Dataset split ratios
splitRatio = {"train": 0.7, "val": 0.2, "test": 0.1}

# Clean and recreate output folder
if os.path.exists(outputFolderPath):
    shutil.rmtree(outputFolderPath)
os.makedirs(outputFolderPath)

for phase in ["train", "val", "test"]:
    os.makedirs(f"{outputFolderPath}/{phase}/images", exist_ok=True)
    os.makedirs(f"{outputFolderPath}/{phase}/labels", exist_ok=True)

# Get all .jpg base filenames
allFiles = os.listdir(inputFolderPath)
baseNames = list(set([name.split('.')[0] for name in allFiles if name.endswith(".jpg")]))
random.shuffle(baseNames)

# Calculate splits
total = len(baseNames)
lenTrain = int(splitRatio["train"] * total)
lenVal = int(splitRatio["val"] * total)

splitNames = [
    baseNames[:lenTrain],
    baseNames[lenTrain:lenTrain + lenVal],
    baseNames[lenTrain + lenVal:]
]
phases = ["train", "val", "test"]

# Copy files
for phase, names in zip(phases, splitNames):
    for name in names:
        img_src = os.path.join(inputFolderPath, name + ".jpg")
        txt_src = os.path.join(inputFolderPath, name + ".txt")
        img_dst = os.path.join(outputFolderPath, phase, "images", name + ".jpg")
        txt_dst = os.path.join(outputFolderPath, phase, "labels", name + ".txt")
        if os.path.exists(img_src) and os.path.exists(txt_src):
            shutil.copy(img_src, img_dst)
            shutil.copy(txt_src, txt_dst)
        else:
            print(f"❌ Missing pair: {name}")

print(f"✅ Data Split Completed: Train={lenTrain}, Val={lenVal}, Test={total - lenTrain - lenVal}")

# YAML file
dataYamlContent = f"""path: Dataset/SplitData
train: train/images
val: val/images
test: test/images

nc: {len(classes)}
names: {classes}
"""

with open(f"{outputFolderPath}/data.yaml", "w") as f:
    f.write(dataYamlContent)

print("✅ data.yaml file created successfully.")
