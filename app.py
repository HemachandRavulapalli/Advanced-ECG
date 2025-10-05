import os
import shutil
import kagglehub

# your target folder
target_dir = "/home/Hemachand/D4/data/raw/kardia"

# make sure it exists
os.makedirs(target_dir, exist_ok=True)

print("📥 Downloading Kardia 6L ECG dataset from Kaggle...")
path = kagglehub.dataset_download("saadkhan0410/kardia-6l-ecg-dataset")

print("✅ Download complete. Files cached at:", path)

# copy downloaded files into your project folder
for item in os.listdir(path):
    src = os.path.join(path, item)
    dst = os.path.join(target_dir, item)
    if os.path.isdir(src):
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)

print(f"📦 Dataset moved to {target_dir}")
