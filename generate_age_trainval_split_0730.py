# WristFX_0730/generate_age_trainval_split_0730.py
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from glob import glob

# 현재 파일 기준 경로
BASE_DIR = os.path.dirname(__file__)
BASE_IMG_DIR = "/mnt/data/KimJG/ELBOW_test/Kaggle_dataset"
PARTS = ["images_part1", "images_part2", "images_part3", "images_part4"]

def resolve_image_path(filestem):
    for part in PARTS:
        matches = glob(os.path.join(BASE_IMG_DIR, part, f"{filestem}.*"))
        if matches:
            return matches[0]
    return None

def age_group_label(age):
    age = float(age)
    if age < 5: return 0
    elif age < 10: return 1
    elif age < 15: return 2
    else: return 3

def generate_split_by_projection(df, df_test, projection_value, proj_name):
    df_proj = df[df["projection"] == projection_value]
    df_proj["label"] = df_proj["fracture_visible"].apply(lambda x: 1 if x == 1 else 0)
    df_proj["age_group_label"] = df_proj["age"].astype(float).apply(age_group_label)
    df_proj["image_path"] = df_proj["filestem"].apply(resolve_image_path)
    df_proj = df_proj[df_proj["image_path"].notnull()]

    df_test["filestem"] = df_test["image_path"].apply(lambda x: os.path.basename(x).split(".")[0])
    df_proj = df_proj[~df_proj["filestem"].isin(df_test["filestem"])]

    print(f"🔍 {proj_name}: Split 대상 샘플 수 = {len(df_proj)}")

    df_train, df_val = train_test_split(
        df_proj, test_size=0.2, random_state=42, stratify=df_proj["age_group_label"])

    df_train.to_csv(os.path.join(BASE_DIR, f"age_train_tmp_{proj_name}.csv"), index=False)
    df_val.to_csv(os.path.join(BASE_DIR, f"age_val_tmp_{proj_name}.csv"), index=False)
    print(f"✅ Saved: age_train_tmp_{proj_name}.csv, age_val_tmp_{proj_name}.csv")

def generate_train_val():
    print("📂 Loading full dataset and fixed test set...")
    df = pd.read_csv(os.path.join(BASE_DIR, "dataset.csv"))
    df_test = pd.read_csv(os.path.join(BASE_DIR, "age_test.csv"))

    df = df[df['metal'] != 1]
    df = df[df['age'].notnull()]

    generate_split_by_projection(df, df_test, projection_value=1, proj_name="AP")
    generate_split_by_projection(df, df_test, projection_value=2, proj_name="Lat")

if __name__ == "__main__":
    print("📌 Step 2: Generating random train/val split by projection...")
    generate_train_val()
