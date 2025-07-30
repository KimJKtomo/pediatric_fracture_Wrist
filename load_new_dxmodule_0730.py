# load_new_dxmodule_0730.py
import os
import pandas as pd
from glob import glob
from sklearn.model_selection import train_test_split
from ao_to_salter_utils import extract_ao_subtypes, is_growth_plate_fracture

def extract_part(ao_code):
    if pd.isna(ao_code):
        return 'all'
    ao_code = str(ao_code).lower()
    if '23r' in ao_code:
        return 'radius'
    elif '23u' in ao_code:
        return 'ulna'
    elif '23m' in ao_code or '-m' in ao_code:
        return 'metaphyseal'
    elif 'scaphoid' in ao_code:
        return 'scaphoid'
    else:
        return 'all'

def age_group(age):
    try:
        age = float(age)
        if age < 5:
            return 0
        elif age < 10:
            return 1
        elif age < 15:
            return 2
        else:
            return 3
    except:
        return 'unknown'

def get_combined_dataset_ao(projection=None):
    records = []

    kaggle_path = "/mnt/data/KimJG/ELBOW_test/Kaggle_dataset"
    kaggle_csv = os.path.join(kaggle_path, "dataset.csv")
    df = pd.read_csv(kaggle_csv)
    df = df[df["metal"] != 1]

    # ✅ projection 필터링: 1(AP), 2(Lateral)
    if projection in [1, 2]:
        df = df[df["projection"] == projection]

    # ✅ 이미지 경로 연결
    df["image_path"] = df["filestem"].apply(
        lambda f: next(iter(glob(os.path.join(kaggle_path, "images_part*", f"{f}.*"))), None)
    )
    df = df[df["image_path"].notnull()]

    # ✅ 정보 파생
    df["part"] = df["ao_classification"].apply(extract_part)
    df["age_group"] = df["age"].apply(age_group)
    df = df[df["age_group"] != "unknown"]
    df["fracture_visible"] = df["fracture_visible"].fillna(0).astype(int)
    df["label"] = df["fracture_visible"]
    df["source"] = "kaggle"
    df["gender"] = df["gender"].fillna("U")
    df["ao_subtypes"] = df["ao_classification"].apply(extract_ao_subtypes)
    df["ao_primary"] = df["ao_subtypes"].apply(lambda x: x[0] if x else "Unknown")

    # ✅ split 부여
    train_kaggle, val_kaggle = train_test_split(
        df, test_size=0.2, stratify=df["label"], random_state=42
    )
    train_kaggle["split"] = "train"
    val_kaggle["split"] = "val"
    records.append(pd.concat([train_kaggle, val_kaggle], ignore_index=True))

    all_df = pd.concat(records, ignore_index=True)

    def determine_final_class(row):
        if row["fracture_visible"] == 0:
            return "Normal"
        elif is_growth_plate_fracture(row["ao_primary"]):
            return "GrowthPlate_Fx"
        else:
            return "Fracture"

    all_df["final_class"] = all_df.apply(determine_final_class, axis=1)

    if all_df.empty:
        raise ValueError("❌ all_df is empty! 데이터가 하나도 로드되지 않았습니다.")

    return all_df
