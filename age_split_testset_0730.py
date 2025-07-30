# WristFX_0730/age_split_testset_0730.py
import os
import shutil
import pandas as pd
from load_new_dxmodule_0730 import get_combined_dataset_ao

# 현재 경로 기준 설정
BASE_DIR = os.path.dirname(__file__)
OUTPUT_DIR = os.path.join(BASE_DIR, "test_set_images_0730")
YEARS = list(range(0, 20))
N_PER_CLASS = 10

def process_projection_split(df_proj, proj_name):
    output_proj_dir = os.path.join(OUTPUT_DIR, f"Test_{proj_name}")
    os.makedirs(output_proj_dir, exist_ok=True)
    test_rows = []

    for i in YEARS:
        bin_df = df_proj[(df_proj['age'] >= i) & (df_proj['age'] < i + 1)]
        if len(bin_df) == 0:
            continue

        frac = bin_df[bin_df["fracture_visible"] == 1]
        norm = bin_df[bin_df["fracture_visible"] == 0]

        if len(frac) == 0 or len(norm) == 0:
            print(f"⚠️ Age {i}~{i+1} ({proj_name}): 한쪽 클래스 부족")
            continue

        sampled = pd.concat([
            frac.sample(min(N_PER_CLASS, len(frac)), random_state=42),
            norm.sample(min(N_PER_CLASS, len(norm)), random_state=42)
        ])

        age_dir = os.path.join(output_proj_dir, f"age_{i}_{i+1}")
        os.makedirs(age_dir, exist_ok=True)
        for _, row in sampled.iterrows():
            if os.path.exists(row["image_path"]):
                fname = os.path.basename(row["image_path"])
                shutil.copy(row["image_path"], os.path.join(age_dir, fname))

        test_rows.append(sampled)

    if test_rows:
        final_df = pd.concat(test_rows, ignore_index=True)
        csv_path = os.path.join(BASE_DIR, f"test_set_0730_{proj_name}.csv")
        final_df[[
            'filestem', 'image_path', 'age', 'fracture_visible',
            'gender', 'ao_primary', 'ao_subtypes'
        ]].to_csv(csv_path, index=False)
        print(f"✅ Test set ({proj_name}) 저장 완료 → {csv_path} | 총 {len(final_df)}개")
        return csv_path
    else:
        print(f"❌ No samples saved for projection {proj_name}")
        return None

# ✅ AP / Lat 분리 및 처리
df_ap = get_combined_dataset_ao(projection=1)
df_lat = get_combined_dataset_ao(projection=2)

csv_ap = process_projection_split(df_ap, "AP")
csv_lat = process_projection_split(df_lat, "Lat")

# ✅ age_test.csv 백업 (AP 기준)
if csv_ap:
    try:
        shutil.copy(csv_ap, os.path.join(BASE_DIR, "age_test.csv"))
        print("📁 Saved age_test.csv (AP 기준) for fixed test set.")
    except Exception as e:
        print(f"❌ Failed to copy age_test.csv: {e}")
