import os
import subprocess

# 현재 파일 기준 폴더 경로
BASE_DIR = os.path.dirname(__file__)

def run_step(desc, command_list):
    print(f"\n📌 {desc}")
    subprocess.run(command_list, check=True)

if __name__ == '__main__':
    # Step 1: Test Set 생성 (AP / Lat)
    age_test_path = os.path.join(BASE_DIR, "age_test.csv")
    if not os.path.exists(age_test_path):
        run_step("Step 1: Creating test set...", ["python", os.path.join(BASE_DIR, "age_split_testset_0730.py")])
    else:
        print("✅ Step 1: age_test.csv already exists. Skipping test set creation.")

    # Step 2: Train / Val 분할
    run_step("Step 2: Generating train/val splits...", ["python", os.path.join(BASE_DIR, "generate_age_trainval_split_0730.py")])

    # Step 3: 8개 모델 학습 (torchrun)
    run_step("Step 3: Training all AP/Lat × AgeGroup models...",
             ["torchrun", "--nproc_per_node=2", os.path.join(BASE_DIR, "train_ddp_fracture_per_agegroup_convnextv2_0730.py")])

    # Step 4: Grad-CAM + 결과 저장
    run_step("Step 4: Generating Grad-CAM visualizations...",
             ["python", os.path.join(BASE_DIR, "Generate_Gradcam_ConvNeXtV2_gradcampp.py")])

    print("\n🎉 All training and evaluation steps completed successfully!")
