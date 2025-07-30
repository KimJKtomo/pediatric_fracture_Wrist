---

## ✅ `README.md` (최신)

```markdown
# 🦴 Pediatric Wrist Fracture Classification (AP / Lateral View)

This repository provides a complete pipeline for classifying pediatric wrist fractures using **ConvNeXtV2**, with models trained separately for:
- **Projection Type**: AP vs Lateral
- **Age Group**: 0–4 / 5–9 / 10–14 / 15–19

---

## 📁 Folder Structure: `WristFX_0730/`

| File | Description |
|------|-------------|
| `age_split_testset_0730.py` | Creates fixed AP / Lat test sets by age (20 bins × 2 classes each) |
| `generate_age_trainval_split_0730.py` | Splits train/val data by projection (AP/Lat) and age group |
| `train_ddp_fracture_per_agegroup_convnextv2_0730.py` | Trains 8 models via DDP (AP & Lat × Age 4 groups) |
| `Generate_Gradcam_ConvNeXtV2_gradcampp.py` | Visualizes Grad-CAM++ heatmaps for each model |
| `run_all_training_0730.py` | Unified launcher for test split, train split, training, and Grad-CAM |
| `load_new_dxmodule_0730.py` | Dataset loader that filters by projection, age, and AO classification |

---

## 🧪 Model Outputs

- Trained model checkpoints:
```

best\_ddp\_convnextv2\_AP\_0.pt  →  best\_ddp\_convnextv2\_Lat\_3.pt

```
- Grad-CAM results:
```

WristFX\_0730/gradcam\_results\_0730/AP/agegroup0/...
WristFX\_0730/gradcam\_results\_0730/Lat/agegroup3/...

````

---

## 🚀 Quickstart

```bash
# Step into the working directory
cd WristFX_0730

# Run full pipeline (test split → train split → model training → Grad-CAM)
python run_all_training_0730.py
````

---

## 🧠 Age Group Strategy

| Group | Age Range (Years) |
| ----- | ----------------- |
| 0     | 0–4               |
| 1     | 5–9               |
| 2     | 10–14             |
| 3     | 15–19             |

---

## 🛠 Requirements

* Python ≥ 3.8
* PyTorch ≥ 1.13
* TIMM
* Grad-CAM (`pytorch-grad-cam`)
* scikit-learn, pandas, matplotlib, tqdm
* (optional) MLflow for logging

---

## 📈 Results & Evaluation

* Each age/projection model is evaluated with:

  * Accuracy / F1
  * Confusion Matrix
  * Grad-CAM++ overlay image per sample
* Results saved in `gradcam_results_0730/` per projection and age group

---

## 👨‍⚕️ Author

Developed by **[KimJKtomo](https://github.com/KimJKtomo)**
Specialized in pediatric medical AI using multimodal X-ray fracture classification

```

---

원하시면 아래 항목도 추가 가능해요:

- `📊 Result Table` (성능 요약 표)
- `📂 Example Inference Script`
- `📦 requirements.txt` 자동 생성

필요하시면 말씀 주세요!
```

