# Notebook: train_bga_models.ipynb

## Tổng Quan

Notebook hoàn chỉnh để training và evaluating models cho BGA-AISNP classification dựa trên `merged_matrix.csv`.

## Cấu Trúc Notebook (34 cells)

### 📚 Phần 1: Thiết Lập & Dữ Liệu (Cells 1-4)
- Cell 1: Tiêu đề chính
- Cell 2: Import thư viện cần thiết (pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn)
- Cell 3: Load merged_matrix.csv
- Cell 4: Khám phá dữ liệu (shape, metadata, class distribution)

### 🔧 Phần 2: Tiền Xử Lý (Cells 5-7)
- Cell 5: Xử lý missing values (imputation với median)
- Cell 6: Encode labels cho continental và population
- Cell 7: Kiểm tra class imbalance

### 📊 Phần 3: Chia Data & Training (Cells 8-15)
- Cell 8: Stratified 80/20 train/test split (random_state=42)
- Cell 9: Train XGBoost cho continental ancestry (24 SNPs)
- Cell 10: Train XGBoost cho East Asian populations (34 SNPs)
- Cell 11: Implement Generative Bayesian Model class
- Cell 12: Train generative models cho cả 2 tầng

### 📈 Phần 4: Đánh Giá (Cells 13-16)
- Cell 13: Classification report & confusion matrix cho continental (XGBoost + Generative)
- Cell 14: Classification report & confusion matrix cho East Asian (XGBoost + Generative)
- Cell 15: Confusion matrix heatmaps (4 biểu đồ)
- Cell 16: Feature importance plots (top 15 SNPs cho mỗi stage)
- Cell 17: Model performance comparison bars
  
### 🎯 Phần 5: Inference Pipeline (Cells 18-20)
- Cell 18: Two-stage XGBoost inference function
- Cell 19: Two-stage Generative Bayesian inference với uncertainty estimation
- Cell 20: Example predictions

### 💾 Phần 6: Lưu Artifacts (Cell 21)
- Cell 21: Save models, label encoders, imputer, và results summary

## Kết Quả Đầu Ra

### Models Được Lưu:
```
models/
├── continent_xgb_merged.pkl
├── continent_label_encoder_merged.pkl
├── continent_snp_names_merged.pkl
├── continent_gen_model_merged.pkl
├── eastasia_xgb_merged.pkl
├── eastasia_label_encoder_merged.pkl
├── eastasia_gen_model_merged.pkl
└── imputer_merged.pkl
```

### Reports Được Lưu:
```
reports/
├── confusion_matrices.png       (4 subplots)
├── feature_importance.png       (top SNPs)
├── model_comparison.png         (accuracy comparison)
└── training_results_merged.json (results summary)
```

## Tính Năng Chính

✅ **Two-Stage Classification Pipeline**
- Tầng 1: Continental (EAS/EUR/AFR/AMR/SAS)
- Tầng 2: East Asian populations (CHB/CHS/CDX/KHV)

✅ **Hai Loại Mô Hình**
- XGBoost: Gradient boosting, hiệu suất cao
- Generative Bayesian: Binomial likelihood, uncertainty estimation

✅ **Comprehensive Evaluation**
- Accuracy, Precision, Recall, F1-score
- Confusion matrices
- Classification reports
- Feature importance analysis

✅ **Inference Functions**
- Two-stage XGBoost prediction
- Two-stage Generative with confidence scores
- Posterior probabilities cho tất cả classes

## Cách Sử Dụng

1. **Open notebook**: `train_bga_models.ipynb`
2. **Run all cells**: Hoặc chạy từng section một
3. **Models & results** được lưu tự động vào `models/` và `reports/`

## Dependencies

- pandas >= 2.0
- numpy >= 1.24
- scikit-learn >= 1.3, < 1.6
- xgboost >= 2.0
- matplotlib >= 3.8
- seaborn >= 0.13

## Workflow Tuân Theo

✓ Load dữ liệu original từ merged_matrix.csv
✓ Xử lý genotypes: 0/1/2 encoding
✓ Stratified train/test split (80/20)
✓ Train XGBoost & Generative models
✓ Evaluate trên test set
✓ Analyze feature importance
✓ Two-stage inference pipeline
✓ Save artifacts

