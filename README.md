# BGA-AISNP: Phân Loại Nguồn Gốc Địa Lý Sinh Học Dựa Trên Ancestry Informative SNPs

Dự án này thực hiện phân loại nguồn gốc địa lý sinh học (Biogeographic Ancestry - BGA) sử dụng các Ancestry Informative Single Nucleotide Polymorphisms (AISNP) thông qua mô hình học máy hai tầng: phân loại châu lục (continental) và phân loại quần thể Đông Á chi tiết.

## 📋 Tổng Quan

Hệ thống sử dụng hai mô hình phân cấp:

1. **Tầng 1 - Phân loại châu lục**: Dự đoán châu lục nguồn gốc (ví dụ: EAS, EUR, AFR, AMR, SAS)
2. **Tầng 2 - Phân loại quần thể Đông Á**: Nếu mẫu được dự đoán là Đông Á (EAS), mô hình sẽ phân loại chi tiết các quần thể con trong khu vực Đông Á

Dự án hỗ trợ hai loại mô hình:
- **XGBoost**: Mô hình gradient boosting mạnh mẽ cho phân loại đa lớp
- **Generative Bayesian Model**: Mô hình Bayes sinh đơn giản với khả năng ước lượng độ không chắc chắn

## ✨ Tính Năng

- 🧬 Phân loại hai tầng: châu lục → quần thể Đông Á
- 🎯 XGBoost với hyperparameter tuning
- 📊 Generative Bayesian model với uncertainty estimation
- 🔍 Feature importance analysis
- 📈 Đánh giá hiệu suất chi tiết (accuracy, classification report, confusion matrix)
- 💾 Pipeline inference hoàn chỉnh

## 📂 Dữ Liệu

- **RAW**: dữ liệu AISNP gốc (VCF, bảng panel và phụ lục paper) được lưu trong `data/1kgp_58AISNPs_*`, `data/1-s2.0-...xlsx`, `data/integrated_call_samples_v3.20130502.ALL.panel.txt`. Các file này dùng cho bước trích xuất và chưa encode thành số.
- **Đã xử lý**: `data/AISNP_by_sample_continental.csv` và `data/AISNP_by_sample_eastasian.csv` là đầu ra của `data/convert_aisnp_by_sample.py`, mỗi dòng là một sample cùng allele `_1/_2`. Các script train sẽ encode về 0/1/2 trước khi train.
- **Split**: repo không lưu sẵn train/dev/test; mỗi script train dùng `train_test_split(test_size=0.2, random_state=42, stratify=label)` để tạo train/test tạm thời. Thông tin chi tiết hơn xem `data.txt`.

## 📁 Cấu Trúc Dự Án

```
bga-aisnp/
├── data/                              # Dữ liệu SNP
│   ├── AISNP_by_sample_continental.csv
│   └── AISNP_by_sample_eastasian.csv
├── models/                            # Mô hình đã train
│   ├── continent_xgb.pkl
│   ├── continent_label_encoder.pkl
│   ├── continent_snp_names.pkl
│   ├── continent_gen_model.pkl
│   ├── eastasia_xgb.pkl
│   ├── eastasia_label_encoder.pkl
│   ├── eastasia_snp_names.pkl
│   └── eastasia_gen_model.pkl
├── scripts/                           # Scripts chính
│   ├── train_continental_xgb.py      # Train XGBoost cho châu lục
│   ├── train_eastasian_xgb.py        # Train XGBoost cho Đông Á
│   ├── train_generative_bga.py       # Train generative Bayesian model
│   ├── tune_eastasian_xgb.py         # Hyperparameter tuning
│   ├── inference_pipeline.py         # Pipeline inference hai tầng (XGBoost)
│   ├── inference_generative_pipeline.py  # Inference với generative model
│   ├── eval_generative_uncertainty.py    # Đánh giá uncertainty
│   └── feature_importance.py         # Phân tích feature importance
├── src/                               # Source code
│   ├── data_utils.py                 # Utilities xử lý dữ liệu
│   ├── models.py                     # XGBoost model definitions
│   └── generative_model.py           # Generative Bayesian model
├── requirements.txt                   # Python dependencies
└── README.md                          # Tài liệu này
```

## 🚀 Cài Đặt

### Yêu Cầu Hệ Thống

- Python >= 3.8
- pip hoặc conda

### Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

Các thư viện chính:
- `numpy >= 1.24`
- `pandas >= 2.0`
- `scikit-learn >= 1.3, < 1.6`
- `xgboost >= 2.0`
- `matplotlib >= 3.8` (optional, cho visualization)
- `seaborn >= 0.13` (optional, cho visualization)
- `jupyterlab >= 4.0` (optional, cho Jupyter notebooks)

## 📖 Hướng Dẫn Sử Dụng

### 1. Training Mô Hình

#### Train XGBoost cho Phân Loại Châu Lục

```bash
python scripts/train_continental_xgb.py
```

Script này sẽ:
- Đọc dữ liệu từ `data/AISNP_by_sample_continental.csv`
- Encode genotypes (0/1/2) từ allele pairs (_1/_2)
- Train XGBoost model với stratified train/test split
- Lưu model, label encoder và danh sách SNPs vào `models/`

#### Train XGBoost cho Phân Loại Đông Á

```bash
python scripts/train_eastasian_xgb.py
```

Script này chỉ train trên các mẫu có `super_pop == "EAS"` để phân loại các quần thể con trong Đông Á.

#### Hyperparameter Tuning (Tùy Chọn)

```bash
python scripts/tune_eastasian_xgb.py
```

Thực hiện grid search để tìm hyperparameters tối ưu cho mô hình Đông Á.

#### Train Generative Bayesian Model

```bash
python scripts/train_generative_bga.py
```

Train mô hình Bayesian đơn giản cho cả hai tầng. Mô hình này có thể ước lượng độ không chắc chắn (uncertainty) của dự đoán.

### 2. Inference

#### Inference Với XGBoost (Pipeline Hai Tầng)

```bash
python scripts/inference_pipeline.py
```

Hoặc sử dụng trong code:

```python
from scripts.inference_pipeline import predict_sample

# Dự đoán cho một sample
result = predict_sample("HG01168")
print(result)
```

Kết quả trả về:
```python
{
    'sample': 'HG01168',
    'continent_pred': 'EAS',
    'continent_probs': {'EAS': 0.95, 'EUR': 0.03, ...},
    'eastasia_subpop_pred': 'CHB',  # Chỉ có nếu continent_pred == 'EAS'
    'eastasia_probs': {'CHB': 0.87, 'JPT': 0.10, ...}
}
```

#### Inference Với Generative Model

```bash
python scripts/inference_generative_pipeline.py
```

Generative model cung cấp thêm khả năng xử lý missing data và uncertainty estimation.

### 3. Đánh Giá và Phân Tích

#### Đánh Giá Uncertainty (Generative Model)

```bash
python scripts/eval_generative_uncertainty.py
```

Đánh giá hiệu suất của mô hình generative khi xử lý các mẫu có độ không chắc chắn cao.

#### Phân Tích Feature Importance

```bash
python scripts/feature_importance.py
```

Xác định các SNPs quan trọng nhất cho việc phân loại.

#### Chạy Toàn Bộ & Xuất Báo Cáo

```bash
bash scripts/run_all_models.sh
```

Script này huấn luyện lại XGBoost + Generative Bayesian cho cả hai tầng, tính Accuracy, MCC, macro F1, AUC từng lớp, vẽ heatmap confusion matrix và xuất kết quả vào `reports/aggregated_results/model_metrics.xlsx`. Đây là cách nhanh nhất để tái lập toàn bộ bảng so sánh.

## 🔬 Mô Hình

### XGBoost

Mô hình XGBoost được cấu hình cho bài toán multi-class classification:

- **Objective**: `multi:softprob`
- **N_estimators**: 200
- **Max_depth**: 4
- **Learning_rate**: 0.1
- **Subsample**: 0.9
- **Colsample_bytree**: 0.9
- **Tree_method**: `hist` (tối ưu cho dữ liệu lớn)

### Generative Bayesian Model

Mô hình Bayesian đơn giản dựa trên allele frequencies:

- Ước lượng allele frequency `p_{k,j}` cho mỗi quần thể `k` và SNP `j`
- Sử dụng Beta prior với `alpha = 1.0` (uniform prior)
- Tính posterior probability sử dụng Bayes theorem
- Hỗ trợ missing data và uncertainty estimation

**Ưu điểm**:
- Xử lý được missing genotypes (np.nan)
- Cung cấp uncertainty scores
- Interpretable (dựa trên allele frequencies)

## 📊 Định Dạng Dữ Liệu

### Input CSV Format

File CSV phải chứa các cột:

1. **Metadata columns**:
   - `sample`: ID mẫu
   - `pop`: Quần thể (ví dụ: "CHB", "JPT", "CEU")
   - `super_pop`: Châu lục (ví dụ: "EAS", "EUR", "AFR")

2. **SNP columns**:
   - Mỗi SNP có 2 cột: `rsXXXX_1` và `rsXXXX_2` (allele pairs)
   - Giá trị có thể là allele bases (A, T, G, C) hoặc các giá trị missing

### Genotype Encoding

Mô hình tự động encode genotypes:
- **0**: Homozygous major allele (cả hai allele giống major allele)
- **1**: Heterozygous (một major, một minor allele)
- **2**: Homozygous minor allele (cả hai allele đều là minor)
- **np.nan**: Missing data

## 📈 Hiệu Suất

Mô hình được đánh giá với:
- **Accuracy**: Tỷ lệ dự đoán đúng
- **Classification Report**: Precision, Recall, F1-score cho từng lớp
- **Confusion Matrix**: Ma trận nhầm lẫn chi tiết

Để xem kết quả cụ thể, chạy các script training và kiểm tra output.

## 🔧 Tùy Chỉnh

### Thay Đổi Hyperparameters XGBoost

Sửa trong `src/models.py` hoặc override khi tạo model:

```python
from src.models import make_xgb_multiclass

model = make_xgb_multiclass(
    num_classes=5,
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05
)
```

### Thay Đổi Generative Model Smoothing

```python
from src.generative_model import GenerativeBGAModel

model = GenerativeBGAModel(smoothing_alpha=0.5)  # Tăng smoothing
```

## 📝 Lưu Ý

- Đảm bảo dữ liệu input có format đúng với các cột metadata và SNP columns
- Models được lưu dưới dạng `.pkl` sử dụng joblib
- Khi inference, cần đảm bảo sample ID tồn tại trong cả hai file CSV (continental và eastasian) nếu cần dự đoán subpopulation

## 🤝 Đóng Góp

Mọi đóng góp đều được chào đón! Vui lòng tạo issue hoặc pull request.

## 📄 License

[Thêm thông tin license nếu có]

## 👥 Tác Giả

[Thêm thông tin tác giả nếu cần]

---

**Lưu ý**: Dự án này phục vụ mục đích nghiên cứu. Việc sử dụng trong các ứng dụng lâm sàng hoặc pháp y cần được xem xét cẩn thận về tính đạo đức và pháp lý.
