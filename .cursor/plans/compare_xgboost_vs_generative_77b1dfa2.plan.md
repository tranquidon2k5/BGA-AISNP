---
name: Compare XGBoost vs Generative
overview: Tạo script mới để train và so sánh XGBoost với Generative model sử dụng 70/15/15 split (train/valid/test), đánh giá trên cả validation và test set.
todos:
  - id: create-compare-script
    content: Tạo scripts/compare_models.py với logic chia 70/15/15 và so sánh 2 model
    status: pending
---

# So sánh XGBoost và Generative Model với 70/15/15 Split

## Tổng quan

Tạo script mới `scripts/compare_models.py` để:

- Chia dữ liệu thành 70% train, 15% validation, 15% test (stratified)
- Train cả XGBoost và Generative model trên tập train
- Đánh giá trên cả validation và test set
- So sánh metrics và in kết quả

## Thay đổi chính

### 1. Tạo file mới: `scripts/compare_models.py`

```python
# Workflow:
# 1. Load và encode dữ liệu (continental + eastasia)
# 2. Chia 70/15/15 với stratified split
# 3. Train XGBoost và Generative model trên train set
# 4. Đánh giá trên valid và test set
# 5. In bảng so sánh metrics
```



### 2. Logic chia dữ liệu

Sử dụng 2 bước `train_test_split` từ sklearn:

- Bước 1: Chia 70% train, 30% temp (stratified)
- Bước 2: Chia temp thành 50% valid, 50% test (15%/15% tổng)

### 3. Metrics so sánh

Cho mỗi model (XGBoost, Generative) trên mỗi tập (valid, test):

- Accuracy
- Precision, Recall, F1-score (macro average)
- Classification report chi tiết

### 4. Output mong đợi

```javascript
=== Continental Classification ===
                    Valid Acc   Test Acc   Valid F1   Test F1
XGBoost             0.9xxx      0.9xxx     0.9xxx     0.9xxx
Generative          0.8xxx      0.8xxx     0.8xxx     0.8xxx

=== East Asian Subpopulation ===
                    Valid Acc   Test Acc   Valid F1   Test F1
XGBoost             0.7xxx      0.7xxx     0.7xxx     0.7xxx
Generative          0.6xxx      0.6xxx     0.6xxx     0.6xxx
```



## Files liên quan

- Tái sử dụng: [`src/data_utils.py`](src/data_utils.py) - load và encode dữ liệu
- Tái sử dụng: [`src/models.py`](src/models.py) - tạo XGBoost model