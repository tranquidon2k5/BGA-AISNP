# scripts/train_generative_bga.py
import os
import sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src.data_utils import (
    load_continental_csv,
    load_eastasian_csv,
    encode_genotypes,
    split_xy,
)
from src.generative_model import GenerativeBGAModel

import joblib


DATA_CONT_PATH = os.path.join("data", "AISNP_by_sample_continental.csv")
DATA_EAS_PATH = os.path.join("data", "AISNP_by_sample_eastasian.csv")
MODELS_DIR = "models"


def eval_task(name, model, X_train, y_train, X_test, y_test):
    """
    Fit model trên train, evaluate trên test.
    """
    model.fit(X_train, y_train, snp_names=model.snp_names)  # snp_names đã set trước
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print(f"Accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return acc


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ======================
    # Task 1: Continental
    # ======================
    df_cont_raw = load_continental_csv(DATA_CONT_PATH)
    df_cont_enc, cont_snp_names = encode_genotypes(df_cont_raw)

    print(f"Continental encoded shape: {df_cont_enc.shape}")
    print(f"SNPs (continental panel): {len(cont_snp_names)}")

    Xc, yc = split_xy(df_cont_enc, cont_snp_names, label_col="super_pop")
    Xc = Xc.astype(float)

    # train/test split stratified theo super_pop
    Xc_train, Xc_test, yc_train, yc_test = train_test_split(
        Xc,
        yc,
        test_size=0.2,
        random_state=42,
        stratify=yc,
    )

    cont_model = GenerativeBGAModel(smoothing_alpha=1.0)
    cont_model.snp_names = cont_snp_names  # set trước để fit() dùng
    acc_cont = eval_task(
        "Generative Continental BGA", cont_model, Xc_train, yc_train, Xc_test, yc_test
    )

    # Train lại trên full data để lưu
    cont_model.fit(Xc, yc, snp_names=cont_snp_names)
    joblib.dump(cont_model, os.path.join(MODELS_DIR, "continent_gen_model.pkl"))
    print("Saved Generative Continental model -> models/continent_gen_model.pkl")

    # ======================
    # Task 2: East Asia subpop
    # ======================
    df_eas_raw = load_eastasian_csv(DATA_EAS_PATH)
    df_eas_enc, eas_snp_names = encode_genotypes(df_eas_raw)

    print(f"\nEastAsia encoded shape (all): {df_eas_enc.shape}")
    print(f"SNPs (East Asia panel): {len(eas_snp_names)}")

    # Lọc chỉ các mẫu có super_pop == 'EAS'
    df_eas = df_eas_enc[df_eas_enc["super_pop"] == "EAS"].copy()
    print(f"EastAsia only shape: {df_eas.shape}")
    print("Population counts:")
    print(df_eas["pop"].value_counts())

    Xe, ye = split_xy(df_eas, eas_snp_names, label_col="pop")
    Xe = Xe.astype(float)

    Xe_train, Xe_test, ye_train, ye_test = train_test_split(
        Xe,
        ye,
        test_size=0.2,
        random_state=42,
        stratify=ye,
    )

    eas_model = GenerativeBGAModel(smoothing_alpha=1.0)
    eas_model.snp_names = eas_snp_names
    acc_eas = eval_task(
        "Generative EastAsia BGA", eas_model, Xe_train, ye_train, Xe_test, ye_test
    )

    # Train full để lưu
    eas_model.fit(Xe, ye, snp_names=eas_snp_names)
    joblib.dump(eas_model, os.path.join(MODELS_DIR, "eastasia_gen_model.pkl"))
    print("Saved Generative EastAsia model -> models/eastasia_gen_model.pkl")

    print("\n=== Summary ===")
    print(f"Continental generative acc: {acc_cont:.4f}")
    print(f"EastAsia generative acc:    {acc_eas:.4f}")


if __name__ == "__main__":
    main()
