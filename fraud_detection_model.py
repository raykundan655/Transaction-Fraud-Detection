import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

DATASET_PATH = "data/paysim.csv" if os.path.exists("data/paysim.csv") else "PS_20174392719_1491204439457_log.csv"
MODEL_PATH = "fraud_model.pkl"
RANDOM_STATE = 42


def load_and_explore_data(filepath):
    print("Loading dataset from:", filepath)
    df = pd.read_csv(filepath)

    print("Dataset Overview")
    print("Shape:", df.shape)
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("\nDuplicate Rows:", df.duplicated().sum())

    print("\nSummary Statistics:")
    print(df.describe().round(2))

    # Target variable distribution
    fraud_counts = df["isFraud"].value_counts()
    fraud_pct = (df["isFraud"].mean()) * 100
    print("\n--- Target Distribution (isFraud) ---")
    print(fraud_counts)
    print(f"Fraud Percentage: {fraud_pct:.4f}%")

    # Transaction types and amount analysis
    print("\n--- Transaction Types Breakdown ---")
    type_summary = df.groupby("type")["isFraud"].agg(["count", "sum", "mean"]).reset_index()
    type_summary.columns = ["type", "total_transactions", "fraud_count", "fraud_rate"]
    type_summary["fraud_rate_%"] = type_summary["fraud_rate"] * 100
    print(type_summary)

    print("\n--- Transaction Amounts by Class ---")
    print(df.groupby("isFraud")["amount"].describe().round(2))

    # EDA Visualizations (using a sample for fast plotting)
    print("\nGenerating EDA plots...")
    sample_legit = df[df["isFraud"] == 0].sample(n=min(50000, (df["isFraud"] == 0).sum()), random_state=RANDOM_STATE)
    sample_fraud = df[df["isFraud"] == 1]
    plot_df = pd.concat([sample_legit, sample_fraud])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Fraud vs Legitimate count
    sns.countplot(ax=axes[0, 0], data=df, x="isFraud", palette=["#2ecc71", "#e74c3c"])
    axes[0, 0].set_title("Fraud vs Legitimate Count (Log Scale)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticklabels(["Legitimate (0)", "Fraud (1)"])

    # 2. Transaction Type distribution
    sns.countplot(ax=axes[0, 1], data=df, x="type", order=df["type"].value_counts().index, palette="Blues_r")
    axes[0, 1].set_title("Transaction Types Distribution")

    # 3. Transaction Amount distribution (Log scale)
    sns.kdeplot(
        ax=axes[1, 0],
        data=plot_df,
        x=np.log10(plot_df["amount"].clip(lower=1)),
        hue="isFraud",
        common_norm=False,
        palette=["#2ecc71", "#e74c3c"],
        fill=True
    )
    axes[1, 0].set_title("Transaction Amount Density (Log10)")
    axes[1, 0].set_xlabel("Log10(Amount)")

    # 4. Correlation heatmap for numerical columns
    num_cols = ["step", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "isFraud"]
    corr = plot_df[num_cols].corr()
    sns.heatmap(ax=axes[1, 1], data=corr, annot=True, fmt=".2f", cmap="coolwarm")
    axes[1, 1].set_title("Numerical Features Correlation")

    plt.tight_layout()
    plt.savefig("eda_summary.png")
    print("Saved EDA plots to 'eda_summary.png'")

    return df


def clean_data(df):
    print("\n--- Cleaning Data ---")
    initial_len = len(df)

    # Remove any missing values and duplicates
    df = df.dropna().drop_duplicates()

    # Verify no invalid negative values for amount and balances
    valid_mask = (
        (df["amount"] >= 0)
        & (df["oldbalanceOrg"] >= 0)
        & (df["newbalanceOrig"] >= 0)
        & (df["oldbalanceDest"] >= 0)
        & (df["newbalanceDest"] >= 0)
    )
    df = df[valid_mask]
    print(f"Removed {initial_len - len(df)} invalid/duplicate rows. Remaining: {len(df):,}")

    return df


def engineer_features(df):
    df_feat = df.copy()

    # 1. Difference between sender expected balance change and actual change
    df_feat["orig_balance_diff"] = (
        df_feat["oldbalanceOrg"] - df_feat["newbalanceOrig"] - df_feat["amount"]
    )

    # 2. Difference between receiver expected balance change and actual change
    df_feat["dest_balance_diff"] = (
        df_feat["newbalanceDest"] - df_feat["oldbalanceDest"] - df_feat["amount"]
    )

    # 3. Flag if sender's balance was completely emptied to zero
    df_feat["is_orig_zero"] = (df_feat["newbalanceOrig"] == 0).astype(int)

    # 4. Relative amount transferred compared to old balance
    df_feat["amount_to_oldbalance_orig"] = (
        df_feat["amount"] / (df_feat["oldbalanceOrg"] + 1.0)
    )

    return df_feat


def build_preprocessor(num_cols, cat_cols):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor


def tune_hyperparameters(preprocessor, X_train, y_train, scale_pos_weight):
    print("\n--- Tuning Hyperparameters on Stratified Sample ---")

    # Sample 50,000 records from training data for fast tuning
    sample_size = min(50000, len(X_train))
    _, X_tune, _, y_tune = train_test_split(
        X_train, y_train,
        test_size=sample_size,
        stratify=y_train,
        random_state=RANDOM_STATE
    )

    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])

    param_dist = {
        "classifier__n_estimators": [60, 100, 140],
        "classifier__max_depth": [4, 6],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8, 1.0],
        "classifier__colsample_bytree": [0.8, 1.0]
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=4,
        cv=cv,
        scoring="average_precision",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    search.fit(X_tune, y_tune)
    print(f"Best Tuning PR-AUC: {search.best_score_:.4f}")
    print("Best Parameters:")
    best_params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    for k, v in best_params.items():
        print(f"  {k}: {v}")

    return best_params


def evaluate_model(pipeline, X_test, y_test):
    print("\n--- Evaluating Model on Test Set ---")
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)

    print(f"Accuracy:  {acc:.6f}")
    print(f"Precision: {prec:.6f}")
    print(f"Recall:    {rec:.6f}")
    print(f"F1-Score:  {f1:.6f}")
    print(f"ROC-AUC:   {roc_auc:.6f}")
    print(f"PR-AUC:    {pr_auc:.6f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Confusion matrix & curves visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legitimate", "Fraud"])
    disp.plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Confusion Matrix")

    # Precision-Recall Curve
    precisions, recalls, _ = precision_recall_curve(y_test, y_prob)
    axes[1].plot(recalls, precisions, color="#e67e22", lw=2, label=f"PR-AUC = {pr_auc:.4f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[2].plot(fpr, tpr, color="#2980b9", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title("ROC Curve")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("model_evaluation.png")
    print("Saved evaluation plots to 'model_evaluation.png'")


def main():
    # 1. Load and explore
    df = load_and_explore_data(DATASET_PATH)

    # 2. Clean data
    df = clean_data(df)

    # 3. Feature selection & Feature engineering
    feature_cols = ["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
    X = engineer_features(df[feature_cols])
    y = df["isFraud"]

    cat_cols = ["type"]
    num_cols = [c for c in X.columns if c not in cat_cols]

    # 4. Train-test split (80:20 stratified)
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )
    print(f"Train samples: {len(X_train):,}, Test samples: {len(X_test):,}")

    # 5. Handle class imbalance
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = float(np.sqrt(neg_count / pos_count))
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # 6. Build preprocessor
    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 7. Tune hyperparameters on sample
    best_params = tune_hyperparameters(preprocessor, X_train, y_train, scale_pos_weight)

    # 8. Train final pipeline on full training data
    print("\n--- Training Final Model on Full Training Set ---")
    final_pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=best_params.get("n_estimators", 100),
            max_depth=best_params.get("max_depth", 6),
            learning_rate=best_params.get("learning_rate", 0.05),
            subsample=best_params.get("subsample", 0.8),
            colsample_bytree=best_params.get("colsample_bytree", 0.8),
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            n_jobs=-1
        ))
    ])

    final_pipeline.fit(X_train, y_train)

    # 9. Evaluate final model on test set
    evaluate_model(final_pipeline, X_test, y_test)

    # 10. Save complete pipeline
    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(final_pipeline, MODEL_PATH)
    print("Model successfully saved!")


if __name__ == "__main__":
    main()
