import pandas as pd
import numpy as np
import pickle
import time
import warnings

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from preprocessing import IDSPreprocessor  # <-- now imported from separate module

warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Column names for NSL-KDD dataset
COLUMN_NAMES = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "label",
    "difficulty",
]


def load_data(train_path, test_path, sample_fraction=0.2):
    """Load and prepare NSL-KDD dataset"""
    start_time = time.time()

    print("  Reading training data...")
    train_df = pd.read_csv(
        train_path,
        names=COLUMN_NAMES,
        header=None,
        dtype={"label": str},
    )

    print("  Reading test data...")
    test_df = pd.read_csv(
        test_path,
        names=COLUMN_NAMES,
        header=None,
        dtype={"label": str},
    )

    load_time = time.time() - start_time
    print(f"  Data loading time: {load_time:.3f}s")

    # Drop difficulty column
    if "difficulty" in train_df.columns:
        train_df = train_df.drop("difficulty", axis=1)
    if "difficulty" in test_df.columns:
        test_df = test_df.drop("difficulty", axis=1)

    # Binary classification: normal vs attack
    def map_label(x):
        x = str(x).strip().lower()
        if x.startswith("normal"):
            return "Normal"
        else:
            return "Attack"

    print("\n  Label distribution (training):")
    train_df["label"] = train_df["label"].apply(map_label)
    print(
        "    Normal: ",
        (train_df["label"] == "Normal").sum(),
        f"({(train_df['label'] == 'Normal').mean() * 100:.1f}%)",
    )
    print(
        "    Attack: ",
        (train_df["label"] == "Attack").sum(),
        f"({(train_df['label'] == 'Attack').mean() * 100:.1f}%)",
    )

    print("\n  Label distribution (test):")
    test_df["label"] = test_df["label"].apply(map_label)
    print(
        "    Normal: ",
        (test_df["label"] == "Normal").sum(),
        f"({(test_df['label'] == 'Normal').mean() * 100:.1f}%)",
    )
    print(
        "    Attack: ",
        (test_df["label"] == "Attack").sum(),
        f"({(test_df['label'] == 'Attack').mean() * 100:.1f}%)",
    )

    # Subsample train data for faster experiments
    print(f"\n  Creating {sample_fraction * 100:.1f}% stratified subsample...")
    sample_start = time.time()

    if train_df["label"].nunique() > 1:
        train_df, _ = train_test_split(
            train_df,
            train_size=sample_fraction,
            stratify=train_df["label"],
            random_state=RANDOM_SEED,
        )
    else:
        train_df = train_df.sample(
            frac=sample_fraction, random_state=RANDOM_SEED
        )

    sample_time = time.time() - sample_start
    print(f"  Sampling time: {sample_time:.3f}s")

    return train_df, test_df, load_time


def prepare_features(train_df, test_df):
    """Split into features and labels"""
    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    print(f"\n  Training samples: {len(X_train)}")
    print(f"  Testing samples: {len(X_test)}")
    print(f"  Attack ratio (train): {(y_train == 'Attack').mean() * 100:.2f}%")
    print(f"  Attack ratio (test): {(y_test == 'Attack').mean() * 100:.2f}%")

    return X_train, X_test, y_train, y_test


def train_model(X_train_processed, y_train):
    """Train Random Forest model"""
    print("\n[3] Training Random Forest model...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )

    start_time = time.time()
    model.fit(X_train_processed, y_train)
    training_time = time.time() - start_time

    print(f"  Model training time: {training_time:.3f}s")
    print(f"  Number of trees: {len(model.estimators_)}")
    print(f"  Classes seen by model: {model.classes_}")

    return model, training_time


def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance"""
    print("\n[4] Evaluating model performance...")

    # Training set performance
    train_start = time.time()
    train_preds = model.predict(X_train)
    train_time = time.time() - train_start

    # Test set performance
    test_start = time.time()
    test_preds = model.predict(X_test)
    test_time = time.time() - test_start

    metrics = {
        "train_accuracy": accuracy_score(y_train, train_preds),
        "test_accuracy": accuracy_score(y_test, test_preds),
        "precision": precision_score(
            y_test, test_preds, pos_label="Attack", zero_division=0
        ),
        "recall": recall_score(
            y_test, test_preds, pos_label="Attack", zero_division=0
        ),
        "f1_score": f1_score(
            y_test, test_preds, pos_label="Attack", zero_division=0
        ),
        "train_time": train_time,
        "test_time": test_time,
    }

    # --- Safe ROC-AUC calculation ---
    metrics["roc_auc"] = None
    if hasattr(model, "predict_proba") and len(getattr(model, "classes_", [])) > 1:
        classes = list(model.classes_)
        if "Attack" in classes:
            attack_idx = classes.index("Attack")
            test_probs = model.predict_proba(X_test)[:, attack_idx]
            metrics["roc_auc"] = roc_auc_score(
                (y_test == "Attack").astype(int), test_probs
            )
        else:
            print("  [WARN] 'Attack' not found in model.classes_, skipping ROC-AUC.")
    else:
        print("  [WARN] Model only has one class or no predict_proba; ROC-AUC skipped.")

    cm = confusion_matrix(y_test, test_preds, labels=["Normal", "Attack"])

    print(f"  Train accuracy: {metrics['train_accuracy']:.4f}")
    print(f"  Test accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"  Precision:      {metrics['precision']:.4f}")
    print(f"  Recall:         {metrics['recall']:.4f}")
    print(f"  F1-score:       {metrics['f1_score']:.4f}")
    if metrics["roc_auc"] is not None:
        print(f"  ROC-AUC:        {metrics['roc_auc']:.4f}")

    print("\n  Confusion Matrix (Test):")
    print("                 Predicted Normal    Predicted Attack")
    print(f"  Actual Normal       {cm[0, 0]:5d}              {cm[0, 1]:5d}")
    print(f"  Actual Attack       {cm[1, 0]:5d}              {cm[1, 1]:5d}")

    return metrics, cm


def benchmark_latency(model, X_sample):
    """Benchmark inference latency"""
    print("\n[5] Benchmarking inference latency...")

    n_samples = len(X_sample)
    print(f"  Number of samples for latency test: {n_samples}")

    start_time = time.time()
    _ = model.predict(X_sample)
    total_time = time.time() - start_time

    avg_latency_ms = (total_time / n_samples) * 1000 if n_samples > 0 else 0
    throughput = n_samples / total_time if total_time > 0 else float("inf")

    print(f"  Total inference time: {total_time:.4f}s")
    print(f"  Average latency per sample: {avg_latency_ms:.4f} ms")
    print(f"  Throughput: {throughput:.2f} samples/second")

    performance = {
        "total_inference_time": total_time,
        "avg_latency_ms": avg_latency_ms,
        "throughput_samples_per_second": throughput,
    }

    return performance


def save_model(preprocessor, model, metrics, performance, filename="model.pkl"):
    """Save preprocessor and model"""
    print(f"\n[6] Saving model to '{filename}'...")

    model_data = {
        "preprocessor": preprocessor,
        "model": model,
        "metrics": metrics,
        "performance": performance,
    }

    with open(filename, "wb") as f:
        pickle.dump(model_data, f)

    print("  Model saved successfully.")


def main():
    print("============================================================")
    print("ML-Based Intrusion Detection System - Training Pipeline")
    print("============================================================\n")

    train_path = "data/KDDTrain.csv"
    test_path = "data/KDDTest.csv"

    print("[1] Loading NSL-KDD dataset...")
    train_df, test_df, load_time = load_data(
        train_path, test_path, sample_fraction=0.2
    )

    X_train, X_test, y_train, y_test = prepare_features(train_df, test_df)

    print("\n[2] Preprocessing data...")
    preprocessor = IDSPreprocessor()

    # Fit + transform training data
    X_train_processed = preprocessor.fit_transform(X_train)

    # Transform test data
    X_test_processed, transform_time, latency_per_sample = preprocessor.transform(
        X_test
    )

    print(
        f"  Feature preprocessing time (test set): {transform_time:.3f}s"
    )
    print(
        f"  Average preprocessing latency per sample: {latency_per_sample:.4f} ms"
    )

    model, training_time = train_model(X_train_processed, y_train)

    metrics, cm = evaluate_model(
        model, X_train_processed, X_test_processed, y_train, y_test
    )

    performance = benchmark_latency(model, X_test_processed)

    # Add times
    performance["data_loading_time"] = load_time
    performance["feature_preprocessing_time"] = transform_time
    performance["model_training_time"] = training_time

    save_model(preprocessor, model, metrics, performance, filename="model.pkl")

    print("\n[7] Training pipeline completed successfully.")
    print("============================================================")


if __name__ == "__main__":
    main()
