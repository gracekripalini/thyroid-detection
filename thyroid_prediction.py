
import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
OUTPUT_DIR = "thyroid_outputs"

mapping = {}


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    print(f"Loaded dataset with shape: {df.shape}")
    return df


def detect_target(df):
    for candidate in ["target", "class", "label", "diagnosis", "Result", "result"]:
        if candidate in df.columns:
            return candidate
    return df.columns[-1]


def plot_correlation_heatmap(df, numeric_features, outpath=None):
    if len(numeric_features) < 2:
        return
    corr = df[numeric_features].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Features)")
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def plot_confusion_heatmap(cm, classes, title="Confusion Matrix", outpath=None):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.title(title)
    if outpath:
        plt.savefig(outpath, bbox_inches="tight")
    plt.show()


def preprocess_dataframe(df, target_col):
    print("\n--- Basic Info ---")
    print(df.info())
    print("\n--- Missing Values ---")
    print(df.isnull().sum().sort_values(ascending=False).head(30))

    df = df.dropna(subset=[target_col]).copy()

    global mapping
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].astype(str).str.strip()
        unique_vals = sorted(df[target_col].unique())
        mapping = {v: i for i, v in enumerate(unique_vals)}
        print("Target mapping:", mapping)
        df[target_col] = df[target_col].map(mapping)

    X = df.drop(columns=[target_col]).copy()
    y = df[target_col].copy()

    bool_cols = X.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(int)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

    print(f"\nNumeric features ({len(numeric_features)}): {numeric_features[:15]}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = [("num", numeric_transformer, numeric_features)]
    if categorical_features:
        transformers.append(("cat", categorical_transformer, categorical_features))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    return X, y, preprocessor, numeric_features, categorical_features


def evaluate_model(model, X_test, y_test, model_name="Model", classes=None, save_plots_prefix=None):
    y_pred = model.predict(X_test)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
        except Exception:
            y_proba = None
    elif hasattr(model, "decision_function"):
        try:
            y_proba = model.decision_function(X_test)
        except Exception:
            y_proba = None

    acc = accuracy_score(y_test, y_pred)
    average_type = "binary" if y_test.nunique() == 2 else "weighted"
    prec = precision_score(y_test, y_pred, zero_division=0, average=average_type)
    rec = recall_score(y_test, y_pred, zero_division=0, average=average_type)
    f1 = f1_score(y_test, y_pred, zero_division=0, average=average_type)

    roc_auc = None
    if y_proba is not None and y_test.nunique() == 2:
        try:
            score = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            roc_auc = roc_auc_score(y_test, score)
            fpr, tpr, _ = roc_curve(y_test, score)
            plt.figure()
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
            plt.plot([0, 1], [0, 1], "k--")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC Curve - {model_name}")
            plt.legend()
            if save_plots_prefix:
                plt.savefig(f"{save_plots_prefix}_roc.png", bbox_inches="tight")
            plt.show()
        except Exception:
            pass

    cm = confusion_matrix(y_test, y_pred)
    class_names = classes if classes else [str(c) for c in sorted(np.unique(y_test))]
    plot_confusion_heatmap(
        cm,
        class_names,
        title=f"Confusion Matrix - {model_name}",
        outpath=f"{save_plots_prefix}_confusion.png" if save_plots_prefix else None,
    )

    print(classification_report(y_test, y_pred, zero_division=0))

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": roc_auc}


def interactive_prediction(final_model, feature_columns, mapping_inv=None):
    print("\n" + "=" * 50)
    print("--- Thyroid Disorder Screening ---")
    print("=" * 50)
    print("\nHow would you like to proceed?")
    print("1. Lab result-based diagnosis (Recommended)")
    print("2. Symptom-based preliminary check")

    choice = input("Enter your choice (1 or 2): ").strip()

    if choice == "2":
        print("\nPlease answer the following questions:")
        user_data = {col: 0 for col in feature_columns}
        for col in feature_columns:
            if col in ["age", "TSH", "T3", "TT4", "T4U", "FTI"]:
                user_data[col] = np.nan

        user_data["age"] = float(input("Enter your age: "))
        gender = input("Enter your gender (M/F): ").upper()
        user_data["sex"] = 1 if gender == "M" else 0

        symptoms = [
            "sick",
            "pregnant",
            "thyroid surgery",
            "I131 treatment",
            "query hypothyroid",
            "query hyperthyroid",
            "lithium",
            "goitre",
            "tumor",
            "hypopituitar",
            "psych",
        ]

        for symptom in symptoms:
            if symptom in user_data:
                user_data[symptom] = int(input(f"Do you have {symptom}? (1=Yes, 0=No): "))

        if "on thyroxine" in user_data:
            user_data["on thyroxine"] = int(input("Are you on thyroxine? (1=Yes, 0=No): "))
        if "on antithyroid medication" in user_data:
            user_data["on antithyroid medication"] = int(
                input("Are you on antithyroid medication? (1=Yes, 0=No): ")
            )

        symptom_df = pd.DataFrame([user_data])[feature_columns]
        print("\nAnalyzing your symptoms...")
        prediction = final_model.predict(symptom_df)[0]

    else:
        print("\nPlease enter your thyroid function test results:")
        user_data = {col: 0 for col in feature_columns}
        for col in feature_columns:
            if col in ["age"]:
                user_data[col] = np.nan

        user_data["age"] = float(input("Enter your age: "))
        gender = input("Enter your gender (M/F): ").upper()
        user_data["sex"] = 1 if gender == "M" else 0

        if "TSH" in user_data:
            user_data["TSH"] = float(input("Enter TSH level: "))
        if "T3" in user_data:
            user_data["T3"] = float(input("Enter T3 level: "))
        if "TT4" in user_data:
            user_data["TT4"] = float(input("Enter TT4 level: "))
        if "T4U" in user_data:
            user_data["T4U"] = float(input("Enter T4U value: "))
        if "FTI" in user_data:
            user_data["FTI"] = float(input("Enter FTI value: "))

        lab_df = pd.DataFrame([user_data])[feature_columns]
        print("\nAnalyzing your lab results...")
        prediction = final_model.predict(lab_df)[0]

    if mapping_inv and prediction in mapping_inv:
        result = mapping_inv[prediction]
    else:
        result = f"Class {prediction}"

    print(f"\n{'=' * 50}")
    print(f"AI Diagnosis: {result}")
    print(f"{'=' * 50}")

    result_lower = str(result).lower()
    if "normal" in result_lower or "negative" in result_lower:
        print("Your thyroid levels seem normal.")
    elif "hyper" in result_lower:
        print("Possible hyperthyroidism detected. Please consult an endocrinologist.")
    elif "hypo" in result_lower:
        print("Possible hypothyroidism detected. Please consult an endocrinologist.")
    else:
        print("Thyroid abnormality detected. Please consult a healthcare professional.")

    print("\n" + "=" * 50)
    print("Disclaimer: This AI prediction is for educational purposes only.")
    print("Always consult healthcare professionals for medical diagnosis.")
    print("=" * 50)


def main(data_path, output_dir=OUTPUT_DIR, interactive=False):
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(data_path)
    target_col = detect_target(df)
    print(f"Detected target column: {target_col}")

    X, y, preprocessor, num_cols, cat_cols = preprocess_dataframe(df, target_col)

    try:
        plot_correlation_heatmap(
            pd.concat([X[num_cols], y.rename("target")], axis=1),
            num_cols,
            outpath=os.path.join(output_dir, "correlation_heatmap.png"),
        )
    except Exception:
        print("Could not plot correlation heatmap")

    stratify_arg = y if len(np.unique(y)) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=stratify_arg,
    )
    print(f"\nTrain shape: {X_train.shape}, Test shape: {X_test.shape}")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
        "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    }

    trained_pipelines = {}
    metrics = {}

    for name, clf in models.items():
        print(f"\n--- Training {name} ---")
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])
        pipe.fit(X_train, y_train)
        trained_pipelines[name] = pipe
        save_prefix = os.path.join(output_dir, name)
        metrics[name] = evaluate_model(
            pipe,
            X_test,
            y_test,
            model_name=name,
            classes=sorted(np.unique(y)),
            save_plots_prefix=save_prefix,
        )

    if "RandomForest" in trained_pipelines:
        rf_pipe = trained_pipelines["RandomForest"]
        try:
            feature_names = num_cols.copy()
            if cat_cols:
                ohe = rf_pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps[
                    "onehot"
                ]
                cat_encoded_cols = list(ohe.get_feature_names_out(cat_cols))
                feature_names.extend(cat_encoded_cols)

            importances = rf_pipe.named_steps["classifier"].feature_importances_
            if len(importances) == len(feature_names):
                feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)
                print("\n--- Top 15 Feature Importances (Random Forest) ---")
                print(feat_imp.head(15))

                plt.figure(figsize=(8, 6))
                feat_imp.head(15).plot(kind="barh")
                plt.gca().invert_yaxis()
                plt.title("Top 15 Feature Importances (Random Forest)")
                plt.savefig(
                    os.path.join(output_dir, "rf_feature_importances.png"),
                    bbox_inches="tight",
                )
                plt.show()
        except Exception as e:
            print("Could not compute feature importances:", e)

    print("\n--- Hyperparameter Tuning (Random Forest) ---")
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", rf)])
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1_weighted", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"Best params: {grid.best_params_}")

    best_rf = grid.best_estimator_
    metrics["RandomForest_Tuned"] = evaluate_model(
        best_rf,
        X_test,
        y_test,
        model_name="RandomForest_Tuned",
        classes=sorted(np.unique(y)),
        save_plots_prefix=os.path.join(output_dir, "RandomForest_Tuned"),
    )

    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    metrics_df = metrics_df[[c for c in cols if c in metrics_df.columns]]
    print("\n--- Metrics Summary ---")
    print(metrics_df)
    metrics_df.to_csv(os.path.join(output_dir, "model_metrics_summary.csv"))

    best_key = "accuracy"
    best_overall = metrics_df[best_key].idxmax()
    print(f"\nBest model by {best_key}: {best_overall}")

    print(f"\nRetraining {best_overall} on entire dataset...")
    if best_overall == "RandomForest_Tuned":
        final_pipeline = best_rf
    else:
        final_pipeline = trained_pipelines.get(best_overall)

    final_pipeline.fit(X, y)
    final_model_path = os.path.join(output_dir, "best_thyroid_model_final.pkl")
    joblib.dump(final_pipeline, final_model_path)
    print(f"Saved final model to {final_model_path}")

    mapping_path = os.path.join(output_dir, "target_mapping.json")
    pd.Series(mapping).to_json(mapping_path)
    print(f"Saved target mapping to {mapping_path}")

    if interactive:
        mapping_inv = {v: k for k, v in mapping.items()}
        interactive_prediction(final_pipeline, list(X.columns), mapping_inv)

    return final_pipeline, mapping, preprocessor, list(X.columns)


def parse_args():
    parser = argparse.ArgumentParser(description="Train thyroid disorder detection models.")
    parser.add_argument(
        "--data",
        default="data/Thyroid-Dataset.csv",
        help="Path to the thyroid CSV dataset.",
    )
    parser.add_argument(
        "--output",
        default=OUTPUT_DIR,
        help="Directory for saved models and plots.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive screening after training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    final_model, target_mapping, preprocessor, _ = main(
        data_path=args.data,
        output_dir=args.output,
        interactive=args.interactive,
    )

    print("\n" + "=" * 50)
    print("Model training complete!")
    print(f"Final model saved in '{args.output}/best_thyroid_model_final.pkl'")
    print("=" * 50)
