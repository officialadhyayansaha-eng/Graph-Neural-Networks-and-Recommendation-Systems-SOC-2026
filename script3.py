import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    roc_auc_score
)


# =====================================
# LOAD DATA
# =====================================

df = pd.read_pickle("submission_data.pkl")

print("Dataset Shape:")
print(df.shape)

# =====================================
# CREATE TARGET
# =====================================

df["target"] = (
    df["Submission_Delay"] > 0
).astype(int)

# remove leakage column
df.drop(
    columns=["Submission_Delay"],
    inplace=True
)

# =====================================
# USE ONLY x_* FEATURES
# =====================================

feature_cols = [
    col
    for col in df.columns
    if col.startswith("x_")
]

print("\nNumber of Features:")
print(len(feature_cols))

# =====================================
# SCALE FEATURES
# =====================================

X_scaled = StandardScaler().fit_transform(
    df[feature_cols]
)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=feature_cols,
    index=df.index
)

y = df["target"]

# =====================================
# CORRELATION ANALYSIS
# =====================================

correlations = []

for col in feature_cols:

    corr = abs(
        X_scaled[col].corr(y)
    )

    correlations.append(corr)

corr_series = pd.Series(
    correlations,
    index=feature_cols
)

top15 = (
    corr_series
    .sort_values(
        ascending=False
    )
    .head(15)
)

selected_features = (
    top15.index.tolist()
)

print("\nTop 15 Features:")
print(top15)

# save for report
top15.to_csv(
    "top15_features.csv"
)

# =====================================
# TRAIN TEST SPLIT
# =====================================

X = X_scaled[selected_features]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)

# =====================================
# MODELS
# =====================================

models = {
    "Logistic Regression":
        LogisticRegression(
            max_iter=5000
        ),

    "Support Vector Machine":
        SVC(
            probability=True
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            random_state=42
        )
}

results = []

# =====================================
# TRAIN + EVALUATE
# =====================================

for name, model in models.items():

    print(f"\nTraining {name}")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    report = (
        classification_report(
            y_test,
            predictions,
            output_dict=True
        )
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision (Class 1)": report["1"]["precision"],
        "Recall (Class 1)": report["1"]["recall"],
        "F1-Score (Class 1)": report["1"]["f1-score"],
        "AUC_ROC": auc
    })

# =====================================
# RESULTS TABLE
# =====================================

results_df = pd.DataFrame(
    results
)

print("\nFinal Results:")
print(results_df)


results_df.to_csv(
    "model_results.csv",
    index=False
)

print("\nSaved:")
print("model_results.md")
print("model_results.csv")
print("top15_features.csv")
