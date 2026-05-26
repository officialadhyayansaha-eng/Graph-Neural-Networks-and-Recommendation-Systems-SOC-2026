import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

import umap

import matplotlib.pyplot as plt
import seaborn as sns


# ==================================
# LOAD DATA
# ==================================

df = pd.read_pickle("submission_data.pkl")

print("Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum().sum())


# ==================================
# CREATE TARGET
# ==================================

df["target"] = (df["Submission_Delay"] > 0).astype(int)

df.drop(columns=["Submission_Delay"], inplace=True)


# ==================================
# USE ONLY x_* FEATURES
# ==================================

feature_cols = [
    col for col in df.columns
    if col.startswith("x_")
]

print("\nNumber of Features:", len(feature_cols))

# ==================================
# STANDARDIZATION
# ==================================

scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    df[feature_cols]
)

scaled_df = pd.DataFrame(
    X_scaled,
    columns=feature_cols
)

scaled_df["target"] = df["target"].values


# ==================================
# CORRELATIONS
# ==================================

correlations = []

for col in feature_cols:

    corr = scaled_df[col].corr(
        scaled_df["target"]
    )

    correlations.append(corr)

corr_df = pd.DataFrame({
    "feature": feature_cols,
    "corr": correlations
})

corr_df["abs_corr"] = np.abs(
    corr_df["corr"]
)

top15 = (
    corr_df
    .sort_values(
        "abs_corr",
        ascending=False
    )
    .head(15)
)

top15_features = (
    top15["feature"]
    .tolist()
)

print("\nTop 15 Features:")
print(top15)

top15.to_csv(
    "top15_features.csv",
    index=False
)


# ==================================
# HEATMAP
# ==================================

plt.figure(figsize=(12, 10))

sns.heatmap(
    scaled_df[top15_features].corr(),
    cmap="coolwarm",
    annot=True,
    fmt=".2f"
)

plt.title(
    "Correlation Heatmap of Top 15 Features"
)

plt.tight_layout()

plt.savefig(
    "top15_heatmap.png",
    dpi=300
)

plt.show()


# ==================================
# PREPARE DATA
# ==================================

X_all = scaled_df[feature_cols]

X_top = scaled_df[top15_features]

y = scaled_df["target"]


# ==================================
# TSNE - ALL FEATURES
# ==================================

tsne_all = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne_all = tsne_all.fit_transform(
    X_all
)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_tsne_all[:, 0],
    X_tsne_all[:, 1],
    c=y,
    alpha=0.7
)

plt.colorbar(scatter)

plt.title("t-SNE (All Features)")

plt.savefig(
    "tsne_all_features.png",
    dpi=300
)

plt.show()


# ==================================
# TSNE - TOP 15 FEATURES
# ==================================

tsne_top = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42
)

X_tsne_top = tsne_top.fit_transform(
    X_top
)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_tsne_top[:, 0],
    X_tsne_top[:, 1],
    c=y,
    alpha=0.7
)

plt.colorbar(scatter)

plt.title("t-SNE (Top 15 Features)")

plt.savefig(
    "tsne_top15_features.png",
    dpi=300
)

plt.show()


# ==================================
# UMAP - ALL FEATURES
# ==================================

umap_all = umap.UMAP(
    random_state=42
)

X_umap_all = umap_all.fit_transform(
    X_all
)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_umap_all[:, 0],
    X_umap_all[:, 1],
    c=y,
    alpha=0.7
)

plt.colorbar(scatter)

plt.title("UMAP (All Features)")

plt.savefig(
    "umap_all_features.png",
    dpi=300
)

plt.show()


# ==================================
# UMAP - TOP 15 FEATURES
# ==================================

umap_top = umap.UMAP(
    random_state=42
)

X_umap_top = umap_top.fit_transform(
    X_top
)

plt.figure(figsize=(8, 6))

scatter = plt.scatter(
    X_umap_top[:, 0],
    X_umap_top[:, 1],
    c=y,
    alpha=0.7
)

plt.colorbar(scatter)

plt.title("UMAP (Top 15 Features)")

plt.savefig(
    "umap_top15_features.png",
    dpi=300
)

plt.show()


print("\nDone.")
print("Generated files:")
print("- top15_features.csv")
print("- top15_heatmap.png")
print("- tsne_all_features.png")
print("- tsne_top15_features.png")
print("- umap_all_features.png")
print("- umap_top15_features.png")
