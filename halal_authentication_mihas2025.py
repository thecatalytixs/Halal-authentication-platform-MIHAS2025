import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import classification_report, confusion_matrix
import plotly.express as px

# ========== Config ==========
RANDOM_STATE = 42
DEMO_PATH = "/mnt/data/All dataset percentage 40 porcine, 40 bovine, 40 fish gelatines - training dataset.csv"
UNKNOWN_DEMO_PATH = "/mnt/data/All dataset percentage unknown gelatines - testing dataset.csv"

st.set_page_config(page_title="Halal Authentication Platform MIHAS2025", layout="wide")
st.title("Halal Authentication Platform MIHAS2025")

# ========== Sidebar ==========
with st.sidebar:
    st.header("Settings")
    iqr_k = st.slider("Outlier cut off multiplier IQR", 1.0, 3.0, 1.5, 0.1)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_pls = st.slider("PLS DA components", 2, 5, 3, 1)
    show_pls_labels = st.checkbox("Show SampleID on PLS DA points", value=False)
    use_demo = st.checkbox("Use built in MIHAS demo dataset", value=True)

uploaded_file = st.file_uploader("Upload your dataset CSV", type=["csv"])

# ========== Load data ==========
@st.cache_data
def load_data(file, use_demo_flag):
    if file is not None and not use_demo_flag:
        return pd.read_csv(file)
    return pd.read_csv(DEMO_PATH)

df_raw = load_data(uploaded_file, use_demo)

# Basic checks
required = {"SampleID", "Class"}
if not required.issubset(df_raw.columns):
    st.error(f"Missing required columns {required}. Found {set(df_raw.columns)}")
    st.stop()

feature_cols = [c for c in df_raw.columns if c not in ["SampleID", "Class"]]
X0 = df_raw[feature_cols].copy()
y_series = df_raw["Class"].copy()

# ========== 1. Dataset preview ==========
st.subheader("1. Dataset preview and class balance")
c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(df_raw.head(), use_container_width=True)
with c2:
    st.write("Class counts")
    st.dataframe(y_series.value_counts().rename_axis("Class").to_frame("Count"))

# ========== Helpers ==========
def iqr_outlier_mask(X: pd.DataFrame, k: float = 1.5) -> pd.Series:
    """True means keep row. Row is kept only if all features lie within IQR fences."""
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    within = (X >= lower) & (X <= upper)
    return within.all(axis=1)

def ddof1_standardise(X: pd.DataFrame, mean: pd.Series = None, std: pd.Series = None):
    """Z score using sample std ddof 1."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0, ddof=1).replace(0.0, 1.0)
    Z = (X - mean) / std
    return Z, mean, std

def fit_minmax_params_after_z(X_z: pd.DataFrame):
    x_min = X_z.min(axis=0)
    x_max = X_z.max(axis=0)
    rng = (x_max - x_min).replace(0.0, 1.0)
    return x_min, rng

def minmax_scale_1_100_from_params(X_z: pd.DataFrame, x_min: pd.Series, rng: pd.Series):
    return 1.0 + 99.0 * (X_z - x_min) / rng

def transform_full_pipeline(X: pd.DataFrame, mean: pd.Series, std: pd.Series, x_min: pd.Series, rng: pd.Series):
    Xz, _, _ = ddof1_standardise(X, mean, std)
    return minmax_scale_1_100_from_params(Xz, x_min, rng)

def kmo_statistic(X_for_kmo: pd.DataFrame) -> float:
    """
    Kaiser Meyer Olkin overall statistic.
    KMO = sum r_ij^2 / (sum r_ij^2 + sum p_ij^2)
    where p_ij are partial correlations from the inverse correlation matrix.
    """
    R = np.corrcoef(X_for_kmo.values, rowvar=False)
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        invR = inv(R)
    except np.linalg.LinAlgError:
        eps = 1e-6
        invR = inv(R + eps * np.eye(R.shape[0]))
    D = np.sqrt(np.outer(np.diag(invR), np.diag(invR)))
    P = -invR / D
    np.fill_diagonal(P, 0.0)
    np.fill_diagonal(R, 0.0)
    r2_sum = float(np.sum(R**2))
    p2_sum = float(np.sum(P**2))
    denom = r2_sum + p2_sum
    return 0.0 if denom == 0.0 else r2_sum / denom

# ========== 2. Processing pipeline ==========
st.subheader("2. Processing pipeline")
# 2.1 Outlier removal
mask_keep = iqr_outlier_mask(X0, k=iqr_k)
removed = int((~mask_keep).sum())
st.write(f"Outlier removal using IQR k equals {iqr_k}. Rows removed {removed} of {len(X0)}")
df_filtered = df_raw.loc[mask_keep].reset_index(drop=True)
Xf = df_filtered[feature_cols].copy()
y = df_filtered["Class"].copy()

# 2.2 Split
le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    Xf, y_enc, test_size=test_size, stratify=y_enc, random_state=RANDOM_STATE
)

# 2.3 Standardise with ddof 1 fit on TRAIN only
X_train_z, mean_train, std_train = ddof1_standardise(X_train)
X_test_z, _, _ = ddof1_standardise(X_test, mean_train, std_train)

# 2.4 Scale to 1–100 using TRAIN parameters after standardisation
x_min, rng = fit_minmax_params_after_z(X_train_z)
X_train_scaled = minmax_scale_1_100_from_params(X_train_z, x_min, rng)
X_test_scaled = minmax_scale_1_100_from_params(X_test_z, x_min, rng)

# For later visualisation of all filtered rows
X_all_scaled = transform_full_pipeline(Xf, mean_train, std_train, x_min, rng)

st.caption("Preview of processed training data after outlier removal, ddof 1 standardisation and 1 to 100 scaling")
df_processed_preview = pd.concat(
    [
        df_filtered.loc[X_train.index, ["SampleID", "Class"]].reset_index(drop=True),
        pd.DataFrame(X_train_scaled, columns=feature_cols).reset_index(drop=True),
    ],
    axis=1,
)
st.dataframe(df_processed_preview.head(), use_container_width=True)

# ========== 3. KMO test ==========
st.subheader("3. Kaiser Meyer Olkin KMO test")
kmo_value = kmo_statistic(pd.DataFrame(X_train_z, columns=feature_cols))
adequate = kmo_value >= 0.5
st.metric(label="KMO statistic overall", value=f"{kmo_value:.3f}")
if adequate:
    st.success("Dataset adequacy for halal authentication purpose is acceptable KMO equal or more than 0.5")
else:
    st.warning("Dataset adequacy for halal authentication purpose is NOT acceptable KMO less than 0.5. Consider collecting more samples removing noisy variables or improving measurement quality")

# ========== 4. PLS DA ==========
st.subheader("4. PLS DA on processed data")
Y_train_oh = np.eye(len(le.classes_))[y_train]
pls = PLSRegression(n_components=n_pls)
pls.fit(X_train_scaled, Y_train_oh)

# Scores for all filtered rows
pls_scores_all = pls.transform(X_all_scaled)  # x_scores_
pls_cols = [f"PLS{i}" for i in range(1, n_pls + 1)]
pls_df = pd.DataFrame(pls_scores_all, columns=pls_cols, index=Xf.index)
pls_df["Class"] = le.inverse_transform(y_enc)
pls_df["SampleID"] = df_filtered.loc[Xf.index, "SampleID"].values

fig_pls = px.scatter_3d(
    pls_df,
    x=pls_cols[0],
    y=pls_cols[1],
    z=pls_cols[min(2, n_pls - 1)],
    color="Class",
    text="SampleID" if show_pls_labels else None,
    title="PLS DA Scores",
)
st.plotly_chart(fig_pls, use_container_width=True)

# Test predictions and metrics
Y_pred_test = pls.predict(X_test_scaled)           # n_samples x n_classes
y_pred_test = Y_pred_test.argmax(axis=1)
report_pls = classification_report(
    le.inverse_transform(y_test),
    le.inverse_transform(y_pred_test),
    output_dict=True,
    zero_division=0,
)
st.markdown("**PLS DA test set report**")
st.dataframe(pd.DataFrame(report_pls).transpose().round(3), use_container_width=True)

cm_pls = confusion_matrix(
    le.inverse_transform(y_test),
    le.inverse_transform(y_pred_test),
    labels=le.classes_,
)
st.markdown("**PLS DA test set confusion matrix**")
st.dataframe(pd.DataFrame(cm_pls, index=le.classes_, columns=le.classes_))

# ========== 5. VIP scores ==========
st.subheader("5. VIP scores")
T = pls.x_scores_
W = pls.x_weights_
Q = pls.y_loadings_
p, h = W.shape
SStotal = np.sum(T**2, axis=0) * np.sum(Q**2, axis=0)
vip = np.sqrt(p * np.sum((W**2) * SStotal.reshape(1, -1), axis=1) / np.sum(SStotal))
vip_df = pd.DataFrame({"Variable": feature_cols, "VIP_Score": vip}).sort_values("VIP_Score", ascending=False)

st.dataframe(vip_df, use_container_width=True)
fig_vip = px.bar(vip_df.head(20), x="Variable", y="VIP_Score", title="Top 20 VIP")
st.plotly_chart(fig_vip, use_container_width=True)
st.download_button("Download VIP CSV", vip_df.to_csv(index=False).encode(), "vip_scores.csv", "text/csv")

# ========== 6. Predict unknown dataset with trained PLS DA ==========
st.subheader("6. Predict unknown dataset")
st.caption("Upload a CSV with the same feature columns as the training data. Required columns: SampleID plus all amino acid features. The model will reuse training standardisation and scaling parameters.")

unknown_file = st.file_uploader("Upload unknown dataset CSV", type=["csv"], key="unknown_uploader")

@st.cache_data
def load_unknown(file):
    if file is not None:
        return pd.read_csv(file)
    try:
        return pd.read_csv(UNKNOWN_DEMO_PATH)
    except Exception:
        return None

df_unknown_raw = load_unknown(unknown_file)

if df_unknown_raw is None:
    st.info("No unknown dataset provided yet. Upload a CSV to generate predictions.")
else:
    if "SampleID" not in df_unknown_raw.columns:
        st.error("The unknown dataset is missing the SampleID column.")
    else:
        missing_feats = [c for c in feature_cols if c not in df_unknown_raw.columns]
        extra_feats = [c for c in df_unknown_raw.columns if c not in feature_cols + ["SampleID", "Class"]]

        if missing_feats:
            st.error(f"The unknown dataset is missing required feature columns. Missing count {len(missing_feats)}. First few missing {missing_feats[:10]}")
        else:
            Xu_raw = df_unknown_raw[feature_cols].copy()

            # Apply the same preprocessing learned from TRAIN
            std_safe = std_train.replace(0.0, 1.0)
            Xu_z = (Xu_raw - mean_train) / std_safe
            Xu_scaled = 1.0 + 99.0 * (Xu_z - x_min) / rng.replace(0.0, 1.0)

            # Predict using trained PLS DA
            Y_pred_u = pls.predict(Xu_scaled)
            y_pred_idx = Y_pred_u.argmax(axis=1)
            y_pred_labels = le.inverse_transform(y_pred_idx)

            # Softmax style confidence from raw PLS outputs
            def softmax(a, axis=1):
                a = np.asarray(a, dtype=float)
                a = a - np.max(a, axis=axis, keepdims=True)
                e = np.exp(a)
                return e / np.sum(e, axis=axis, keepdims=True)

            probs = softmax(Y_pred_u, axis=1)
            conf = probs.max(axis=1)

            results = pd.DataFrame({
                "SampleID": df_unknown_raw["SampleID"].values,
                "Predicted_Class": y_pred_labels,
                "Confidence": np.round(conf, 4),
            })

            class_names = list(le.classes_)
            score_cols = {f"Score_{cls}": Y_pred_u[:, i] for i, cls in enumerate(class_names)}
            prob_cols = {f"Prob_{cls}": probs[:, i] for i, cls in enumerate(class_names)}
            results = pd.concat([results, pd.DataFrame(score_cols), pd.DataFrame(prob_cols)], axis=1)

            st.markdown("**Prediction results on unknown dataset**")
            st.dataframe(results, use_container_width=True)

            st.markdown("**Predicted class distribution**")
            pred_counts = results["Predicted_Class"].value_counts().rename_axis("Class").to_frame("Count")
            st.dataframe(pred_counts)

            st.download_button(
                "Download predictions as CSV",
                results.to_csv(index=False).encode(),
                "unknown_predictions_plsda.csv",
                "text/csv"
            )

            if extra_feats:
                st.caption(f"Note the unknown dataset contains {len(extra_feats)} extra columns that were ignored. Example {extra_feats[:8]}")
