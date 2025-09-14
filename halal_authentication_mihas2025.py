import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

RANDOM_STATE = 42

st.set_page_config(page_title="Halal Authentication Platform MIHAS2025", layout="wide")
st.title("Halal Authentication Platform MIHAS2025")

# ============== Sidebar controls ==============
with st.sidebar:
    st.header("Processing and modelling settings")
    iqr_k = st.slider("Outlier cut off multiplier IQR", 1.0, 3.0, 1.5, 0.1)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    n_pc = st.slider("PCA components", 2, 5, 3, 1)
    n_pls = st.slider("PLS DA components", 2, 5, 3, 1)
    show_labels = st.checkbox("Show SampleID on PCA points")
    show_pls_labels = st.checkbox("Show SampleID on PLS DA points")

uploaded_file = st.file_uploader("Upload your FTIR dataset CSV only", type=["csv"])

@st.cache_data
def load_data(file):
    if file is not None:
        return pd.read_csv(file)
    # simple demo fallback
    demo = pd.DataFrame({
        "SampleID":[f"S{i:03d}" for i in range(1,61)],
        "Class": np.repeat(["Halal","Non-Halal","Borderline"], 20)
    })
    rng = np.random.RandomState(0)
    X = rng.normal(size=(60, 200))
    X[demo["Class"].eq("Non-Halal")] += 0.5
    X[demo["Class"].eq("Borderline")] += rng.normal(0.2, 0.1, size=(20,200))
    demo = pd.concat([demo, pd.DataFrame(X, columns=[f"WN_{i}" for i in range(200)])], axis=1)
    return demo

df_raw = load_data(uploaded_file)

required = {"SampleID","Class"}
if not required.issubset(df_raw.columns):
    st.error(f"Missing required columns {required}. Found {set(df_raw.columns)}")
    st.stop()

feature_cols = [c for c in df_raw.columns if c not in ["SampleID","Class"]]
X0 = df_raw[feature_cols].copy()
y_series = df_raw["Class"].copy()

st.subheader("1. Dataset preview and class balance")
col1, col2 = st.columns([2,1])
with col1:
    st.dataframe(df_raw.head(), use_container_width=True)
with col2:
    st.write("Class counts")
    st.dataframe(y_series.value_counts().rename_axis("Class").to_frame("Count"))

# ============== Helper functions ==============
def iqr_outlier_mask(X: pd.DataFrame, k: float = 1.5) -> pd.Series:
    """True means keep the row no feature outside IQR fence."""
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    within = (X >= lower) & (X <= upper)
    return within.all(axis=1)

def fit_ddof1_minmax_params(X_train_z: pd.DataFrame):
    """Compute min and max after ddof1 standardisation for scaling to 1 100."""
    x_min = X_train_z.min(axis=0)
    x_max = X_train_z.max(axis=0)
    # Protect against zero range
    rng = (x_max - x_min).replace(0.0, 1.0)
    return x_min, x_max, rng

def ddof1_standardise(X: pd.DataFrame, mean: pd.Series = None, std: pd.Series = None):
    """Z score using sample std ddof 1. If mean and std provided use them else fit from X."""
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0, ddof=1).replace(0.0, 1.0)
    Z = (X - mean) / std
    return Z, mean, std

def minmax_scale_1_100(X_z: pd.DataFrame, x_min: pd.Series, rng: pd.Series):
    """Scale each column to [1, 100] using precomputed min and range on TRAIN set."""
    return 1.0 + 99.0 * (X_z - x_min) / rng

def kmo_statistic(X_for_kmo: pd.DataFrame) -> float:
    """
    Compute Kaiser Meyer Olkin overall statistic.
    KMO = sum_{i!=j} r_ij^2 / [ sum_{i!=j} r_ij^2 + sum_{i!=j} p_ij^2 ]
    where p_ij are partial correlations from the inverse correlation matrix.
    """
    # Correlation matrix
    R = np.corrcoef(X_for_kmo.values, rowvar=False)
    # Numerical safety
    R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
    # Inverse correlation and partial correlations
    try:
        invR = inv(R)
    except np.linalg.LinAlgError:
        # Add small ridge if singular
        eps = 1e-6
        invR = inv(R + eps * np.eye(R.shape[0]))
    D = np.sqrt(np.outer(np.diag(invR), np.diag(invR)))
    P = -invR / D
    np.fill_diagonal(P, 0.0)
    np.fill_diagonal(R, 0.0)
    r2_sum = np.sum(R**2)
    p2_sum = np.sum(P**2)
    if r2_sum + p2_sum == 0:
        return 0.0
    return float(r2_sum / (r2_sum + p2_sum))

# ============== 2. Processing pipeline ==============
st.subheader("2. Processing pipeline")
# 2.1 Outlier removal using IQR across all features
mask_keep = iqr_outlier_mask(X0, k=iqr_k)
removed = (~mask_keep).sum()
st.write(f"Outlier removal using IQR k equals {iqr_k}. Rows removed {int(removed)} of {len(X0)}")
df_filtered = df_raw.loc[mask_keep].reset_index(drop=True)
Xf = df_filtered[feature_cols].copy()
y = df_filtered["Class"].copy()

# 2.2 Train test split after outlier filtering
le = LabelEncoder()
y_enc_all = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    Xf, y_enc_all, test_size=test_size, stratify=y_enc_all, random_state=RANDOM_STATE
)

# 2.3 Standardise with ddof 1 fit on TRAIN only
X_train_z, mean_train, std_train = ddof1_standardise(X_train)
X_test_z, _, _ = ddof1_standardise(X_test, mean_train, std_train)

# 2.4 Scale to 1 100 using TRAIN parameters computed AFTER standardisation
x_min, x_max, rng = fit_ddof1_minmax_params(X_train_z)
X_train_scaled = minmax_scale_1_100(X_train_z, x_min, rng)
X_test_scaled = minmax_scale_1_100(X_test_z, x_min, rng)

# For plots tables join back identifiers
df_processed = pd.concat([
    df_filtered.loc[X_train.index, ["SampleID","Class"]].reset_index(drop=True),
    pd.DataFrame(X_train_scaled, columns=feature_cols).reset_index(drop=True)
], axis=1)
df_processed_test = pd.concat([
    df_filtered.loc[X_test.index, ["SampleID","Class"]].reset_index(drop=True),
    pd.DataFrame(X_test_scaled, columns=feature_cols).reset_index(drop=True)
], axis=1)

st.caption("Preview of processed training data after outlier removal, ddof 1 standardisation and 1 to 100 scaling")
st.dataframe(df_processed.head(), use_container_width=True)

# ============== 3. KMO test on standardised TRAIN data ==============
st.subheader("3. Kaiser Meyer Olkin KMO test")
# KMO is usually computed on z scored variables. We use TRAIN only to avoid leakage.
kmo_value = kmo_statistic(pd.DataFrame(X_train_z, columns=feature_cols))
adequate = kmo_value >= 0.5
st.metric(label="KMO statistic overall", value=f"{kmo_value:.3f}")
if adequate:
    st.success("Dataset adequacy for halal authentication purpose is acceptable KMO equal or more than 0 point 5")
else:
    st.warning("Dataset adequacy for halal authentication purpose is NOT acceptable KMO less than 0 point 5. Consider collecting more samples removing noisy variables or improving measurement quality")

# ============== 4. PCA using processed TRAIN fit ==============
st.subheader("4. PCA visualisation")
pca = PCA(n_components=n_pc, random_state=RANDOM_STATE)
pca.fit(X_train_scaled)
# transform all kept rows to plot everyone
X_all_scaled = pd.concat([
    pd.DataFrame(minmax_scale_1_100(*ddof1_standardise(Xf, mean_train, std_train)[:1], x_min, rng), columns=feature_cols,
                 index=Xf.index)
], axis=1)
scores_all = pca.transform(X_all_scaled.values)
pca_cols = [f"PC{i}" for i in range(1, n_pc+1)]
pca_df = pd.DataFrame(scores_all, columns=pca_cols, index=Xf.index)
pca_df["Class"] = le.inverse_transform(y_enc_all)
pca_df["SampleID"] = df_filtered.loc[Xf.index, "SampleID"].values

fig_pca = px.scatter_3d(
    pca_df, x=pca_cols[0], y=pca_cols[1], z=pca_cols[min(2, n_pc-1)],
    color="Class", text="SampleID" if show_labels else None, title="PCA Score Plot"
)
st.plotly_chart(fig_pca, use_container_width=True)

# Loadings
loadings = pd.DataFrame(pca.components_.T, index=feature_cols, columns=pca_cols)
fig_load = px.scatter_3d(
    loadings.reset_index(), x=pca_cols[0], y=pca_cols[1], z=pca_cols[min(2, n_pc-1)],
    text="index", title="PCA Loadings"
)
st.plotly_chart(fig_load, use_container_width=True)

# Biplot
fig_bi = go.Figure()
for label in pca_df["Class"].unique():
    sub = pca_df[pca_df["Class"] == label]
    fig_bi.add_trace(go.Scatter3d(
        x=sub[pca_cols[0]], y=sub[pca_cols[1]], z=sub[pca_cols[min(2, n_pc-1)]],
        mode="markers+text" if show_labels else "markers",
        text=sub["SampleID"] if show_labels else None,
        name=label
    ))
scale = 3.0
for i, row in loadings.iterrows():
    fig_bi.add_trace(go.Scatter3d(
        x=[0, row[pca_cols[0]]*scale], y=[0, row[pca_cols[1]]*scale], z=[0, row[pca_cols[min(2, n_pc-1)]]*scale],
        mode="lines", showlegend=False
    ))
fig_bi.update_layout(title="PCA Biplot")
st.plotly_chart(fig_bi, use_container_width=True)

# ============== 5. PLS DA and VIP on processed data ==============
st.subheader("5. PLS DA on processed data")
# Build matrices from processed splits
Y_train_oh = np.eye(len(le.classes_))[y_train]
pls = PLSRegression(n_components=n_pls)
pls.fit(X_train_scaled, Y_train_oh)

# Scores for everyone
T_all = pls.transform(X_all_scaled.values)  # x_scores_
pls_cols = [f"PLS{i}" for i in range(1, n_pls+1)]
pls_df = pd.DataFrame(T_all, columns=pls_cols, index=Xf.index)
pls_df["Class"] = le.inverse_transform(y_enc_all)
pls_df["SampleID"] = df_filtered.loc[Xf.index, "SampleID"].values

fig_pls = px.scatter_3d(
    pls_df, x=pls_cols[0], y=pls_cols[1], z=pls_cols[min(2,n_pls-1)],
    color="Class", text="SampleID" if show_pls_labels else None, title="PLS DA Scores"
)
st.plotly_chart(fig_pls, use_container_width=True)

# Test predictions
Y_pred_test = pls.predict(X_test_scaled)
y_pred_test = Y_pred_test.argmax(axis=1)
st.markdown("**PLS DA test set report**")
rep_pls = classification_report(le.inverse_transform(y_test),
                                le.inverse_transform(y_pred_test),
                                output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(rep_pls).transpose().round(3), use_container_width=True)
cm_pls = confusion_matrix(le.inverse_transform(y_test),
                          le.inverse_transform(y_pred_test),
                          labels=le.classes_)
st.markdown("**PLS DA test set confusion matrix**")
st.dataframe(pd.DataFrame(cm_pls, index=le.classes_, columns=le.classes_))

# VIP scores
# Extract internals
T = pls.x_scores_
W = pls.x_weights_
Q = pls.y_loadings_
p, h = W.shape
SStotal = np.sum(T**2, axis=0) * np.sum(Q**2, axis=0)
vip = np.sqrt(p * np.sum((W**2) * SStotal.reshape(1, -1), axis=1) / np.sum(SStotal))
vip_df = pd.DataFrame({"Variable": feature_cols, "VIP_Score": vip}).sort_values("VIP_Score", ascending=False)
st.subheader("6. VIP scores")
st.dataframe(vip_df, use_container_width=True)
fig_vip = px.bar(vip_df.head(20), x="Variable", y="VIP_Score", title="Top 20 VIP")
st.plotly_chart(fig_vip, use_container_width=True)
st.download_button("Download VIP CSV", vip_df.to_csv(index=False).encode(), "vip_scores.csv", "text/csv")

# ============== 7. Logistic regression baseline on processed data ==============
st.subheader("7. Logistic regression baseline")
logit = LogisticRegression(max_iter=300, random_state=RANDOM_STATE)
logit.fit(X_train_scaled, y_train)
y_pred_lr = logit.predict(X_test_scaled)
rep_lr = classification_report(y_test, y_pred_lr, target_names=le.classes_, output_dict=True, zero_division=0)
st.markdown("**Logistic regression test set report**")
st.dataframe(pd.DataFrame(rep_lr).transpose().round(3), use_container_width=True)
cm_lr = confusion_matrix(le.inverse_transform(y_test),
                         le.inverse_transform(y_pred_lr),
                         labels=le.classes_)
st.markdown("**Logistic regression test set confusion matrix**")
st.dataframe(pd.DataFrame(cm_lr, index=le.classes_, columns=le.classes_))

# Quick CV snapshot on entire processed set using same pipeline
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
def fit_score(seed_split):
    return cross_val_score(
        LogisticRegression(max_iter=300, random_state=RANDOM_STATE),
        np.vstack([X_train_scaled, X_test_scaled]),
        np.hstack([y_train, y_test]),
        cv=cv, scoring="accuracy"
    )
cv_scores = fit_score(RANDOM_STATE)
st.caption(f"Five fold CV accuracy mean {cv_scores.mean():.3f}  std {cv_scores.std():.3f}")
