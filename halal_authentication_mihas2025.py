import streamlit as st
import pandas as pd
import numpy as np
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import confusion_matrix
import plotly.express as px
import os
from io import StringIO

# ========= Config =========
RANDOM_STATE = 42
DEMO_RELATIVE = "data/mihas_training.csv"
UNKNOWN_RELATIVE = "data/mihas_unknown.csv"
DEMO_ABSOLUTE = "/mnt/data/All dataset percentage 40 porcine, 40 bovine, 40 fish gelatines - training dataset.csv"
UNKNOWN_ABSOLUTE = "/mnt/data/All dataset percentage unknown gelatines - testing dataset.csv"

st.set_page_config(page_title="Halal Authentication Platform MIHAS2025", layout="wide")
st.title("Halal Authentication Platform MIHAS2025")

# ========= Sidebar =========
with st.sidebar:
    st.header("Settings")
    iqr_k = st.slider("Outlier cutoff multiplier IQR", 1.0, 3.0, 1.5, 0.1)
    n_pls_req = st.slider("PLS DA components", 2, 6, 3, 1)
    show_pls_labels = st.checkbox("Show SampleID on PLS DA points", value=False)
    use_demo = st.checkbox("Use built in MIHAS demo dataset", value=True)

uploaded_file = st.file_uploader("Upload your training dataset CSV", type=["csv"])

# ========= File helpers =========
def _exists(path: str) -> bool:
    try:
        return bool(path) and os.path.exists(path)
    except Exception:
        return False

def _tiny_training_demo() -> pd.DataFrame:
    csv = StringIO(
        "SampleID,Hyp,His,Ser,Arg,Gly,Asp,Glu,Thr,Ala,Pro,Lys,Tyr,Met,Val,Ile,Leu,Phe,Class\n"
        "P1,3776,0,1151,2255,11982,1267,2433,660,3629,4484,1445,180,300,884,388,907,560,Porcine\n"
        "B1,4070,0,1180,2300,12010,1290,2500,670,3650,4550,1500,175,295,900,395,920,565,Bovine\n"
        "F1,3400,8,990,2005,11200,1200,2300,600,3500,4300,1350,150,260,830,360,870,530,Fish\n"
    )
    return pd.read_csv(csv)

def _load_training_demo() -> pd.DataFrame:
    if _exists(DEMO_ABSOLUTE):
        return pd.read_csv(DEMO_ABSOLUTE)
    if _exists(DEMO_RELATIVE):
        return pd.read_csv(DEMO_RELATIVE)
    return _tiny_training_demo()

def _tiny_unknown_demo(feature_cols: list[str]) -> pd.DataFrame:
    rng = np.random.RandomState(123)
    row = {"SampleID": "U1"}
    for c in feature_cols:
        row[c] = float(rng.normal(1000, 300))
    return pd.DataFrame([row])

def _load_unknown_demo(feature_cols: list[str]) -> pd.DataFrame | None:
    if _exists(UNKNOWN_ABSOLUTE):
        return pd.read_csv(UNKNOWN_ABSOLUTE)
    if _exists(UNKNOWN_RELATIVE):
        return pd.read_csv(UNKNOWN_RELATIVE)
    return _tiny_unknown_demo(feature_cols)

# ========= Cached loaders =========
@st.cache_data
def load_data(file, use_demo_flag):
    if file is not None and not use_demo_flag:
        return pd.read_csv(file)
    return _load_training_demo()

@st.cache_data
def load_unknown(file, feature_cols: list[str]):
    if file is not None:
        return pd.read_csv(file)
    return _load_unknown_demo(feature_cols)

# ========= Load training data =========
df_raw = load_data(uploaded_file, use_demo)
required = {"SampleID", "Class"}
if not required.issubset(df_raw.columns):
    st.error(f"Missing required columns {required}. Found {set(df_raw.columns)}")
    st.stop()

feature_cols = [c for c in df_raw.columns if c not in ["SampleID", "Class"]]
X0 = df_raw[feature_cols].copy()
y_series = df_raw["Class"].copy()

# ========= 1. Dataset preview =========
st.subheader("1. Dataset preview and class balance")
c1, c2 = st.columns([2, 1])
with c1:
    st.dataframe(df_raw.head(), use_container_width=True)
with c2:
    st.write("Class counts")
    st.dataframe(y_series.value_counts().rename_axis("Class").to_frame("Count"))

# ========= Helpers =========
def iqr_outlier_mask(X: pd.DataFrame, k: float = 1.5) -> pd.Series:
    Q1 = X.quantile(0.25)
    Q3 = X.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - k * IQR
    upper = Q3 + k * IQR
    within = (X >= lower) & (X <= upper)
    return within.all(axis=1)

def ddof1_standardise(X: pd.DataFrame, mean: pd.Series = None, std: pd.Series = None):
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

def safe_pls_components(requested: int, X_train_np: np.ndarray, n_classes: int) -> int:
    n_samp = X_train_np.shape[0]
    n_feat = X_train_np.shape[1]
    limit = max(1, min(n_samp - 1, n_feat, n_classes))
    return min(requested, limit)

# ========= 2. Processing pipeline =========
st.subheader("2. Processing pipeline")
mask_keep = iqr_outlier_mask(X0, k=iqr_k)
df_filtered = df_raw.loc[mask_keep].reset_index(drop=True)
Xf = df_filtered[feature_cols].copy()
y = df_filtered["Class"].copy()

le = LabelEncoder()
y_enc = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    Xf, y_enc, test_size=0.2, stratify=y_enc, random_state=RANDOM_STATE
)

X_train_z, mean_train, std_train = ddof1_standardise(X_train)
x_min, rng = fit_minmax_params_after_z(X_train_z)
X_all_scaled = transform_full_pipeline(Xf, mean_train, std_train, x_min, rng)

# ========= 3. KMO test =========
st.subheader("3. Kaiser Meyer Olkin KMO test")
kmo_value = kmo_statistic(pd.DataFrame(X_train_z, columns=feature_cols))
adequate = kmo_value >= 0.5
st.metric(label="KMO statistic overall", value=f"{kmo_value:.3f}")
if adequate:
    st.success("Dataset adequacy is acceptable (KMO ≥ 0.5)")
else:
    st.warning("Dataset adequacy is not acceptable (KMO < 0.5)")

# ========= 4. PLS DA =========
st.subheader("4. PLS DA on processed data")
n_pls = safe_pls_components(n_pls_req, X_train, n_classes=len(le.classes_))
Y_train_oh = np.eye(len(le.classes_))[y_train]
pls = PLSRegression(n_components=n_pls)
pls.fit(X_all_scaled, np.eye(len(le.classes_))[y_enc])

pls_scores_all = pls.transform(X_all_scaled)
pls_cols = [f"PLS{i}" for i in range(1, n_pls + 1)]
pls_df = pd.DataFrame(pls_scores_all, columns=pls_cols, index=Xf.index)
pls_df["Class"] = le.inverse_transform(y_enc)
pls_df["SampleID"] = df_filtered["SampleID"].values

fig_pls = px.scatter_3d(
    pls_df, x=pls_cols[0], y=pls_cols[1], z=pls_cols[min(2, n_pls - 1)],
    color="Class", text="SampleID" if show_pls_labels else None,
    title="PLS DA Scores"
)
st.plotly_chart(fig_pls, use_container_width=True)

# ========= 4b. Leave-one-out evaluation =========
st.subheader("4b. PLS DA leave-one-out evaluation")
n_samples = Xf.shape[0]
classes = list(le.classes_)
y_true_all, y_pred_all = [], []

for i in range(n_samples):
    train_mask = np.ones(n_samples, dtype=bool)
    train_mask[i] = False

    X_tr = Xf.iloc[train_mask].copy()
    X_te = Xf.iloc[~train_mask].copy()
    y_tr = y_enc[train_mask]
    y_te = y_enc[~train_mask]

    X_tr_z, mean_tr, std_tr = ddof1_standardise(X_tr)
    x_min_tr, rng_tr = fit_minmax_params_after_z(X_tr_z)
    X_tr_sc = minmax_scale_1_100_from_params(X_tr_z, x_min_tr, rng_tr)

    X_te_z, _, _ = ddof1_standardise(X_te, mean_tr, std_tr)
    X_te_sc = minmax_scale_1_100_from_params(X_te_z, x_min_tr, rng_tr)

    n_pls_fold = safe_pls_components(n_pls, X_tr_sc, n_classes=len(classes))
    Y_tr_oh = np.eye(len(classes))[y_tr]
    pls_fold = PLSRegression(n_components=n_pls_fold)
    pls_fold.fit(X_tr_sc, Y_tr_oh)

    y_hat = pls_fold.predict(X_te_sc).argmax(axis=1)[0]
    y_true_all.append(int(y_te[0]))
    y_pred_all.append(int(y_hat))

cm_loo = confusion_matrix(le.inverse_transform(y_true_all), le.inverse_transform(y_pred_all), labels=classes)
df_cm_loo = pd.DataFrame(cm_loo, index=classes, columns=classes)
df_cm_loo["Correct"] = np.diag(cm_loo)
st.dataframe(df_cm_loo, use_container_width=True)
st.metric("Total correct classifications", f"{int(np.trace(cm_loo))} of {n_samples}")
st.caption(f"Overall accuracy {np.trace(cm_loo)/n_samples:.3%}")

# ========= 5. VIP scores =========
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

# ========= 6. Predict unknown dataset =========
st.subheader("6. Predict unknown dataset")
unknown_file = st.file_uploader("Upload unknown dataset CSV", type=["csv"], key="unknown_uploader")
df_unknown_raw = load_unknown(unknown_file, feature_cols)

if df_unknown_raw is not None and not df_unknown_raw.empty and "SampleID" in df_unknown_raw.columns:
    Xu_raw = df_unknown_raw[feature_cols].copy().apply(pd.to_numeric, errors="coerce").fillna(mean_train)
    std_safe = std_train.replace(0.0, 1.0)
    Xu_z = (Xu_raw - mean_train) / std_safe
    Xu_scaled = 1.0 + 99.0 * (Xu_z - x_min) / rng.replace(0.0, 1.0)

    Y_pred_u = pls.predict(Xu_scaled)
    y_pred_idx = Y_pred_u.argmax(axis=1)
    y_pred_labels = le.inverse_transform(y_pred_idx)

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
    st.dataframe(results, use_container_width=True)
