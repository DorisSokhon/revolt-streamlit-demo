import io
import time
import numpy as np
import pandas as pd

# ---- safe matplotlib import (optional) ----
try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_OK = True
except Exception:
    MATPLOTLIB_OK = False

from datetime import datetime
from sklearn.ensemble import IsolationForest
import streamlit as st

st.set_page_config(page_title="Re-Volt: Room Consumption Demo", layout="wide")

# ------------------ Helpers ------------------
DAY_START, DAY_END = 7, 22

def is_day(ts: pd.Timestamp) -> bool:
    return DAY_START <= ts.hour <= DAY_END

FEATURES = [
    "watts","voltage","current","pf","S_va","Q_var",
    "roll_watts_mean","roll_watts_std","delta_watts",
    "tod_sin","tod_cos","dow_sin","dow_cos"
]

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["sensor_id","timestamp"]).copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["minute"] = df["timestamp"].dt.hour*60 + df["timestamp"].dt.minute
    df["dow"] = df["timestamp"].dt.weekday

    # cyclic time
    df["tod_sin"] = np.sin(2*np.pi*df["minute"]/1440)
    df["tod_cos"] = np.cos(2*np.pi*df["minute"]/1440)
    df["dow_sin"] = np.sin(2*np.pi*df["dow"]/7)
    df["dow_cos"] = np.cos(2*np.pi*df["dow"]/7)

    # power terms
    df["S_va"] = df["voltage"] * df["current"]
    df["Q_var"] = df["S_va"] * np.sqrt(np.clip(1 - df["pf"]**2, 0, 1))

    # rolling stats per sensor (12 samples ≈ 1 hour if 5-min sampling)
    df["roll_watts_mean"] = df.groupby("sensor_id")["watts"].transform(
        lambda s: s.rolling(12, min_periods=1).mean()
    )
    df["roll_watts_std"] = df.groupby("sensor_id")["watts"].transform(
        lambda s: s.rolling(12, min_periods=1).std().fillna(0)
    )
    df["delta_watts"] = df.groupby("sensor_id")["watts"].diff().fillna(0)

    # z-score (optional diagnostic)
    df["z"] = (df["watts"] - df["roll_watts_mean"]) / (df["roll_watts_std"] + 1e-6)
    return df

def compute_thresholds(feat_df: pd.DataFrame, q=0.95) -> pd.DataFrame:
    rows = []
    g = feat_df.copy()
    g["hour"] = g["timestamp"].dt.hour
    for sid, grp in g.groupby("sensor_id"):
        g_day   = grp[(g["hour"] >= DAY_START) & (g["hour"] <= DAY_END)]
        g_night = grp[(g["hour"] <  DAY_START) | (g["hour"] >  DAY_END)]
        thr_day   = np.quantile(g_day["watts"], q)   if len(g_day)   else np.nan
        thr_night = np.quantile(g_night["watts"], q) if len(g_night) else np.nan
        rows += [
            {"sensor_id": sid, "period": "day", "thr": thr_day},
            {"sensor_id": sid, "period": "night","thr": thr_night},
        ]
    return pd.DataFrame(rows)

def fit_iforest(feat_df: pd.DataFrame, contamination=0.03, random_state=42):
    models = {}
    for sid, grp in feat_df.groupby("sensor_id"):
        X = grp[FEATURES].values
        if len(grp) < 50:  # not enough samples
            models[sid] = None
            continue
        model = IsolationForest(
            n_estimators=200,
            contamination=contamination,
            max_samples="auto",
            random_state=random_state
        )
        model.fit(X)
        models[sid] = model
    return models

def score_iforest(model, xrow):
    if model is None:
        return 0.0
    return model.score_samples(xrow.reshape(1, -1))[0]

def detect_events(feat_df, thr_df, models, z_threshold=3.0, if_threshold=-0.2):
    # map thresholds
    thr_map = {}
    for r in thr_df.itertuples():
        thr_map.setdefault(r.sensor_id, {})[r.period] = r.thr

    events = []
    for _, row in feat_df.iterrows():
        ts = pd.to_datetime(row["timestamp"])
        sid = row["sensor_id"]
        watts = float(row["watts"])

        # day/night learned thresholds
        dkey = "day" if is_day(ts) else "night"
        dn_thr = thr_map.get(sid, {}).get(dkey, np.nan)
        trig_dn = (not np.isnan(dn_thr)) and (watts > dn_thr)

        # z-score
        z = float(row["z"])
        trig_z = z > z_threshold

        # iforest
        m = models.get(sid)
        x = row[FEATURES].values
        if_score = score_iforest(m, x) if m else 0.0
        trig_if = (m is not None) and (if_score < if_threshold)

        reasons = []
        if trig_dn: reasons.append("DN")
        if trig_z:  reasons.append(f"Z({z:.1f})")
        if trig_if: reasons.append(f"IF({if_score:.2f})")

        if reasons:
            events.append({
                "timestamp": ts,
                "room": row.get("room","?"),
                "sensor_id": sid,
                "watts": watts,
                "reason": "+".join(reasons)
            })
    return pd.DataFrame(events)

# ------------------ UI ------------------
st.title("⚡ Re-Volt – Room Consumption Detection (Demo)")
st.write("Upload a CSV with columns: **timestamp, sensor_id, room, appliance, watts, voltage, current, pf**.")

col1, col2 = st.columns([2,1])
with col1:
    uploaded = st.file_uploader("Upload your data CSV", type=["csv"])
with col2:
    z_th = st.slider("Z-score threshold", 2.0, 5.0, 3.0, 0.1)
    if_th = st.slider("IsolationForest threshold (more negative = stricter)", -0.8, 0.0, -0.2, 0.05)
    q_thr = st.slider("Day/Night quantile threshold", 0.80, 0.99, 0.95, 0.01)

def render_outputs(df):
    # Features
    feat_df = add_features(df)

    # Thresholds + model
    thr_df = compute_thresholds(feat_df, q=q_thr)
    models = fit_iforest(feat_df, contamination=0.03)

    # Detect
    with st.spinner("Detecting anomalies..."):
        events = detect_events(feat_df, thr_df, models, z_threshold=z_th, if_threshold=if_th)

    st.write(f"**Detected events:** {len(events)}")
    st.dataframe(events.head(50))

    # Charts (per-day summaries similar to notebook)
    feat_df["date"] = pd.to_datetime(feat_df["timestamp"]).dt.date
    day = feat_df["date"].min()
    df_day = feat_df[feat_df["date"] == day]

    st.subheader("Energy by Room (first day, kWh approx)")
    if not df_day.empty:
        dt_hours = 5/60.0  # assume 5-min samples
        energy_room = (df_day.groupby("room")["watts"].sum() * dt_hours) / 1000.0
        if MATPLOTLIB_OK:
            fig1 = plt.figure(figsize=(7,3))
            energy_room.sort_values(ascending=False).plot(kind="bar")
            plt.ylabel("kWh"); plt.tight_layout()
            st.pyplot(fig1)
        else:
            st.bar_chart(energy_room.sort_values(ascending=False))

    st.subheader("Load by Hour (first day)")
    if not df_day.empty:
        load_hour = df_day.groupby(df_day["timestamp"].dt.hour)["watts"].sum()
        if MATPLOTLIB_OK:
            fig2 = plt.figure(figsize=(7,3))
            load_hour.plot(kind="line", marker="o")
            plt.xlabel("Hour"); plt.ylabel("Watts (approx)"); plt.tight_layout()
            st.pyplot(fig2)
        else:
            st.line_chart(load_hour)

    st.subheader("Alerts by Room (detected)")
    if not events.empty:
        alerts_by_room = events.groupby("room")["sensor_id"].count().sort_values(ascending=False)
        if MATPLOTLIB_OK:
            fig3 = plt.figure(figsize=(7,3))
            alerts_by_room.plot(kind="bar")
            plt.ylabel("Count"); plt.tight_layout()
            st.pyplot(fig3)
        else:
            st.bar_chart(alerts_by_room)

    # Download events
    st.download_button(
        "⬇️ Download detected events (CSV)",
        data=events.to_csv(index=False).encode("utf-8"),
        file_name="revolt_detected_events.csv",
        mime="text/csv",
    )

# ------------------ Main logic ------------------
if uploaded is not None:
    # minimal checks (case-insensitive rename)
    df = pd.read_csv(uploaded)
    required = {"timestamp","sensor_id","room","appliance","watts","voltage","current","pf"}
    rename_map = {}
    for c in df.columns:
        lc = c.lower()
        if lc in required:
            rename_map[c] = lc
    df = df.rename(columns=rename_map)
    if required - set(df.columns):
        missing_cols = sorted(list(required - set(df.columns)))
        st.error(f"CSV missing columns: {missing_cols}")
        st.stop()

    st.success(f"Loaded {len(df):,} rows · {df['sensor_id'].nunique()} sensors · {df['room'].nunique()} rooms")
    render_outputs(df)

else:
    # ---- AUTO-LOAD DEMO MODE ----
    st.info("No file uploaded — demo data is shown automatically. Upload your own CSV to replace it.")
    ts = pd.date_range("2025-01-01", periods=288, freq="5min")
    demo = pd.DataFrame({
        "timestamp": ts,
        "sensor_id": ["S_LIVING_TV_1"]*len(ts),
        "room": ["LIVING"]*len(ts),
        "appliance": ["TV"]*len(ts),
        "watts": np.random.normal(80, 10, len(ts)).clip(0),
        "voltage": np.random.normal(230, 2, len(ts)),
        "current": 0.4,
        "pf": 0.8,
    })
    render_outputs(demo)
