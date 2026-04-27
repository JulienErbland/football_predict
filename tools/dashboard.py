"""
Football Prediction Dashboard — Streamlit app for model interpretation.

Run from repo root:
    backend/.venv/bin/streamlit run tools/dashboard.py

Four tabs:
    1. Model Performance  — metrics on train/test split, calibration curves
    2. Feature Importance — XGBoost/LightGBM importances + SHAP summary
    3. Match Inspector    — select any match, see every feature + SHAP force plot
    4. Upcoming Matches   — full feature breakdown for next predicted matches
"""
from __future__ import annotations

import json
import pickle  # loads only locally-generated trusted model files
import sys
import warnings
from pathlib import Path

# LightGBM fitted with feature names; ensemble passes numpy arrays → harmless warning
warnings.filterwarnings("ignore", message="X does not have valid feature names")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend"))

st.set_page_config(
    page_title="Football Predict — Dashboard",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

META_COLS = {
    "match_id", "league", "season", "date", "matchday",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "referee", "home_goals", "away_goals", "result",
}
RESULT_LABELS = {0: "Home win", 1: "Draw", 2: "Away win"}

FEATURE_DESCRIPTIONS = {
    "home_elo": "Home team Elo rating (higher = stronger historically)",
    "away_elo": "Away team Elo rating",
    "elo_difference": "Elo gap (home − away). Positive = home is stronger",
    "elo_expected_home": "Elo-implied home win probability",
    "home_w3_win_rate": "Home team win rate — last 3 matches",
    "away_w3_win_rate": "Away team win rate — last 3 matches",
    "home_w5_win_rate": "Home team win rate — last 5 matches",
    "away_w5_win_rate": "Away team win rate — last 5 matches",
    "home_w10_win_rate": "Home team win rate — last 10 matches",
    "away_w10_win_rate": "Away team win rate — last 10 matches",
    "home_w5_ppg": "Home team points per game — last 5 matches",
    "away_w5_ppg": "Away team points per game — last 5 matches",
    "home_w5_avg_gf": "Home team avg goals scored — last 5",
    "away_w5_avg_gf": "Away team avg goals scored — last 5",
    "home_w5_avg_ga": "Home team avg goals conceded — last 5",
    "away_w5_avg_ga": "Away team avg goals conceded — last 5",
    "home_w5_avg_gd": "Home team avg goal difference — last 5",
    "away_w5_avg_gd": "Away team avg goal difference — last 5",
    "home_w5_clean_sheet_rate": "Home team clean sheet rate — last 5",
    "away_w5_clean_sheet_rate": "Away team clean sheet rate — last 5",
    "h2h_games": "Head-to-head meetings in last 5 years",
    "h2h_home_win_rate": "H2H win rate for home team (last 5 years)",
    "h2h_draw_rate": "H2H draw rate (last 5 years)",
    "h2h_away_win_rate": "H2H win rate for away team (last 5 years)",
    "h2h_avg_goals": "H2H average total goals per match",
    "home_rest_days": "Days since home team's last match",
    "away_rest_days": "Days since away team's last match",
    "rest_advantage": "Rest advantage (home − away rest days)",
    "home_congestion_30d": "Home team matches in last 30 days",
    "away_congestion_30d": "Away team matches in last 30 days",
    "season_stage": "Season progress (0 = start, 1 = end)",
    "home_league_pos": "Home team current league position",
    "away_league_pos": "Away team current league position",
    "position_gap": "Position gap (home − away). Negative = home is higher",
    "home_relegation_pressure": "Home team in relegation zone (bottom 3)",
    "away_relegation_pressure": "Away team in relegation zone (bottom 3)",
    "home_title_race": "Home team in title race (top 4)",
    "away_title_race": "Away team in title race (top 4)",
    "referee_encoded": "Referee identifier (encoded integer)",
}


# ── data loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_features():
    path = ROOT / "backend" / "data" / "features" / "features.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_resource
def load_models():
    models_dir = ROOT / "backend" / "data" / "models"
    ep = models_dir / "ensemble.pkl"
    cp = models_dir / "feature_cols.pkl"
    if not ep.exists() or not cp.exists():
        return None
    with open(ep, "rb") as f:
        ensemble = pickle.load(f)
    with open(cp, "rb") as f:
        feature_cols = pickle.load(f)
    return ensemble, feature_cols


@st.cache_data
def load_predictions_json():
    for name in ["predictions.json", "predictions_test.json"]:
        path = ROOT / "backend" / "data" / "output" / name
        if path.exists():
            with open(path) as f:
                return json.load(f)
    return None


# ── metrics ───────────────────────────────────────────────────────────────────
# Literature convention (see backend/evaluation/metrics.py). Brier = mean across
# K=3 classes, RPS normalised by K-1. Competitive: Brier 0.18–0.22, RPS 0.19–0.23.

from evaluation.metrics import (  # noqa: E402  (sys.path was just patched above)
    brier_score as brier,
    rps as rps_score,
)


def compute_all_metrics(y_true, y_proba):
    from sklearn.metrics import log_loss
    return {
        "Brier score": round(brier(y_true, y_proba), 4),
        "Log loss": round(log_loss(y_true, y_proba, labels=[0, 1, 2]), 4),
        "RPS": round(rps_score(y_true, y_proba), 4),
        "Accuracy": f"{(y_proba.argmax(axis=1) == y_true).mean():.1%}",
    }


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.title("⚽ Football Predict")
        st.markdown("---")
        features_df = load_features()
        models = load_models()
        preds = load_predictions_json()

        st.markdown("### Data status")
        st.markdown(f"{'✅' if features_df is not None else '❌'} **Features** "
                    f"{'— ' + str(len(features_df)) + ' matches' if features_df is not None else '— not found'}")
        st.markdown(f"{'✅' if models is not None else '❌'} **Models** "
                    f"{'— ensemble loaded' if models is not None else '— run models.train first'}")
        pred_info = f"— {len(preds['matches'])} matches" if preds is not None else "— not found"
        st.markdown(f"{'✅' if preds is not None else '❌'} **Predictions JSON** {pred_info}")

        if features_df is not None:
            st.markdown("---")
            st.markdown("### Dataset")
            seasons = sorted(features_df["season"].unique())
            st.markdown(f"**Seasons:** {', '.join(str(s) for s in seasons)}")
            st.markdown(f"**Leagues:** {features_df['league'].nunique()}")
            feat_cols = [c for c in features_df.columns if c not in META_COLS]
            st.markdown(f"**Features:** {len(feat_cols)}")

        st.markdown("---")
        st.markdown("### Run commands")
        st.code("cd backend\npython -m features.build\npython -m models.train\npython -m output.predict", language="bash")


# ── tab 1: model performance ──────────────────────────────────────────────────

def tab_performance(features_df, models):
    st.header("Model Performance")

    if features_df is None:
        st.warning("No features.parquet found. Run `python -m features.build` first.")
        return
    if models is None:
        st.warning("No trained models found. Run `python -m models.train` first.")
        return

    ensemble, feature_cols = models
    avail = [c for c in feature_cols if c in features_df.columns]

    try:
        from config.loader import model_config
        mc = model_config()
        test_seasons = mc["evaluation"]["test_seasons"]
    except Exception:
        test_seasons = [2024]

    train_df = features_df[~features_df["season"].isin(test_seasons)].dropna(subset=["result"])
    test_df = features_df[features_df["season"].isin(test_seasons)].dropna(subset=["result"])

    X_train = train_df[avail].fillna(0).values.astype(np.float32)
    y_train = train_df["result"].values.astype(int)
    X_test = test_df[avail].fillna(0).values.astype(np.float32) if len(test_df) > 0 else X_train[:30]
    y_test = test_df["result"].values.astype(int) if len(test_df) > 0 else y_train[:30]

    train_proba = ensemble.predict_proba(X_train)
    test_proba = ensemble.predict_proba(X_test)
    train_m = compute_all_metrics(y_train, train_proba)
    test_m = compute_all_metrics(y_test, test_proba)

    st.subheader("Metrics — train vs test")
    cols = st.columns(4)
    metric_help = {
        "Brier score": "Literature convention (mean across K=3 classes). "
                       "Competitive range: 0.18–0.22.",
        "RPS": "Literature convention (normalised by K-1). "
               "Competitive range: 0.19–0.23.",
        "Log loss": "Cross-entropy. Random 3-class baseline ≈ 1.099.",
        "Accuracy": "Argmax accuracy. Random baseline ≈ 33%.",
    }
    for i, metric in enumerate(train_m):
        cols[i].metric(
            metric,
            str(test_m[metric]),
            delta=f"train: {train_m[metric]}",
            help=metric_help.get(metric),
        )
    st.caption(
        "Lower is better for Brier / Log loss / RPS. "
        "Brier and RPS use literature convention (normalised): "
        "Brier ≈ 0.22, RPS ≈ 0.22 on random; ~0.20 on competitive models."
    )

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("Outcome distribution (test set)")
        counts = pd.Series(y_test).map(RESULT_LABELS).value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     color_discrete_sequence=["#2563eb", "#d97706", "#dc2626"])
        fig.update_traces(textinfo="label+percent")
        st.plotly_chart(fig, width='stretch', key="perf_pie")

    with c2:
        st.subheader("Confusion matrix (test set)")
        pred_labels = test_proba.argmax(axis=1)
        confusion = pd.crosstab(
            pd.Series(y_test).map(RESULT_LABELS),
            pd.Series(pred_labels).map(RESULT_LABELS),
            rownames=["Actual"], colnames=["Predicted"],
        )
        fig = px.imshow(confusion, text_auto=True, color_continuous_scale="Blues")
        st.plotly_chart(fig, width='stretch', key="perf_confusion")

    st.markdown("---")
    st.subheader("Calibration curves — are the probabilities trustworthy?")
    st.caption("Points near the diagonal = well-calibrated. Above = underconfident. Below = overconfident.")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(dash="dash", color="gray"), name="Perfect"))
    for cls_idx, (label, color) in enumerate(zip(["Home win", "Draw", "Away win"],
                                                  ["#2563eb", "#d97706", "#dc2626"])):
        probs = test_proba[:, cls_idx]
        actual = (y_test == cls_idx).astype(float)
        bins = np.linspace(0, 1, 11)
        cx, cy, csz = [], [], []
        for i in range(len(bins) - 1):
            mask = (probs >= bins[i]) & (probs < bins[i + 1])
            if mask.sum() < 3:
                continue
            cx.append((bins[i] + bins[i + 1]) / 2)
            cy.append(actual[mask].mean())
            csz.append(mask.sum())
        fig.add_trace(go.Scatter(x=cx, y=cy, mode="lines+markers", name=label,
                                 line=dict(color=color),
                                 marker=dict(size=[max(6, s // 5) for s in csz])))
    fig.update_layout(xaxis_title="Mean predicted probability", yaxis_title="Actual rate",
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]), height=430)
    st.plotly_chart(fig, width='stretch', key="perf_calibration")

    st.markdown("---")
    st.subheader("Predicted probability distribution by actual outcome")
    outcome_filter = st.selectbox("Show probabilities for:", ["Home win", "Draw", "Away win"])
    oi = ["Home win", "Draw", "Away win"].index(outcome_filter)
    fig = go.Figure()
    for cls_idx, label in RESULT_LABELS.items():
        mask = y_test == cls_idx
        fig.add_trace(go.Histogram(x=test_proba[mask, oi], name=f"Actual: {label}",
                                   opacity=0.65, nbinsx=20))
    fig.update_layout(barmode="overlay", xaxis_title=f"Predicted P({outcome_filter})",
                      yaxis_title="Count", height=340)
    st.plotly_chart(fig, width='stretch', key="perf_prob_dist")


# ── tab 2: feature importance ─────────────────────────────────────────────────

def tab_features(features_df, models):
    st.header("Feature Importance & Analysis")

    if features_df is None or models is None:
        st.warning("Features and trained models are required.")
        return

    ensemble, feature_cols = models
    avail = [c for c in feature_cols if c in features_df.columns]

    # model importances
    importances = {}
    for name, model in ensemble.models.items():
        inner = getattr(model, "model", model)
        fi = getattr(inner, "feature_importances_", None)
        fn = getattr(inner, "feature_names_", None)
        if fi is not None and fn is not None and len(fi) == len(fn):
            importances[name] = dict(zip(fn, fi))

    if importances:
        st.subheader("Feature importances — top features by model")
        top_n = st.slider("Show top N features", 10, min(50, len(avail)), 20, key="top_n")
        model_name = st.selectbox("Model", list(importances.keys()))
        imp = importances[model_name]
        df_imp = (pd.DataFrame({"feature": list(imp.keys()), "importance": list(imp.values())})
                  .sort_values("importance", ascending=False).head(top_n))
        df_imp["description"] = df_imp["feature"].map(lambda x: FEATURE_DESCRIPTIONS.get(x, ""))
        fig = px.bar(df_imp.sort_values("importance"), x="importance", y="feature",
                     orientation="h", hover_data=["description"],
                     color="importance", color_continuous_scale="Blues",
                     height=max(400, top_n * 22))
        fig.update_layout(showlegend=False, yaxis_title="", xaxis_title="Importance score")
        st.plotly_chart(fig, width='stretch', key="feat_importances")
    else:
        st.info("Feature importances not available (models may not have been trained with feature names).")

    # SHAP
    st.markdown("---")
    st.subheader("SHAP — global feature impact on home win probability")
    st.caption("SHAP values explain how much each feature pushed each prediction up or down. "
               "This uses a random sample of 500 matches.")

    if st.button("Compute SHAP values (10–30 seconds)"):
        with st.spinner("Computing SHAP values..."):
            try:
                import shap
                df_sample = features_df.dropna(subset=["result"]).sample(
                    min(500, len(features_df)), random_state=42)
                X_sample = df_sample[avail].fillna(0).values.astype(np.float32)

                inner = None
                for _, model in ensemble.models.items():
                    # Drill through CalibratedModel → XGBoostModel → XGBClassifier
                    unwrapped = getattr(model, "model", model)        # CalibratedModel → XGBoostModel
                    candidate = getattr(unwrapped, "_model", unwrapped)  # XGBoostModel → XGBClassifier
                    if hasattr(candidate, "get_booster"):
                        inner = candidate.get_booster()
                        break
                    elif hasattr(candidate, "booster_"):
                        inner = candidate
                        break

                if inner is None:
                    st.warning("No tree model found inside ensemble for SHAP.")
                else:
                    explainer = shap.TreeExplainer(inner)
                    shap_vals = explainer.shap_values(X_sample)
                    if isinstance(shap_vals, list):
                        sv_home = np.array(shap_vals[0])
                    elif hasattr(shap_vals, "ndim") and shap_vals.ndim == 3:
                        sv_home = shap_vals[:, :, 0]
                    else:
                        sv_home = shap_vals

                    mean_abs = (pd.Series(np.abs(sv_home).mean(axis=0), index=avail)
                                .sort_values(ascending=False).head(20))
                    fig = px.bar(x=mean_abs.values, y=mean_abs.index, orientation="h",
                                 labels={"x": "Mean |SHAP| — Home Win", "y": ""},
                                 color=mean_abs.values, color_continuous_scale="Reds", height=500)
                    fig.update_layout(showlegend=False)
                    st.plotly_chart(fig, width='stretch', key="feat_shap_global")

                    st.session_state["shap_values"] = sv_home
                    st.session_state["shap_features"] = avail
                    st.session_state["shap_X"] = X_sample
                    st.session_state["shap_df"] = df_sample.reset_index(drop=True)
                    st.success("Done. Go to Match Inspector to see per-match SHAP contributions.")
            except Exception as e:
                st.error(f"SHAP failed: {e}")

    # feature distribution
    st.markdown("---")
    st.subheader("Feature distribution by match outcome")
    st.caption("How does this feature look for actual home wins vs draws vs away wins?")

    feat_choices = [c for c in avail if not c.startswith("referee")]
    selected = st.selectbox("Feature",
                            feat_choices,
                            format_func=lambda x: f"{x}  —  {FEATURE_DESCRIPTIONS.get(x, '')}",
                            key="feat_violin")
    df_plot = features_df.dropna(subset=["result", selected]).copy()
    df_plot["Outcome"] = df_plot["result"].map(RESULT_LABELS)
    fig = px.violin(df_plot, x="Outcome", y=selected, color="Outcome", box=True, points=False,
                    color_discrete_map={"Home win": "#2563eb", "Draw": "#d97706", "Away win": "#dc2626"},
                    labels={"Outcome": "", selected: FEATURE_DESCRIPTIONS.get(selected, selected)},
                    height=420)
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, width='stretch', key="feat_violin_chart")

    # home vs away scatter
    st.markdown("---")
    st.subheader("Home vs Away — compare two features across all matches")
    c1, c2 = st.columns(2)
    with c1:
        fx = st.selectbox("X axis", feat_choices, index=0,
                          format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x), key="scatter_x")
    with c2:
        fy = st.selectbox("Y axis", feat_choices, index=min(1, len(feat_choices)-1),
                          format_func=lambda x: FEATURE_DESCRIPTIONS.get(x, x), key="scatter_y")
    df_sc = features_df.dropna(subset=["result", fx, fy]).copy()
    df_sc["Outcome"] = df_sc["result"].map(RESULT_LABELS)
    fig = px.scatter(df_sc, x=fx, y=fy, color="Outcome",
                     color_discrete_map={"Home win": "#2563eb", "Draw": "#d97706", "Away win": "#dc2626"},
                     opacity=0.5, height=420,
                     labels={fx: FEATURE_DESCRIPTIONS.get(fx, fx), fy: FEATURE_DESCRIPTIONS.get(fy, fy)})
    st.plotly_chart(fig, width='stretch', key="feat_scatter")


# ── tab 3: match inspector ────────────────────────────────────────────────────

@st.cache_data
def _batch_predict(_ensemble, _feature_cols, _features_df):
    """Run predict_proba on the whole dataset and return a summary DataFrame."""
    avail = [c for c in _feature_cols if c in _features_df.columns]
    scored = _features_df.dropna(subset=["result"]).copy()
    X = scored[avail].fillna(0).values.astype(np.float32)
    probas = _ensemble.predict_proba(X)
    scored = scored.reset_index(drop=True)
    scored["prob_home"] = probas[:, 0]
    scored["prob_draw"] = probas[:, 1]
    scored["prob_away"] = probas[:, 2]
    scored["predicted"] = probas.argmax(axis=1)
    scored["correct"] = scored["predicted"] == scored["result"].astype(int)
    scored["max_prob"] = probas.max(axis=1)
    return scored


def _style_overview(row):
    """Green for correct, red for wrong predictions."""
    # column name is "Correct" after renaming
    correct_val = row.get("Correct", row.get("correct", False))
    color = "#dcfce7" if correct_val else "#fee2e2"
    return [f"background-color: {color}"] * len(row)


def _deep_dive(row, ensemble, avail, selected_id):
    """Render the deep feature analysis for one match row."""
    home, away = row["home_team"], row["away_team"]
    result_val = row.get("result")
    result_str = RESULT_LABELS.get(int(result_val), "?") if pd.notna(result_val) else "Upcoming"
    predicted_str = RESULT_LABELS.get(int(row["predicted"]), "?")
    correct = bool(row.get("correct", False))

    col_h, col_vs, col_a = st.columns([3, 1, 3])
    col_h.metric("Home", home)
    col_vs.markdown("<div style='text-align:center;padding-top:28px;font-size:22px'>vs</div>",
                    unsafe_allow_html=True)
    col_a.metric("Away", away)

    ic = st.columns(5)
    ic[0].metric("Date", str(row["date"])[:10])
    ic[1].metric("League", row["league"])
    ic[2].metric("Matchday", str(row.get("matchday", "—")))
    ic[3].metric("Actual result", result_str)
    ic[4].metric("Prediction correct", "✅ Yes" if correct else "❌ No")

    st.markdown("---")
    st.subheader("Predicted probabilities vs actual outcome")

    ph, pd_, pa = float(row["prob_home"]), float(row["prob_draw"]), float(row["prob_away"])
    predicted_idx = int(row["predicted"])
    actual_idx = int(result_val) if pd.notna(result_val) else None

    pc = st.columns(3)
    for i, (label, prob) in enumerate(zip(["Home win", "Draw", "Away win"], [ph, pd_, pa])):
        tags = []
        if i == predicted_idx:
            tags.append("← predicted")
        if actual_idx is not None and i == actual_idx:
            tags.append("← actual")
        pc[i].metric(f"P({label})", f"{prob:.1%}",
                     delta="  |  ".join(tags) if tags else None,
                     delta_color="normal" if correct else ("off" if i == predicted_idx else "normal"))

    colors_bar = []
    for i in range(3):
        if i == actual_idx and i == predicted_idx:
            colors_bar.append("#16a34a")   # correct prediction — green
        elif i == actual_idx:
            colors_bar.append("#d97706")   # actual but not predicted — amber
        elif i == predicted_idx:
            colors_bar.append("#dc2626")   # predicted but wrong — red
        else:
            colors_bar.append("#94a3b8")   # neither — grey

    fig = go.Figure(go.Bar(
        x=[ph, pd_, pa], y=["Home win", "Draw", "Away win"], orientation="h",
        marker_color=colors_bar,
        text=[f"{p:.1%}" for p in [ph, pd_, pa]], textposition="auto",
    ))
    fig.update_layout(height=160, margin=dict(l=0, r=0, t=4, b=0),
                      xaxis=dict(range=[0, 1], showgrid=False), yaxis=dict(showgrid=False))
    st.plotly_chart(fig, width='stretch', key=f"inspector_proba_{selected_id}")
    st.caption("🟢 Correct prediction  |  🟡 Actual outcome (not predicted)  |  🔴 Predicted (wrong)  |  ⚫ Neither")

    # key stats side-by-side
    st.markdown("---")
    st.subheader("Team comparison — key stats")
    key_pairs = [
        ("Elo rating", "home_elo", "away_elo"),
        ("Elo-implied P(home win)", "elo_expected_home", None),
        ("Win rate (last 3)", "home_w3_win_rate", "away_w3_win_rate"),
        ("Win rate (last 5)", "home_w5_win_rate", "away_w5_win_rate"),
        ("Win rate (last 10)", "home_w10_win_rate", "away_w10_win_rate"),
        ("PPG (last 5)", "home_w5_ppg", "away_w5_ppg"),
        ("Goals scored/game (last 5)", "home_w5_avg_gf", "away_w5_avg_gf"),
        ("Goals conceded/game (last 5)", "home_w5_avg_ga", "away_w5_avg_ga"),
        ("Goal diff/game (last 5)", "home_w5_avg_gd", "away_w5_avg_gd"),
        ("Clean sheet rate (last 5)", "home_w5_clean_sheet_rate", "away_w5_clean_sheet_rate"),
        ("League position", "home_league_pos", "away_league_pos"),
        ("Relegation pressure", "home_relegation_pressure", "away_relegation_pressure"),
        ("Title race", "home_title_race", "away_title_race"),
        ("Rest days", "home_rest_days", "away_rest_days"),
        ("Matches last 30d", "home_congestion_30d", "away_congestion_30d"),
        ("H2H win rate", "h2h_home_win_rate", "h2h_away_win_rate"),
        ("H2H draw rate", "h2h_draw_rate", None),
        ("H2H meetings (5y)", "h2h_games", None),
        ("H2H avg goals", "h2h_avg_goals", None),
    ]
    compare_rows = []
    for label, hcol, acol in key_pairs:
        hv = row.get(hcol, np.nan) if hcol and hcol in row.index else np.nan
        av = row.get(acol, np.nan) if acol and acol in row.index else np.nan
        if pd.isna(hv) and pd.isna(av):
            continue
        r = {"Metric": label, home: round(float(hv), 3) if pd.notna(hv) else None}
        if acol is not None:
            r[away] = round(float(av), 3) if pd.notna(av) else None
        compare_rows.append(r)
    if compare_rows:
        st.dataframe(pd.DataFrame(compare_rows), width='stretch', hide_index=True)

    # full feature table
    st.markdown("---")
    st.subheader("Full feature table")
    st.caption("🔵 Blue = home team feature  |  🔴 Red = away team feature")

    groups = {
        "Elo": [c for c in avail if "elo" in c],
        "Form (last 3)": [c for c in avail if "w3_" in c],
        "Form (last 5)": [c for c in avail if "w5_" in c],
        "Form (last 10)": [c for c in avail if "w10_" in c],
        "Head-to-Head": [c for c in avail if "h2h" in c],
        "Context": [c for c in avail if any(k in c for k in
                    ["rest", "congestion", "season_stage", "referee",
                     "league_pos", "position_gap", "relegation", "title_race"])],
        "xG / Advanced": [c for c in avail if any(k in c for k in ["xg", "ppda", "shot"])],
        "Squad": [c for c in avail if any(k in c for k in ["squad", "market", "age"])],
    }
    group_sel = st.selectbox("Filter by group", ["All"] + [g for g, cols in groups.items() if cols],
                             key=f"group_sel_{selected_id}")
    cols_to_show = avail if group_sel == "All" else groups.get(group_sel, avail)

    feat_rows = [
        {
            "Feature": feat,
            "Value": round(float(row.get(feat, np.nan)), 4) if pd.notna(row.get(feat, np.nan)) else None,
            "Description": FEATURE_DESCRIPTIONS.get(feat, ""),
        }
        for feat in cols_to_show
    ]
    df_feat = pd.DataFrame(feat_rows)

    def highlight_home_away(s):
        if s["Feature"].startswith("home_"):
            return ["background-color: #dbeafe"] * len(s)
        elif s["Feature"].startswith("away_"):
            return ["background-color: #fee2e2"] * len(s)
        return [""] * len(s)

    st.dataframe(df_feat.style.apply(highlight_home_away, axis=1),
                 width='stretch', height=480)

    # SHAP (if pre-computed)
    if "shap_values" in st.session_state:
        st.markdown("---")
        st.subheader("SHAP — what drove this prediction?")
        shap_df_full = st.session_state["shap_df"]
        shap_vals = st.session_state["shap_values"]
        shap_feats = st.session_state["shap_features"]
        match_idx_rows = shap_df_full[shap_df_full["match_id"] == selected_id].index
        if len(match_idx_rows) > 0:
            idx = match_idx_rows[0]
            contributions = (
                pd.DataFrame({
                    "Feature": shap_feats,
                    "SHAP": shap_vals[idx],
                    "Value": st.session_state["shap_X"][idx],
                })
                .sort_values("SHAP", key=abs, ascending=False)
                .head(15)
            )
            bar_colors = ["#2563eb" if v > 0 else "#dc2626" for v in contributions["SHAP"]]
            fig = go.Figure(go.Bar(
                x=contributions["SHAP"], y=contributions["Feature"],
                orientation="h", marker_color=bar_colors,
                hovertemplate="%{y}<br>SHAP: %{x:.4f}<br>Feature value: %{customdata:.3f}<extra></extra>",
                customdata=contributions["Value"],
            ))
            fig.update_layout(height=450,
                              xaxis_title="SHAP value (positive = pushes toward home win)",
                              margin=dict(l=0, r=0))
            st.plotly_chart(fig, width='stretch', key=f"inspector_shap_{selected_id}")
            st.caption("🔵 Positive = pushed toward home win  |  🔴 Negative = pushed toward draw/away win")
        else:
            st.info("This match was not in the SHAP sample.")
    else:
        st.info("Go to Feature Importance tab and click 'Compute SHAP values' to see per-match explanations.")


def tab_inspector(features_df, models):
    st.header("Match Inspector — Predictions vs Reality")

    if features_df is None or models is None:
        st.warning("Features and trained models are required.")
        return

    ensemble, feature_cols = models
    avail = [c for c in feature_cols if c in features_df.columns]

    # ── filters ───────────────────────────────────────────────────────────────
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        f_league = st.selectbox("League", ["All"] + sorted(features_df["league"].unique()),
                                key="insp_league")
    with fc2:
        all_seasons = sorted(features_df["season"].unique(), reverse=True)
        f_season = st.selectbox("Season", ["All"] + [str(s) for s in all_seasons],
                                key="insp_season")
    with fc3:
        f_correct = st.selectbox("Result filter",
                                 ["All", "Correct only", "Wrong only"],
                                 key="insp_correct")
    with fc4:
        f_min_prob = st.slider("Min. predicted probability", 0.0, 0.9, 0.0, 0.05,
                               key="insp_minprob",
                               help="Only show matches where the model was at least this confident")

    # ── batch predictions ─────────────────────────────────────────────────────
    with st.spinner("Computing predictions for all matches…"):
        scored_df = _batch_predict(ensemble, tuple(feature_cols), features_df)

    # apply filters
    view = scored_df.copy()
    if f_league != "All":
        view = view[view["league"] == f_league]
    if f_season != "All":
        view = view[view["season"] == int(f_season)]
    if f_correct == "Correct only":
        view = view[view["correct"]]
    elif f_correct == "Wrong only":
        view = view[~view["correct"]]
    if f_min_prob > 0:
        view = view[view["max_prob"] >= f_min_prob]

    view = view.sort_values("date", ascending=False)

    if view.empty:
        st.info("No matches match the current filters.")
        return

    # ── summary stats ─────────────────────────────────────────────────────────
    n_total = len(view)
    n_correct = view["correct"].sum()
    avg_conf_correct = view[view["correct"]]["max_prob"].mean() if n_correct > 0 else 0
    avg_conf_wrong = view[~view["correct"]]["max_prob"].mean() if n_total - n_correct > 0 else 0

    sc = st.columns(4)
    sc[0].metric("Matches shown", n_total)
    sc[1].metric("Correct predictions", f"{n_correct} ({n_correct/n_total:.1%})")
    sc[2].metric("Avg confidence (correct)", f"{avg_conf_correct:.1%}")
    sc[3].metric("Avg confidence (wrong)", f"{avg_conf_wrong:.1%}")

    # ── overview table ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Predictions overview")
    st.caption("🟢 Correct prediction  |  🔴 Wrong prediction")

    overview = view[[
        "date", "league", "matchday", "home_team", "away_team",
        "prob_home", "prob_draw", "prob_away",
        "predicted", "result", "correct", "max_prob",
    ]].copy()
    overview["date"] = overview["date"].dt.strftime("%Y-%m-%d")
    overview["predicted"] = overview["predicted"].map(RESULT_LABELS)
    overview["actual"] = overview["result"].astype(int).map(RESULT_LABELS)
    overview["confidence"] = overview["max_prob"].apply(lambda x: f"{x:.1%}")
    for col in ["prob_home", "prob_draw", "prob_away"]:
        overview[col] = overview[col].apply(lambda x: f"{x:.1%}")

    display_cols = ["date", "league", "home_team", "away_team",
                    "prob_home", "prob_draw", "prob_away",
                    "predicted", "actual", "confidence", "correct"]
    overview_display = overview[display_cols].rename(columns={
        "date": "Date", "league": "League",
        "home_team": "Home", "away_team": "Away",
        "prob_home": "P(Home)", "prob_draw": "P(Draw)", "prob_away": "P(Away)",
        "predicted": "Predicted", "actual": "Actual",
        "confidence": "Confidence", "correct": "Correct",
    })

    styled = overview_display.style.apply(_style_overview, axis=1)
    st.dataframe(styled, width='stretch', height=420, hide_index=True)

    # ── match deep dive ────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Deep dive — select a match")

    match_options = {
        row_s["match_id"]: (
            f"{'✅' if row_s['correct'] else '❌'}  "
            f"{str(row_s['date'])[:10]}  |  {row_s['league']}  |  "
            f"{row_s['home_team']} vs {row_s['away_team']}  "
            f"(predicted: {RESULT_LABELS[int(row_s['predicted_idx'])]}  /  actual: {RESULT_LABELS[int(row_s['result'])]})"
        )
        for _, row_s in view.assign(
            predicted_idx=view["predicted"]
        ).sort_values("date", ascending=False).iterrows()
    }

    selected_id = st.selectbox(
        "Match",
        list(match_options.keys()),
        format_func=lambda x: match_options[x],
        key="insp_match_sel",
    )

    match_row = scored_df[scored_df["match_id"] == selected_id].iloc[0]
    st.markdown("---")
    _deep_dive(match_row, ensemble, avail, selected_id)


# ── tab 4: upcoming matches ───────────────────────────────────────────────────

_KEY_PAIRS = [
    ("Elo rating",                  "home_elo",                   "away_elo"),
    ("Elo-implied P(home win)",     "elo_expected_home",          None),
    ("Win rate (last 5)",           "home_w5_win_rate",           "away_w5_win_rate"),
    ("PPG (last 5)",                "home_w5_ppg",                "away_w5_ppg"),
    ("Goals scored/game (last 5)",  "home_w5_avg_gf",             "away_w5_avg_gf"),
    ("Goals conceded/game (last 5)","home_w5_avg_ga",             "away_w5_avg_ga"),
    ("Win rate (last 10)",          "home_w10_win_rate",          "away_w10_win_rate"),
    ("League position",             "home_league_pos",            "away_league_pos"),
    ("Rest days",                   "home_rest_days",             "away_rest_days"),
    ("Congestion (30d)",            "home_congestion_30d",        "away_congestion_30d"),
    ("H2H win rate",                "h2h_home_win_rate",          "h2h_away_win_rate"),
    ("H2H matches (5y)",            "h2h_games",                  None),
    ("Relegation pressure",         "home_relegation_pressure",   "away_relegation_pressure"),
    ("Title race",                  "home_title_race",            "away_title_race"),
]


def _render_key_features(features: dict, home: str, away: str) -> None:
    """Render a two-column key-features table from a flat feature dict."""
    rows = []
    for label, hcol, acol in _KEY_PAIRS:
        hv = features.get(hcol)
        av = features.get(acol) if acol else None
        if hv is None and av is None:
            continue
        row = {"Metric": label, home: hv}
        if acol is not None:
            row[away] = av
        rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _render_shap_upcoming(features: dict, feature_cols: list, ensemble, home: str, away: str, key: str) -> None:
    """Compute and render SHAP waterfall chart for the upcoming match."""
    try:
        import shap
    except ImportError:
        st.caption("Install `shap` for explainability: `pip install shap`")
        return

    X = np.array([[features.get(c, 0.0) for c in feature_cols]], dtype=np.float32)

    # Use the XGBoost sub-model if available (best SHAP support).
    # ensemble.models["xgboost"] is a CalibratedModel wrapping XGBoostModel wrapping XGBClassifier.
    calibrated = ensemble.models.get("xgboost")
    if calibrated is None:
        st.caption("SHAP requires the XGBoost sub-model.")
        return
    xgb_wrapper = getattr(calibrated, "model", calibrated)   # CalibratedModel → XGBoostModel
    raw_model = getattr(xgb_wrapper, "_model", None)          # XGBoostModel → XGBClassifier
    if raw_model is None:
        st.caption("Could not access raw XGBoost model for SHAP.")
        return
    # Use the underlying booster for best multi-class SHAP compatibility
    try:
        raw_model = raw_model.get_booster()
    except Exception:
        pass

    outcome_labels = ["Home win", "Draw", "Away win"]
    outcome_sel = st.radio("Explain probability for:", outcome_labels,
                           horizontal=True, key=f"shap_outcome_{key}")
    class_idx = outcome_labels.index(outcome_sel)

    try:
        explainer = shap.TreeExplainer(raw_model)
        shap_vals = explainer.shap_values(X)  # shape: (3, 1, n_features) or (1, n_features, 3)
        # Handle different shap output shapes
        if isinstance(shap_vals, list):
            sv = shap_vals[class_idx][0]
            base = float(explainer.expected_value[class_idx])
        elif shap_vals.ndim == 3:
            sv = shap_vals[0, :, class_idx]
            base = float(explainer.expected_value[class_idx])
        else:
            sv = shap_vals[0]
            base = float(explainer.expected_value)

        # Top 15 features by absolute SHAP value
        top_idx = np.argsort(np.abs(sv))[::-1][:15]
        top_features = [feature_cols[i] for i in top_idx]
        top_shap = sv[top_idx]
        top_values = [features.get(c, 0.0) for c in top_features]

        colors = ["#2563eb" if v > 0 else "#dc2626" for v in top_shap]
        fig = go.Figure(go.Bar(
            x=top_shap[::-1],
            y=[f"{f}={v:.3f}" for f, v in zip(top_features[::-1], top_values[::-1])],
            orientation="h",
            marker_color=colors[::-1],
            text=[f"{v:+.3f}" for v in top_shap[::-1]],
            textposition="auto",
        ))
        fig.update_layout(
            title=f"SHAP contributions → P({outcome_sel})  [base={base:.3f}]",
            height=420,
            margin=dict(l=0, r=0, t=36, b=0),
            xaxis_title="SHAP value (impact on model output)",
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True, key=f"shap_fig_{key}")
        st.caption("Blue = pushes probability UP  |  Red = pushes probability DOWN")
    except Exception as e:
        st.warning(f"SHAP computation failed: {e}")


def tab_upcoming(features_df, models):
    st.header("Upcoming Match Predictions")

    preds = load_predictions_json()
    if preds is None:
        st.warning("No predictions.json found. Run `python -m output.predict --matchday next`.")
        return

    st.markdown(f"*Generated at: {preds['generated_at']}*")
    matches = preds["matches"]

    c1, c2 = st.columns(2)
    with c1:
        leagues = ["All"] + sorted({m["league"] for m in matches})
        lf = st.selectbox("League", leagues, key="up_league")
    with c2:
        value_only = st.checkbox("Value bets only", value=False)

    if lf != "All":
        matches = [m for m in matches if m["league"] == lf]
    if value_only:
        matches = [m for m in matches if len(m.get("value_bets", [])) > 0]

    if not matches:
        st.info("No matches match the current filters.")
        return

    for mi, m in enumerate(matches):
        p = m["prediction"]
        vb_n = len(m.get("value_bets", []))
        header = (f"⚽ **{m['home_team']}** vs **{m['away_team']}**  "
                  f"— {m['date']}  |  {m['league']}"
                  + (f"  🎯 {vb_n} value bet(s)" if vb_n > 0 else ""))

        with st.expander(header):
            # ── probabilities ──────────────────────────────────────────────
            cols = st.columns([2, 2, 2, 1])
            cols[0].metric("P(Home win)", f"{p['home_win']:.1%}")
            cols[1].metric("P(Draw)", f"{p['draw']:.1%}")
            cols[2].metric("P(Away win)", f"{p['away_win']:.1%}")
            cols[3].metric("Confidence", p["confidence"].upper())

            fig = go.Figure(go.Bar(
                x=[p["home_win"], p["draw"], p["away_win"]],
                y=["Home win", "Draw", "Away win"],
                orientation="h",
                marker_color=["#2563eb", "#d97706", "#dc2626"],
                text=[f"{v:.1%}" for v in [p["home_win"], p["draw"], p["away_win"]]],
                textposition="auto",
            ))
            fig.update_layout(height=130, margin=dict(l=0, r=0, t=6, b=0),
                              xaxis=dict(range=[0, 1], showgrid=False),
                              yaxis=dict(showgrid=False))
            st.plotly_chart(fig, use_container_width=True, key=f"upcoming_prob_{mi}")

            # ── feature breakdown ──────────────────────────────────────────
            features = m.get("features", {})
            if features:
                st.markdown("---")
                ft1, ft2 = st.tabs(["📋 Key Features", "🔍 All Features"])

                with ft1:
                    _render_key_features(features, m["home_team"], m["away_team"])

                with ft2:
                    groups = {
                        "Elo":          [c for c in features if "elo" in c],
                        "Form (last 3)":[c for c in features if "w3_" in c],
                        "Form (last 5)":[c for c in features if "w5_" in c],
                        "Form (last 10)":[c for c in features if "w10_" in c],
                        "Head-to-Head": [c for c in features if "h2h" in c],
                        "Context":      [c for c in features if any(k in c for k in
                                         ["rest", "congestion", "season_stage",
                                          "league_pos", "relegation", "title_race"])],
                        "xG / Advanced":[c for c in features if any(k in c for k in
                                         ["xg", "ppda", "shot"])],
                    }
                    grp = st.selectbox("Group", ["All"] + [g for g, cols in groups.items() if cols],
                                       key=f"grp_{mi}")
                    show_cols = list(features.keys()) if grp == "All" else groups.get(grp, [])
                    all_feat_rows = [
                        {
                            "Feature": c,
                            "Value": features[c],
                            "Description": FEATURE_DESCRIPTIONS.get(c, ""),
                        }
                        for c in show_cols if c in features
                    ]
                    if all_feat_rows:
                        st.dataframe(pd.DataFrame(all_feat_rows),
                                     use_container_width=True, hide_index=True)

            # ── SHAP explanation ───────────────────────────────────────────
            if features and models is not None:
                ensemble, feature_cols = models
                st.markdown("---")
                st.markdown("**Why this prediction? (SHAP)**")
                _render_shap_upcoming(features, feature_cols, ensemble,
                                      m["home_team"], m["away_team"], key=str(mi))

            # ── value bets ─────────────────────────────────────────────────
            if vb_n > 0:
                st.markdown("---")
                st.markdown("**Value bets:**")
                vb_df = pd.DataFrame(m["value_bets"])[
                    ["bookmaker", "outcome", "model_prob", "bookmaker_odds",
                     "implied_prob", "edge", "kelly", "confidence_tier"]
                ].copy()
                for col in ["model_prob", "implied_prob", "edge", "kelly"]:
                    if col in vb_df.columns:
                        vb_df[col] = vb_df[col].apply(lambda x: f"{float(x):.1%}")
                st.dataframe(vb_df, use_container_width=True, hide_index=True)

            # ── odds comparison ────────────────────────────────────────────
            if m.get("odds_comparison"):
                st.markdown("---")
                st.markdown("**Bookmaker odds:**")
                odds_df = pd.DataFrame(m["odds_comparison"])
                keep = [c for c in ["bookmaker", "home_odds", "draw_odds", "away_odds",
                                    "home_edge", "draw_edge", "away_edge"] if c in odds_df.columns]
                st.dataframe(odds_df[keep], use_container_width=True, hide_index=True)


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    render_sidebar()
    features_df = load_features()
    models = load_models()

    tabs = st.tabs([
        "📊 Model Performance",
        "🔍 Feature Importance",
        "🔬 Match Inspector",
        "📅 Upcoming Matches",
    ])
    with tabs[0]:
        tab_performance(features_df, models)
    with tabs[1]:
        tab_features(features_df, models)
    with tabs[2]:
        tab_inspector(features_df, models)
    with tabs[3]:
        tab_upcoming(features_df, models)


if __name__ == "__main__":
    main()
