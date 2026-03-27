import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import joblib
import folium
from folium.features import GeoJsonTooltip
from streamlit_folium import st_folium
import os

# ── Page config ──
st.set_page_config(
    page_title="Ghana Crop Insurance Dashboard",
    page_icon="🌾",
    layout="wide",
)

# ── Custom CSS: brown/earth-tone theme ──
st.markdown("""
<style>
    .main { background-color: #fdf6ec; }
    .stApp > header { background-color: #5C4033; }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #3e2723 0%, #5d4037 100%);
        color: #efebe9;
    }
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] span {
        color: #efebe9 !important;
    }
    .stat-card {
        background: linear-gradient(135deg, #3e2723 0%, #5d4037 100%);
        border-radius: 10px;
        padding: 20px;
        color: #efebe9;
        text-align: center;
    }
    .stat-card h2 { color: #ffcc80; margin: 0; font-size: 2rem; }
    .stat-card p { margin: 5px 0 0 0; font-size: 0.9rem; color: #bcaaa4; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
MODELS_DIR = os.path.join(BASE, "models")


@st.cache_resource
def load_model():
    model = joblib.load(os.path.join(MODELS_DIR, "xgboost_crop_loss.joblib"))
    feature_cols = joblib.load(os.path.join(MODELS_DIR, "feature_columns.joblib"))
    return model, feature_cols


@st.cache_data
def load_data():
    ghana = gpd.read_file(os.path.join(DATA, "gadm41_GHA_2.json")).to_crs("EPSG:4326")
    rainfall = pd.read_csv(os.path.join(DATA, "district_rainfall.csv"))
    soil = pd.read_csv(os.path.join(DATA, "district_soil_properties.csv"))
    hazard = pd.read_csv(os.path.join(DATA, "district_hazard.csv"))
    return ghana, rainfall, soil, hazard


def predict_for_year(year, threshold, payout_rate, model, feature_cols,
                     rainfall, soil, hazard):
    rain_year = rainfall[rainfall["year"] == year]
    rain_pivot = rain_year.pivot_table(
        index="district", columns="season",
        values=["rainfall_mm", "anomaly_pct"], aggfunc="first"
    )
    rain_pivot.columns = [f"{v}_{s}" for v, s in rain_pivot.columns]
    rain_pivot = rain_pivot.reset_index()

    features = rain_pivot.merge(soil, on="district", how="inner")
    features = features.merge(
        hazard[["district", "year", "rfh", "r3h", "rfq", "rfh_ratio", "r3h_ratio"]]
        [hazard["year"] == year],
        on="district", how="left"
    )

    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0
    features[feature_cols] = features[feature_cols].fillna(0)

    if len(features) == 0:
        return None

    X = features[feature_cols].values
    probs = model.predict_proba(X)[:, 1]
    features["loss_prob"] = probs
    features["payout_triggered"] = probs > threshold
    features["payout_ghs"] = np.where(features["payout_triggered"], payout_rate, 0)
    return features


def main():
    model, feature_cols = load_model()
    ghana, rainfall, soil, hazard = load_data()
    available_years = sorted(rainfall["year"].unique())

    # ── Sidebar controls ──
    with st.sidebar:
        st.image("https://flagcdn.com/w320/gh.png", width=60)
        st.title("🌾 Crop Insurance")
        st.markdown("**Ghana Drought Risk Dashboard**")
        st.markdown("---")

        year = st.selectbox("Select Year", available_years,
                            index=len(available_years) - 1)
        threshold = st.slider("Payout Threshold", 0.3, 0.9, 0.6, 0.05)
        payout_rate = st.number_input("Payout (GHS/ha)", 100, 5000, 500, 50)

        st.markdown("---")
        st.caption("Parametric crop insurance model using ")
        st.caption("CHIRPS rainfall + iSDA soil + HDX hazard data")

    # ── Run prediction ──
    preds = predict_for_year(year, threshold, payout_rate, model,
                             feature_cols, rainfall, soil, hazard)

    if preds is None:
        st.error(f"No data available for {year}")
        return

    # ── Summary stats ──
    triggered = int(preds["payout_triggered"].sum())
    total = len(preds)
    avg_prob = preds["loss_prob"].mean()
    total_payout = preds["payout_ghs"].sum()

    st.title(f"🌾 Crop Insurance Dashboard — {year}")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{triggered}/{total}</h2>
            <p>Districts Triggering Payout</p>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{avg_prob:.1%}</h2>
            <p>Avg Loss Probability</p>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h2>{threshold:.0%}</h2>
            <p>Payout Threshold</p>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <h2>GHS {total_payout:,.0f}</h2>
            <p>Total Simulated Payout</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Map ──
    map_data = ghana.merge(
        preds[["district", "loss_prob", "payout_triggered", "payout_ghs"]],
        left_on="NAME_2", right_on="district", how="left"
    )
    map_data["loss_prob"] = map_data["loss_prob"].fillna(0)
    map_data["loss_pct"] = (map_data["loss_prob"] * 100).round(1)

    m = folium.Map(location=[7.95, -1.02], zoom_start=7,
                   tiles="CartoDB positron")

    folium.Choropleth(
        geo_data=map_data.__geo_interface__,
        data=map_data,
        columns=["NAME_2", "loss_prob"],
        key_on="feature.properties.NAME_2",
        fill_color="RdYlGn_r",
        fill_opacity=0.7,
        line_opacity=0.3,
        legend_name="Loss Probability",
    ).add_to(m)

    style_fn = lambda x: {"fillColor": "#ffffff", "color": "#000000",
                          "fillOpacity": 0.0, "weight": 0.1}
    highlight_fn = lambda x: {"fillColor": "#000000", "color": "#000000",
                              "fillOpacity": 0.3, "weight": 0.3}

    folium.features.GeoJson(
        map_data,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=GeoJsonTooltip(
            fields=["NAME_2", "loss_pct", "payout_ghs"],
            aliases=["District:", "Loss Prob (%):", "Payout (GHS):"],
            sticky=True
        )
    ).add_to(m)

    col_map, col_table = st.columns([3, 2])

    with col_map:
        st.subheader("District Risk Map")
        st_folium(m, width=700, height=550)

    with col_table:
        st.subheader("Top 15 Districts by Loss Probability")
        top = (preds[["district", "loss_prob", "payout_triggered", "payout_ghs"]]
               .sort_values("loss_prob", ascending=False)
               .head(15).reset_index(drop=True))
        top["loss_prob"] = (top["loss_prob"] * 100).round(1).astype(str) + "%"
        top.columns = ["District", "Loss Prob", "Triggered", "Payout (GHS)"]
        st.dataframe(top, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("Payout Distribution")
        st.bar_chart(
            preds.sort_values("loss_prob", ascending=False)
            .head(20).set_index("district")["loss_prob"],
            color="#5d4037"
        )

    # ── Footer ──
    st.markdown("---")
    st.caption(
        "Parametric Crop Insurance Dashboard · Built with CHIRPS, HDX, iSDA data · "
        "XGBoost drought risk model · Paa Kwesi Gyan Turkson"
    )


if __name__ == "__main__":
    main()