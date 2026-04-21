import streamlit as st
import joblib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from fpdf import FPDF
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
import tempfile, os, warnings
warnings.filterwarnings('ignore')

# ── Load model & saved artifacts ───────────────────────────────────
model    = joblib.load("model/model.pkl")
FEATURES = joblib.load("model/features.pkl")

bat_labels = {0: "Poor", 1: "Average", 2: "Good", 3: "Excellent"}
bat_colors = {0: "#e74c3c", 1: "#f39c12", 2: "#3498db", 3: "#2ecc71"}
bat_emojis = {0: "❌", 1: "⚠️", 2: "👍", 3: "✅"}
bowl_labels = {0: "Poor", 1: "Average", 2: "Good"}
bowl_colors = {0: "#e74c3c", 1: "#f39c12", 2: "#2ecc71"}

# ── Cache data ─────────────────────────────────────────────────────
@st.cache_data
def load_data():
    deliveries = pd.read_csv("data/deliveries.csv")
    matches    = pd.read_csv("data/matches.csv")
    return deliveries, matches

@st.cache_data
def get_all_batsman_stats():
    deliveries, matches = load_data()
    col = 'batsman' if 'batsman' in deliveries.columns else 'batter'
    df  = deliveries.merge(matches[['id','season']], left_on='match_id', right_on='id')
    stats = df.groupby(col).agg(
        total_runs    =('batsman_runs','sum'),
        total_balls   =('ball','count'),
        matches_played=('match_id','nunique')
    ).reset_index()
    fours = df[df['batsman_runs']==4].groupby(col).size().rename('total_fours')
    sixes = df[df['batsman_runs']==6].groupby(col).size().rename('total_sixes')
    dots  = df[df['batsman_runs']==0].groupby(col).size().rename('dot_balls')
    stats = stats.merge(fours, on=col, how='left')
    stats = stats.merge(sixes, on=col, how='left')
    stats = stats.merge(dots,  on=col, how='left')
    stats[['total_fours','total_sixes','dot_balls']] = \
        stats[['total_fours','total_sixes','dot_balls']].fillna(0)
    stats['avg_runs']      = stats['total_runs']   / stats['matches_played']
    stats['strike_rate']   = (stats['total_runs']  / stats['total_balls']) * 100
    stats['boundary_rate'] = (stats['total_fours'] + stats['total_sixes']) / stats['total_balls']
    stats['dot_ball_rate'] = stats['dot_balls']    / stats['total_balls']
    seasons_per_player     = df.groupby(col)['season'].nunique()
    stats['runs_per_season']= stats['total_runs'] / stats[col].map(seasons_per_player)
    stats = stats[stats['matches_played'] >= 5].dropna()
    return stats.rename(columns={col: 'player'})

@st.cache_data
def get_batsman_stats_season(season_filter):
    deliveries, matches = load_data()
    if season_filter != "All":
        ids        = matches[matches['season']==int(season_filter)]['id']
        deliveries = deliveries[deliveries['match_id'].isin(ids)]
    col = 'batsman' if 'batsman' in deliveries.columns else 'batter'
    df  = deliveries.merge(matches[['id']], left_on='match_id', right_on='id')
    stats = df.groupby(col).agg(
        total_runs    =('batsman_runs','sum'),
        total_balls   =('ball','count'),
        matches_played=('match_id','nunique')
    ).reset_index()
    stats['avg_runs']    = stats['total_runs']  / stats['matches_played']
    stats['strike_rate'] = (stats['total_runs'] / stats['total_balls']) * 100
    return stats.rename(columns={col:'player'})

@st.cache_data
def get_batsman_season_history(player_name):
    deliveries, matches = load_data()
    col = 'batsman' if 'batsman' in deliveries.columns else 'batter'
    df  = deliveries[deliveries[col]==player_name].merge(
          matches[['id','season']], left_on='match_id', right_on='id')
    h = df.groupby('season').agg(
        total_runs    =('batsman_runs','sum'),
        total_balls   =('ball','count'),
        matches_played=('match_id','nunique')
    ).reset_index()
    h['avg_runs']    = h['total_runs']  / h['matches_played']
    h['strike_rate'] = (h['total_runs'] / h['total_balls']) * 100
    return h.sort_values('season')

@st.cache_data
def get_bowler_stats(season_filter):
    deliveries, matches = load_data()
    if season_filter != "All":
        ids        = matches[matches['season']==int(season_filter)]['id']
        deliveries = deliveries[deliveries['match_id'].isin(ids)]
    df = deliveries.merge(matches[['id']], left_on='match_id', right_on='id')
    wk = ['caught','bowled','lbw','stumped','caught and bowled','hit wicket']
    df['is_wicket'] = df['dismissal_kind'].isin(wk)
    stats = df.groupby('bowler').agg(
        total_runs_given=('total_runs','sum'),
        total_balls     =('ball','count'),
        total_wickets   =('is_wicket','sum'),
        matches_played  =('match_id','nunique')
    ).reset_index()
    stats['overs']       = stats['total_balls'] / 6
    stats['economy']     = stats['total_runs_given'] / stats['overs']
    stats['avg_wickets'] = stats['total_wickets']    / stats['matches_played']
    return stats.rename(columns={'bowler':'player'})

@st.cache_data
def get_bowler_season_history(player_name):
    deliveries, matches = load_data()
    wk = ['caught','bowled','lbw','stumped','caught and bowled','hit wicket']
    deliveries['is_wicket'] = deliveries['dismissal_kind'].isin(wk)
    df = deliveries[deliveries['bowler']==player_name].merge(
         matches[['id','season']], left_on='match_id', right_on='id')
    h = df.groupby('season').agg(
        total_runs_given=('total_runs','sum'),
        total_balls     =('ball','count'),
        total_wickets   =('is_wicket','sum'),
        matches_played  =('match_id','nunique')
    ).reset_index()
    h['overs']       = h['total_balls'] / 6
    h['economy']     = h['total_runs_given'] / h['overs']
    h['avg_wickets'] = h['total_wickets']    / h['matches_played']
    return h.sort_values('season')

# ── REAL Feature Importance from actual data ───────────────────────
@st.cache_data
def get_real_bat_feature_importance():
    """Train a fresh RF on batsman data and return REAL importances"""
    data = get_all_batsman_stats()
    feat_names = ['avg_runs','strike_rate','boundary_rate',
                  'dot_ball_rate','matches_played','total_fours','total_sixes']
    def label(avg):
        if avg >= 30:   return 3
        elif avg >= 20: return 2
        elif avg >= 10: return 1
        else:           return 0
    data['label'] = data['avg_runs'].apply(label)
    X = data[feat_names].fillna(0).values
    y = data['label'].values
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return feat_names, rf.feature_importances_

@st.cache_data
def get_real_bowl_feature_importance():
    """Train a fresh RF on bowler data and return REAL importances"""
    data = get_bowler_stats("All")
    data = data[data['matches_played'] >= 5].dropna()
    feat_names = ['economy','avg_wickets','matches_played']
    def label_bowl(row):
        score = 0
        if row['economy'] < 7:        score += 2
        elif row['economy'] < 9:      score += 1
        if row['avg_wickets'] >= 2:   score += 2
        elif row['avg_wickets'] >= 1: score += 1
        if row['matches_played'] >= 30: score += 1
        if score >= 4:   return 2
        elif score >= 2: return 1
        else:            return 0
    data['label'] = data.apply(label_bowl, axis=1)
    X = data[feat_names].fillna(0).values
    y = data['label'].values
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    return feat_names, rf.feature_importances_

# ── Smart backend functions ────────────────────────────────────────
@st.cache_data
def compute_clusters(n=3):
    data = get_all_batsman_stats()
    feats = ['avg_runs','strike_rate','boundary_rate','dot_ball_rate']
    sc    = StandardScaler()
    X     = sc.fit_transform(data[feats].fillna(0))
    km    = KMeans(n_clusters=n, random_state=42, n_init=10)
    data['cluster'] = km.fit_predict(X)
    rank = data.groupby('cluster')['avg_runs'].mean().rank(ascending=False)
    names = {c: ["Power Hitter","Aggressive Anchor",
                 "Steady Anchor","Finisher","Accumulator"][i]
             for i, c in enumerate(rank.sort_values().index)}
    data['style'] = data['cluster'].map(names)
    return data[['player','style','avg_runs','strike_rate','boundary_rate']]

@st.cache_data
def compute_outliers():
    data  = get_all_batsman_stats()
    feats = ['avg_runs','strike_rate','boundary_rate','matches_played']
    iso   = IsolationForest(contamination=0.08, random_state=42)
    data['outlier'] = iso.fit_predict(data[feats].fillna(0))
    return set(data[data['outlier']==-1]['player'].tolist())

@st.cache_data
def compute_regression_model():
    data = get_all_batsman_stats()
    X    = data[['strike_rate','matches_played','boundary_rate']].fillna(0).values
    y    = data['avg_runs'].values
    sc   = StandardScaler()
    lr   = LinearRegression()
    lr.fit(sc.fit_transform(X), y)
    return lr, sc

@st.cache_data
def compute_similar_players(player_name, n=4):
    data  = get_all_batsman_stats()
    feats = ['avg_runs','strike_rate','boundary_rate','dot_ball_rate','matches_played']
    sc    = StandardScaler()
    X     = sc.fit_transform(data[feats].fillna(0))
    nn    = NearestNeighbors(n_neighbors=n+1, metric='euclidean')
    nn.fit(X)
    if player_name not in data['player'].values:
        return []
    idx  = data[data['player']==player_name].index[0]
    pos  = data.index.get_loc(idx)
    _, idxs = nn.kneighbors([X[pos]])
    return [data.iloc[i]['player']
            for i in idxs[0]
            if data.iloc[i]['player'] != player_name][:n]

@st.cache_data
def compute_similar_bowlers(player_name, n=4):
    data  = get_bowler_stats("All")
    data  = data[data['matches_played'] >= 5].dropna()
    feats = ['economy','avg_wickets','matches_played']
    sc    = StandardScaler()
    X     = sc.fit_transform(data[feats].fillna(0))
    nn    = NearestNeighbors(n_neighbors=n+1, metric='euclidean')
    nn.fit(X)
    if player_name not in data['player'].values:
        return []
    idx  = data[data['player']==player_name].index[0]
    pos  = data.index.get_loc(idx)
    _, idxs = nn.kneighbors([X[pos]])
    return [data.iloc[i]['player']
            for i in idxs[0]
            if data.iloc[i]['player'] != player_name][:n]

def get_player_style(player_name):
    clusters = compute_clusters()
    row = clusters[clusters['player']==player_name]
    return row['style'].values[0] if len(row) else "Unknown"

def is_exceptional(player_name):
    return player_name in compute_outliers()

def get_projected_runs(strike_rate, matches, boundary_rate):
    lr, sc = compute_regression_model()
    X = sc.transform([[strike_rate, matches, boundary_rate]])
    return round(float(lr.predict(X)[0]), 2)

def predict_bowler(economy, avg_wickets, matches):
    score = 0
    if economy < 7:        score += 2
    elif economy < 9:      score += 1
    if avg_wickets >= 2:   score += 2
    elif avg_wickets >= 1: score += 1
    if matches >= 30:      score += 1
    if score >= 4:   return 2
    elif score >= 2: return 1
    else:            return 0

def export_pdf(mode, player, season, stats_dict,
               prediction, confidence, style=None, exceptional=False):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica","B",20)
    pdf.set_text_color(30,30,30)
    pdf.cell(0,12,"Cricket Performance Report",ln=True,align="C")
    pdf.set_font("Helvetica","",11)
    pdf.set_text_color(120,120,120)
    pdf.cell(0,8,"IPL Performance Predictor - AI Powered",ln=True,align="C")
    pdf.ln(4)
    pdf.set_draw_color(200,200,200)
    pdf.line(10,pdf.get_y(),200,pdf.get_y())
    pdf.ln(5)
    pdf.set_font("Helvetica","B",13)
    pdf.set_text_color(30,30,30)
    pdf.cell(0,9,f"Player  : {player}",ln=True)
    pdf.cell(0,9,f"Mode    : {mode}",ln=True)
    pdf.cell(0,9,f"Season  : {season}",ln=True)
    if style:
        pdf.cell(0,9,f"Style   : {style}",ln=True)
    if exceptional:
        pdf.cell(0,9,"Status  : Exceptional Player",ln=True)
    pdf.ln(3)
    pdf.set_font("Helvetica","B",13)
    pdf.cell(0,9,"Stats",ln=True)
    pdf.set_font("Helvetica","",12)
    for k,v in stats_dict.items():
        pdf.cell(0,8,f"  {k}: {v}",ln=True)
    pdf.ln(3)
    cmap = {"Poor":(231,76,60),"Average":(243,156,18),
            "Good":(46,204,113),"Excellent":(46,204,113)}
    r,g,b = cmap.get(prediction,(30,30,30))
    pdf.set_font("Helvetica","B",15)
    pdf.set_text_color(r,g,b)
    pdf.cell(0,10,f"Prediction : {prediction}",ln=True)
    pdf.set_text_color(30,30,30)
    pdf.set_font("Helvetica","",12)
    pdf.cell(0,8,f"Confidence : {confidence}",ln=True)
    tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".pdf")
    pdf.output(tmp.name)
    return tmp.name

def plot_feature_importance(feat_names, importances, title, color):
    display_names = {
        'avg_runs':       'Avg Runs',
        'strike_rate':    'Strike Rate',
        'boundary_rate':  'Boundary Rate',
        'dot_ball_rate':  'Dot Ball Rate',
        'matches_played': 'Matches Played',
        'total_fours':    'Total Fours',
        'total_sixes':    'Total Sixes',
        'economy':        'Economy Rate',
        'avg_wickets':    'Avg Wickets',
    }
    names = [display_names.get(f, f) for f in feat_names]
    feat_df = pd.DataFrame({'feature': names, 'importance': importances})
    feat_df = feat_df.sort_values('importance', ascending=True)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#1e2130')
    ax.tick_params(colors='#aaa')
    for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
    bar_colors = [color if i == len(feat_df)-1
                  else '#2e5f8a' for i in range(len(feat_df))]
    bars = ax.barh(feat_df['feature'], feat_df['importance'],
                   color=bar_colors, height=0.6)
    ax.set_xlabel('Importance Score', color='#aaa')
    ax.set_title(title, color='white', fontsize=12)
    for bar, val in zip(bars, feat_df['importance']):
        ax.text(bar.get_width()+0.002,
                bar.get_y()+bar.get_height()/2,
                f'{val:.1%}', va='center', color='white', fontsize=9)
    plt.tight_layout()
    return fig

# ── Custom CSS ─────────────────────────────────────────────────────
st.markdown("""
<style>
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    background:#1e2130;border-radius:10px;
    padding:8px 20px;color:#ccc;font-weight:600;
}
.stTabs [aria-selected="true"] {
    background:linear-gradient(135deg,#1a73e8,#0d47a1);
    color:white !important;
}
.metric-card {
    background:#1e2130;border-radius:14px;padding:18px;
    text-align:center;border:1px solid #2e3250;margin-bottom:8px;
}
.metric-card .val { font-size:26px;font-weight:700;color:#4fc3f7; }
.metric-card .lbl { font-size:12px;color:#888;margin-top:4px; }
.badge {
    padding:22px;border-radius:16px;text-align:center;
    color:white;font-size:30px;font-weight:800;margin:14px 0;
}
.style-badge {
    display:inline-block;background:#1a3a5c;border:1px solid #1a73e8;
    color:#4fc3f7;border-radius:20px;padding:5px 16px;
    font-size:14px;font-weight:600;margin:4px 2px;
}
.exceptional-badge {
    display:inline-block;background:#3a1a1a;border:1px solid #e74c3c;
    color:#ef5350;border-radius:20px;padding:5px 16px;
    font-size:14px;font-weight:600;margin:4px 2px;
}
.similar-card {
    background:#1e2130;border-radius:10px;padding:12px;
    border:1px solid #2e3250;text-align:center;
}
.section-title {
    font-size:16px;font-weight:700;color:#4fc3f7;
    margin:16px 0 8px;border-left:4px solid #1a73e8;padding-left:10px;
}
.compare-card {
    background:#1e2130;border-radius:14px;
    padding:16px;border:1px solid #2e3250;
}
div[data-testid="stButton"] button {
    background:linear-gradient(135deg,#1a73e8,#0d47a1);
    color:white;border:none;border-radius:10px;
    font-weight:700;font-size:15px;padding:12px;width:100%;
}
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:20px 0 10px'>
  <span style='font-size:48px'>🏏</span>
  <h1 style='color:#4fc3f7;margin:6px 0 4px;font-size:34px'>
  Cricket Performance Predictor</h1>
  <p style='color:#888;font-size:14px'>
  Real IPL data · AI-powered · Smart insights</p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

_, matches_df = load_data()
seasons = ["All"] + sorted(matches_df['season'].unique().tolist(), reverse=True)

tab1, tab2, tab3 = st.tabs(["🏏 Batsman", "🎯 Bowler", "⚔️ Compare Players"])

# ══════════════════════════════════════════════════════
# TAB 1 — BATSMAN
# ══════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">📅 Season Filter</div>', unsafe_allow_html=True)
    season_bat = st.selectbox("Season", seasons, key="bat_season")
    bat_stats  = get_batsman_stats_season(season_bat)

    st.markdown('<div class="section-title">🔍 Search Batsman</div>', unsafe_allow_html=True)
    bat_names = sorted(bat_stats['player'].tolist())
    sel_bat   = st.selectbox("Select player",
                ["-- Select a player --"] + bat_names, key="bat_player")

    if sel_bat != "-- Select a player --":
        all_stats = get_all_batsman_stats()
        row_s     = bat_stats[bat_stats['player']==sel_bat].iloc[0]
        avg_runs       = float(round(row_s['avg_runs'],2))
        strike_rate    = float(round(row_s['strike_rate'],2))
        matches_played = int(row_s['matches_played'])

        if sel_bat in all_stats['player'].values:
            full_row      = all_stats[all_stats['player']==sel_bat].iloc[0]
            boundary_rate = float(full_row['boundary_rate'])
            dot_ball_rate = float(full_row['dot_ball_rate'])
            total_fours   = int(full_row['total_fours'])
            total_sixes   = int(full_row['total_sixes'])
        else:
            boundary_rate = dot_ball_rate = 0.0
            total_fours = total_sixes = 0

        style       = get_player_style(sel_bat)
        exceptional = is_exceptional(sel_bat)
        projected   = get_projected_runs(strike_rate, matches_played, boundary_rate)

        st.success(f"✅ Stats loaded for **{sel_bat}** — Season: {season_bat}")
        badges_html = f'<span class="style-badge">🎯 {style}</span>'
        if exceptional:
            badges_html += '<span class="exceptional-badge">⚡ Exceptional Player</span>'
        st.markdown(badges_html, unsafe_allow_html=True)
        st.markdown("---")

        # Metric cards
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="val">{avg_runs}</div><div class="lbl">Avg Runs/Match</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="val">{strike_rate}</div><div class="lbl">Strike Rate</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="val">{matches_played}</div><div class="lbl">Matches Played</div></div>', unsafe_allow_html=True)
        with c4:
            st.markdown(f'<div class="metric-card"><div class="val" style="color:#f9a825">{projected}</div><div class="lbl">Projected Avg Runs</div></div>', unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="val">{total_fours}</div><div class="lbl">Total Fours</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="val">{total_sixes}</div><div class="lbl">Total Sixes</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="val">{round(boundary_rate*100,1)}%</div><div class="lbl">Boundary Rate</div></div>', unsafe_allow_html=True)

        # Season history
        st.markdown('<div class="section-title">📈 Season History</div>', unsafe_allow_html=True)
        history = get_batsman_season_history(sel_bat)
        if len(history) > 1:
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,3.5))
            fig.patch.set_facecolor('#0e1117')
            for ax in [ax1,ax2]:
                ax.set_facecolor('#1e2130')
                ax.tick_params(colors='#aaa')
                for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
            ax1.plot(history['season'],history['avg_runs'],marker='o',
                     color='#4fc3f7',linewidth=2.5,markersize=7)
            ax1.fill_between(history['season'],history['avg_runs'],
                             alpha=0.15,color='#4fc3f7')
            ax1.set_title('Avg Runs per Season',color='white',fontsize=12)
            ax1.set_xlabel('Season',color='#aaa')
            ax1.set_ylabel('Avg Runs',color='#aaa')
            ax2.plot(history['season'],history['strike_rate'],marker='s',
                     color='#f9a825',linewidth=2.5,markersize=7)
            ax2.fill_between(history['season'],history['strike_rate'],
                             alpha=0.15,color='#f9a825')
            ax2.set_title('Strike Rate per Season',color='white',fontsize=12)
            ax2.set_xlabel('Season',color='#aaa')
            ax2.set_ylabel('Strike Rate',color='#aaa')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Select 'All' seasons to see full history chart.")

        # REAL Feature Importance
        st.markdown('<div class="section-title">🧠 Feature Importance (from Real Model)</div>',
                    unsafe_allow_html=True)
        feat_names_real, importances_real = get_real_bat_feature_importance()
        fig_fi = plot_feature_importance(
            feat_names_real, importances_real,
            'Which stat matters most for batting prediction?', '#4fc3f7')
        st.pyplot(fig_fi)

        # Radar Chart
        st.markdown('<div class="section-title">🕸️ Player Radar Chart</div>',
                    unsafe_allow_html=True)
        all_stats_radar = get_all_batsman_stats()
        if sel_bat in all_stats_radar['player'].values:
            p_row = all_stats_radar[all_stats_radar['player']==sel_bat].iloc[0]

            def normalize(val, col):
                mn = all_stats_radar[col].min()
                mx = all_stats_radar[col].max()
                return round((val-mn)/(mx-mn)*100,1) if mx > mn else 50

            categories = ['Avg Runs','Strike Rate','Boundary %',
                          'Consistency','Experience']
            raw_vals = [
                normalize(p_row['avg_runs'],       'avg_runs'),
                normalize(p_row['strike_rate'],    'strike_rate'),
                normalize(p_row['boundary_rate'],  'boundary_rate'),
                100 - normalize(p_row['dot_ball_rate'], 'dot_ball_rate'),
                normalize(p_row['matches_played'], 'matches_played'),
            ]
            N      = len(categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            vals   = raw_vals + raw_vals[:1]

            fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e2130')
            ax.spines['polar'].set_color('#2e3250')
            ax.tick_params(colors='#aaa')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, color='#aaa', fontsize=9)
            ax.set_ylim(0,100)
            ax.set_yticks([20,40,60,80,100])
            ax.set_yticklabels(['20','40','60','80','100'],color='#555',fontsize=7)
            ax.grid(color='#2e3250', linewidth=0.5)
            ax.plot(angles, vals, color='#4fc3f7', linewidth=2.5)
            ax.fill(angles, vals, color='#4fc3f7', alpha=0.25)
            ax.scatter(angles[:-1], raw_vals, color='#4fc3f7', s=60, zorder=5)
            ax.set_title(f'{sel_bat} — Skill Radar',color='white',fontsize=12,pad=20)
            plt.tight_layout()
            st.pyplot(fig)

        # Similar Players
        st.markdown('<div class="section-title">👥 Similar Players</div>',
                    unsafe_allow_html=True)
        similar = compute_similar_players(sel_bat)
        if similar:
            cols = st.columns(len(similar))
            for col, name in zip(cols, similar):
                s_style = get_player_style(name)
                s_data  = all_stats[all_stats['player']==name]
                s_avg   = round(s_data['avg_runs'].values[0],1) if len(s_data) else "N/A"
                with col:
                    st.markdown(f"""
                    <div class="similar-card">
                      <div style='color:#4fc3f7;font-weight:700;font-size:13px'>{name}</div>
                      <div style='color:#888;font-size:11px;margin:4px 0'>{s_style}</div>
                      <div style='color:white;font-size:13px'>Avg: {s_avg}</div>
                    </div>""", unsafe_allow_html=True)

        # Top 10 Leaderboard
        st.markdown('<div class="section-title">🏆 Top 10 Leaderboard</div>',
                    unsafe_allow_html=True)
        all_lb  = get_all_batsman_stats().copy()
        lb_rows = []
        for _, r in all_lb.iterrows():
            try:
                X_lb = np.array([[
                    r['avg_runs'],r['strike_rate'],r['boundary_rate'],
                    r['dot_ball_rate'],r['matches_played'],
                    r['total_fours'],r['total_sixes']
                ]])
                pred_lb = model.predict(X_lb)[0]
                prob_lb = model.predict_proba(X_lb)[0]
                lb_rows.append({
                    'Player':      r['player'],
                    'Avg Runs':    round(r['avg_runs'],1),
                    'Strike Rate': round(r['strike_rate'],1),
                    'Matches':     int(r['matches_played']),
                    'Prediction':  bat_labels[pred_lb],
                    'Confidence':  f"{max(prob_lb):.1%}"
                })
            except Exception:
                continue

        lb_df = pd.DataFrame(lb_rows)
        top10 = lb_df[lb_df['Prediction'].isin(['Excellent','Good'])]\
                    .sort_values('Avg Runs', ascending=False)\
                    .head(10).reset_index(drop=True)
        top10.index += 1

        def color_pred(val):
            m = {'Excellent':'background-color:#1a3a1a;color:#2ecc71;font-weight:700',
                 'Good':     'background-color:#1a2a3a;color:#3498db;font-weight:700'}
            return m.get(val,'')

        st.dataframe(top10.style.applymap(color_pred, subset=['Prediction']),
                     use_container_width=True)

        if sel_bat in top10['Player'].values:
            rank = top10[top10['Player']==sel_bat].index[0]
            st.success(f"🏆 **{sel_bat}** is ranked **#{rank}** in the Top 10!")
        else:
            st.info(f"💡 {sel_bat} is not in Top 10")

        # Predict button
        st.markdown("---")
        if st.button("🔍 Predict Performance", use_container_width=True):
            X_input = np.array([[
                avg_runs, strike_rate, boundary_rate,
                dot_ball_rate, matches_played, total_fours, total_sixes
            ]])
            pred  = model.predict(X_input)[0]
            prob  = model.predict_proba(X_input)[0]
            label = bat_labels[pred]
            color = bat_colors[pred]
            emoji = bat_emojis[pred]
            conf  = f"{max(prob):.1%}"

            st.markdown(
                f'<div class="badge" style="background:{color}">'
                f'{emoji} {label} &nbsp;·&nbsp; '
                f'<span style="font-size:18px">Confidence: {conf}</span></div>',
                unsafe_allow_html=True)

            st.markdown('<div class="section-title">📊 Confidence Breakdown</div>',
                        unsafe_allow_html=True)
            fig,ax = plt.subplots(figsize=(6,2.5))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e2130')
            ax.tick_params(colors='#aaa')
            for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
            n_classes    = len(prob)
            class_labels = [bat_labels[i] for i in range(n_classes)]
            class_colors = [bat_colors[i]  for i in range(n_classes)]
            bars = ax.barh(class_labels, prob, color=class_colors, height=0.5)
            ax.set_xlim(0,1)
            ax.set_xlabel('Probability', color='#aaa')
            for bar,val in zip(bars,prob):
                ax.text(bar.get_width()+0.01,
                        bar.get_y()+bar.get_height()/2,
                        f'{val:.1%}', va='center', color='white', fontsize=10)
            st.pyplot(fig)

            st.markdown('<div class="section-title">📄 Export Report</div>',
                        unsafe_allow_html=True)
            pdf_path = export_pdf("Batsman", sel_bat, season_bat,
                {"Avg Runs/Match": avg_runs, "Strike Rate": strike_rate,
                 "Matches Played": matches_played,
                 "Boundary Rate":  f"{round(boundary_rate*100,1)}%",
                 "Projected Avg Runs": projected},
                label, conf, style=style, exceptional=exceptional)
            with open(pdf_path,"rb") as f:
                st.download_button("⬇️ Download PDF Report", f,
                    file_name=f"{sel_bat}_report.pdf",
                    mime="application/pdf", use_container_width=True)
            os.unlink(pdf_path)

    else:
        st.info("👆 Select a season and player to begin")

# ══════════════════════════════════════════════════════
# TAB 2 — BOWLER
# ══════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">📅 Season Filter</div>', unsafe_allow_html=True)
    season_bowl = st.selectbox("Season", seasons, key="bowl_season")
    bowl_stats  = get_bowler_stats(season_bowl)

    st.markdown('<div class="section-title">🔍 Search Bowler</div>', unsafe_allow_html=True)
    bowl_names = sorted(bowl_stats['player'].tolist())
    sel_bowl   = st.selectbox("Select player",
                 ["-- Select a player --"]+bowl_names, key="bowl_player")

    if sel_bowl != "-- Select a player --":
        row         = bowl_stats[bowl_stats['player']==sel_bowl].iloc[0]
        economy     = float(round(row['economy'],2))
        avg_wickets = float(round(row['avg_wickets'],2))
        b_matches   = int(row['matches_played'])

        st.success(f"✅ Stats loaded for **{sel_bowl}** — Season: {season_bowl}")
        st.markdown("---")

        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-card"><div class="val">{economy}</div><div class="lbl">Economy Rate</div></div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-card"><div class="val">{avg_wickets}</div><div class="lbl">Avg Wickets/Match</div></div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-card"><div class="val">{b_matches}</div><div class="lbl">Matches Played</div></div>', unsafe_allow_html=True)

        # Season history
        st.markdown('<div class="section-title">📈 Season History</div>',
                    unsafe_allow_html=True)
        bh = get_bowler_season_history(sel_bowl)
        if len(bh) > 1:
            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,3.5))
            fig.patch.set_facecolor('#0e1117')
            for ax in [ax1,ax2]:
                ax.set_facecolor('#1e2130')
                ax.tick_params(colors='#aaa')
                for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
            ax1.plot(bh['season'],bh['economy'],marker='o',
                     color='#ef5350',linewidth=2.5,markersize=7)
            ax1.fill_between(bh['season'],bh['economy'],alpha=0.15,color='#ef5350')
            ax1.set_title('Economy per Season',color='white',fontsize=12)
            ax1.set_xlabel('Season',color='#aaa')
            ax1.set_ylabel('Economy',color='#aaa')
            ax2.plot(bh['season'],bh['avg_wickets'],marker='s',
                     color='#66bb6a',linewidth=2.5,markersize=7)
            ax2.fill_between(bh['season'],bh['avg_wickets'],alpha=0.15,color='#66bb6a')
            ax2.set_title('Avg Wickets per Season',color='white',fontsize=12)
            ax2.set_xlabel('Season',color='#aaa')
            ax2.set_ylabel('Avg Wickets',color='#aaa')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Select 'All' seasons to see full history.")

        # REAL Bowler Feature Importance
        st.markdown('<div class="section-title">🧠 Feature Importance (from Real Model)</div>',
                    unsafe_allow_html=True)
        bowl_feat_names_real, bowl_importances_real = get_real_bowl_feature_importance()
        fig_bfi = plot_feature_importance(
            bowl_feat_names_real, bowl_importances_real,
            'Which stat matters most for bowler prediction?', '#ef5350')
        st.pyplot(fig_bfi)

        # Bowler Radar Chart
        st.markdown('<div class="section-title">🕸️ Bowler Radar Chart</div>',
                    unsafe_allow_html=True)
        all_bowl_stats = get_bowler_stats("All")
        if sel_bowl in all_bowl_stats['player'].values:
            b_row = all_bowl_stats[all_bowl_stats['player']==sel_bowl].iloc[0]

            def normalize_bowl(val, col):
                mn = all_bowl_stats[col].min()
                mx = all_bowl_stats[col].max()
                return round((val-mn)/(mx-mn)*100,1) if mx > mn else 50

            b_categories = ['Wickets\nAbility','Economy\n(lower=better)',
                            'Experience','Consistency','Match\nImpact']
            b_raw_vals = [
                normalize_bowl(b_row['avg_wickets'],  'avg_wickets'),
                100 - normalize_bowl(b_row['economy'],'economy'),
                normalize_bowl(b_row['matches_played'],'matches_played'),
                100 - normalize_bowl(b_row['economy'], 'economy'),
                normalize_bowl(b_row['total_wickets'], 'total_wickets'),
            ]
            N      = len(b_categories)
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]
            b_vals = b_raw_vals + b_raw_vals[:1]

            fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#1e2130')
            ax.spines['polar'].set_color('#2e3250')
            ax.tick_params(colors='#aaa')
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(b_categories, color='#aaa', fontsize=9)
            ax.set_ylim(0,100)
            ax.set_yticks([20,40,60,80,100])
            ax.set_yticklabels(['20','40','60','80','100'],color='#555',fontsize=7)
            ax.grid(color='#2e3250', linewidth=0.5)
            ax.plot(angles, b_vals, color='#ef5350', linewidth=2.5)
            ax.fill(angles, b_vals, color='#ef5350', alpha=0.25)
            ax.scatter(angles[:-1], b_raw_vals, color='#ef5350', s=60, zorder=5)
            ax.set_title(f'{sel_bowl} — Bowler Radar',color='white',fontsize=12,pad=20)
            plt.tight_layout()
            st.pyplot(fig)

        # Similar Bowlers
        st.markdown('<div class="section-title">👥 Similar Bowlers</div>',
                    unsafe_allow_html=True)
        sim_bowlers = compute_similar_bowlers(sel_bowl)
        if sim_bowlers:
            b_cols = st.columns(len(sim_bowlers))
            for col, name in zip(b_cols, sim_bowlers):
                b_data = all_bowl_stats[all_bowl_stats['player']==name]
                b_eco  = round(b_data['economy'].values[0],1) if len(b_data) else "N/A"
                b_wkt  = round(b_data['avg_wickets'].values[0],2) if len(b_data) else "N/A"
                with col:
                    st.markdown(f"""
                    <div class="similar-card">
                      <div style='color:#ef5350;font-weight:700;font-size:13px'>{name}</div>
                      <div style='color:#888;font-size:11px;margin:4px 0'>Economy: {b_eco}</div>
                      <div style='color:white;font-size:13px'>Avg Wkts: {b_wkt}</div>
                    </div>""", unsafe_allow_html=True)

        # Top 10 Bowlers Leaderboard
        st.markdown('<div class="section-title">🏆 Top 10 Bowlers Leaderboard</div>',
                    unsafe_allow_html=True)
        all_bowl_lb = get_bowler_stats("All").copy()
        all_bowl_lb = all_bowl_lb[all_bowl_lb['matches_played'] >= 5].dropna()
        b_lb_rows = []
        for _, r in all_bowl_lb.iterrows():
            pred_b = predict_bowler(r['economy'], r['avg_wickets'], r['matches_played'])
            b_lb_rows.append({
                'Player':      r['player'],
                'Economy':     round(r['economy'],2),
                'Avg Wickets': round(r['avg_wickets'],2),
                'Matches':     int(r['matches_played']),
                'Prediction':  bowl_labels[pred_b],
            })
        b_lb_df = pd.DataFrame(b_lb_rows)
        b_top10 = b_lb_df[b_lb_df['Prediction']=='Good']\
                      .sort_values('Avg Wickets', ascending=False)\
                      .head(10).reset_index(drop=True)
        b_top10.index += 1

        def color_bowl_pred(val):
            m = {'Good':    'background-color:#1a3a1a;color:#2ecc71;font-weight:700',
                 'Average': 'background-color:#2a2a1a;color:#f39c12',
                 'Poor':    'background-color:#2a1a1a;color:#e74c3c'}
            return m.get(val,'')

        st.dataframe(b_top10.style.applymap(color_bowl_pred, subset=['Prediction']),
                     use_container_width=True)
        if sel_bowl in b_top10['Player'].values:
            b_rank = b_top10[b_top10['Player']==sel_bowl].index[0]
            st.success(f"🏆 **{sel_bowl}** is ranked **#{b_rank}** in Top 10 Bowlers!")
        else:
            st.info(f"💡 {sel_bowl} is not in Top 10 bowlers")

        st.markdown("---")
        if st.button("🎯 Predict Bowler Performance", use_container_width=True):
            pred  = predict_bowler(economy, avg_wickets, b_matches)
            label = bowl_labels[pred]
            color = bowl_colors[pred]

            st.markdown(
                f'<div class="badge" style="background:{color}">🎯 {label}</div>',
                unsafe_allow_html=True)

            st.markdown('<div class="section-title">📋 Stats Summary</div>',
                        unsafe_allow_html=True)
            st.table({"Stat":  ["Economy Rate","Avg Wickets/Match","Matches Played"],
                      "Value": [economy, avg_wickets, b_matches]})

            st.markdown('<div class="section-title">📄 Export Report</div>',
                        unsafe_allow_html=True)
            pdf_path = export_pdf("Bowler", sel_bowl, season_bowl,
                {"Economy Rate": economy, "Avg Wickets/Match": avg_wickets,
                 "Matches Played": b_matches},
                label, "Rule-based")
            with open(pdf_path,"rb") as f:
                st.download_button("⬇️ Download PDF Report", f,
                    file_name=f"{sel_bowl}_report.pdf",
                    mime="application/pdf", use_container_width=True)
            os.unlink(pdf_path)
    else:
        st.info("👆 Select a season and player to begin")

# ══════════════════════════════════════════════════════
# TAB 3 — COMPARE PLAYERS
# ══════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">⚔️ Compare Two Batsmen</div>',
                unsafe_allow_html=True)
    all_bat   = get_all_batsman_stats()
    all_names = sorted(all_bat['player'].tolist())

    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div style="color:#4fc3f7;font-weight:700;margin-bottom:6px">Player 1</div>',
                    unsafe_allow_html=True)
        p1 = st.selectbox("",["-- Select --"]+all_names, key="cmp1")
    with col2:
        st.markdown('<div style="color:#f9a825;font-weight:700;margin-bottom:6px">Player 2</div>',
                    unsafe_allow_html=True)
        p2 = st.selectbox("",["-- Select --"]+all_names, key="cmp2")

    if p1!="-- Select --" and p2!="-- Select --" and p1!=p2:
        r1 = all_bat[all_bat['player']==p1].iloc[0]
        r2 = all_bat[all_bat['player']==p2].iloc[0]
        s1 = get_player_style(p1)
        s2 = get_player_style(p2)
        e1 = is_exceptional(p1)
        e2 = is_exceptional(p2)

        st.markdown("---")
        c1,c2 = st.columns(2)

        def compare_card(player, row, style, exceptional, color):
            exc = '<span style="color:#ef5350;font-size:12px"> ⚡ Exceptional</span>' \
                  if exceptional else ''
            return f"""
            <div class="compare-card" style="border-top:3px solid {color}">
              <div style="color:{color};font-size:16px;font-weight:700">{player}{exc}</div>
              <div style="color:#4fc3f7;font-size:12px;margin:4px 0 10px">🎯 {style}</div>
              <div style="display:flex;justify-content:space-between;margin:6px 0">
                <span style="color:#888">Avg Runs</span>
                <span style="color:white;font-weight:600">{round(row['avg_runs'],2)}</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin:6px 0">
                <span style="color:#888">Strike Rate</span>
                <span style="color:white;font-weight:600">{round(row['strike_rate'],2)}</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin:6px 0">
                <span style="color:#888">Boundary Rate</span>
                <span style="color:white;font-weight:600">{round(row['boundary_rate']*100,1)}%</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin:6px 0">
                <span style="color:#888">Dot Ball Rate</span>
                <span style="color:white;font-weight:600">{round(row['dot_ball_rate']*100,1)}%</span>
              </div>
              <div style="display:flex;justify-content:space-between;margin:6px 0">
                <span style="color:#888">Matches</span>
                <span style="color:white;font-weight:600">{int(row['matches_played'])}</span>
              </div>
            </div>"""

        with c1: st.markdown(compare_card(p1,r1,s1,e1,"#4fc3f7"), unsafe_allow_html=True)
        with c2: st.markdown(compare_card(p2,r2,s2,e2,"#f9a825"), unsafe_allow_html=True)

        st.markdown('<div class="section-title">📊 Head to Head</div>',
                    unsafe_allow_html=True)
        v1 = [r1['avg_runs'],r1['strike_rate'],
              r1['boundary_rate']*100,r1['dot_ball_rate']*100]
        v2 = [r2['avg_runs'],r2['strike_rate'],
              r2['boundary_rate']*100,r2['dot_ball_rate']*100]
        labels_ = ['Avg Runs','Strike Rate','Boundary %','Dot Ball %']
        x = np.arange(len(labels_))
        fig,ax = plt.subplots(figsize=(9,3.5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1e2130')
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
        ax.bar(x-0.2, v1, 0.35, label=p1, color='#4fc3f7', alpha=0.9)
        ax.bar(x+0.2, v2, 0.35, label=p2, color='#f9a825', alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(labels_, color='#aaa')
        ax.legend(facecolor='#1e2130', labelcolor='white')
        plt.tight_layout(); st.pyplot(fig)

        st.markdown('<div class="section-title">📈 Season Trend</div>',
                    unsafe_allow_html=True)
        h1 = get_batsman_season_history(p1)
        h2 = get_batsman_season_history(p2)
        fig,ax = plt.subplots(figsize=(9,3.5))
        fig.patch.set_facecolor('#0e1117')
        ax.set_facecolor('#1e2130')
        ax.tick_params(colors='#aaa')
        for spine in ax.spines.values(): spine.set_edgecolor('#2e3250')
        ax.plot(h1['season'],h1['avg_runs'],marker='o',color='#4fc3f7',
                linewidth=2.5,markersize=7,label=p1)
        ax.plot(h2['season'],h2['avg_runs'],marker='s',color='#f9a825',
                linewidth=2.5,markersize=7,label=p2)
        ax.fill_between(h1['season'],h1['avg_runs'],alpha=0.1,color='#4fc3f7')
        ax.fill_between(h2['season'],h2['avg_runs'],alpha=0.1,color='#f9a825')
        ax.set_xlabel('Season',color='#aaa'); ax.set_ylabel('Avg Runs',color='#aaa')
        ax.legend(facecolor='#1e2130', labelcolor='white')
        plt.tight_layout(); st.pyplot(fig)

        st.markdown('<div class="section-title">🤖 AI Prediction</div>',
                    unsafe_allow_html=True)
        pc1,pc2 = st.columns(2)
        for col,player,row in [(pc1,p1,r1),(pc2,p2,r2)]:
            X_in = np.array([[row['avg_runs'],row['strike_rate'],
                              row['boundary_rate'],row['dot_ball_rate'],
                              row['matches_played'],row['total_fours'],
                              row['total_sixes']]])
            pred = model.predict(X_in)[0]
            prob = model.predict_proba(X_in)[0]
            lbl  = bat_labels[pred]
            clr  = bat_colors[pred]
            emj  = bat_emojis[pred]
            with col:
                st.markdown(
                    f'<div class="badge" style="background:{clr};font-size:20px">'
                    f'{emj} {player}<br>'
                    f'<span style="font-size:26px">{lbl}</span><br>'
                    f'<span style="font-size:13px;opacity:0.8">'
                    f'Confidence: {max(prob):.1%}</span></div>',
                    unsafe_allow_html=True)
    elif p1==p2 and p1!="-- Select --":
        st.warning("Please select two different players.")
    else:
        st.info("👆 Select two players above to compare")