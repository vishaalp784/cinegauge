import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ── PAGE CONFIG ───────────────────────────────────────────
st.set_page_config(
    page_title="CineGauge",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── DARK CINEMA THEME ─────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0A0A0F; color: #F5F0E8; }
    .main { background-color: #0A0A0F; }
    section[data-testid="stSidebar"] { background-color: #0D0B15; border-right: 1px solid #C9A84C33; }
    h1, h2, h3 { color: #C9A84C; font-family: Georgia, serif; }
    .metric-card { background: #141420; border: 1px solid #C9A84C33; border-radius: 12px; padding: 20px; text-align: center; }
    .metric-val { font-size: 42px; font-weight: 900; color: #C9A84C; font-family: Georgia; }
    .metric-lbl { font-size: 13px; color: #9A9A9A; margin-top: 4px; }
    .result-pos { background: #0a1a0a; border: 1.5px solid #4CAF50; border-radius: 12px; padding: 20px; }
    .result-neg { background: #1a0a0a; border: 1.5px solid #B52B2B; border-radius: 12px; padding: 20px; }
    .result-neu { background: #1a1a0a; border: 1.5px solid #C9A84C; border-radius: 12px; padding: 20px; }
    .stTextArea textarea { background-color: #141420 !important; color: #F5F0E8 !important; border: 1px solid #C9A84C44 !important; }
    .stButton > button { background: #C9A84C; color: #0A0A0F; font-weight: 700; border: none; border-radius: 8px; padding: 10px 24px; font-size: 15px; }
    .stButton > button:hover { background: #E8C96A; }
    .stSelectbox > div { background-color: #141420 !important; }
    div[data-testid="stMetric"] { background: #141420; border: 1px solid #C9A84C33; border-radius: 10px; padding: 14px; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("vishaalp784/cinegauge-sentiment")
    model     = AutoModelForSequenceClassification.from_pretrained("vishaalp784/cinegauge-sentiment")
    model.eval()
    return tokenizer, model

# ── LOAD DATA ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/movies.csv")
    GENRES = {
        28:"Action",12:"Adventure",16:"Animation",35:"Comedy",
        80:"Crime",99:"Documentary",18:"Drama",10751:"Family",
        14:"Fantasy",36:"History",27:"Horror",10402:"Music",
        9648:"Mystery",10749:"Romance",878:"Sci-Fi",
        53:"Thriller",10752:"War",37:"Western"
    }
    def parse_genres(g):
        try:
            ids = ast.literal_eval(str(g))
            return [GENRES.get(i,"Other") for i in ids[:2]]
        except:
            return ["Other"]
    df["genres"]        = df["genre_ids"].apply(parse_genres)
    df["primary_genre"] = df["genres"].apply(lambda x: x[0] if x else "Other")
    df["release_year"]  = pd.to_numeric(df["release_year"], errors="coerce")
    df = df.dropna(subset=["release_year"])
    df["release_year"]  = df["release_year"].astype(int)
    return df

# ── PREDICT ───────────────────────────────────────────────
def predict(text, tokenizer, model):
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=256, padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    score = probs[0][1].item()
    if score >= 0.65:
        label = "POSITIVE"
    elif score <= 0.35:
        label = "NEGATIVE"
    else:
        label = "MIXED"
    return label, score

# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎬 CineGauge")
    st.markdown("*Cinema Intelligence Platform*")
    st.markdown("---")
    page = st.selectbox(
        "Navigate",
        ["🎭 Sentiment Analyzer", "📊 Box Office Dashboard", "🔍 Film Explorer"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown(f"""
    <div style='font-size:12px;color:#9A9A9A;line-height:1.8'>
    <b style='color:#C9A84C'>Model:</b> DistilBERT<br>
    <b style='color:#C9A84C'>Accuracy:</b> 92.04%<br>
    <b style='color:#C9A84C'>Trained on:</b> 50,000 reviews<br>
    <b style='color:#C9A84C'>Dataset:</b> 1,678 films<br>
    <b style='color:#C9A84C'>Built by:</b> Vishaal Pedapatnam
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# PAGE 1 — SENTIMENT ANALYZER
# ════════════════════════════════════════════════════════
if page == "🎭 Sentiment Analyzer":
    st.title("🎭 Sentiment Analyzer")
    st.markdown("*Paste any film review — our AI model will analyze the sentiment instantly*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model", "DistilBERT")
    with col2:
        st.metric("Accuracy", "92.04%")
    with col3:
        st.metric("Trained on", "50,000 reviews")

    st.markdown("###")
    review = st.text_area(
        "Paste your film review here:",
        placeholder="e.g. This film was an absolute masterpiece. The cinematography was breathtaking and the performances were unforgettable...",
        height=160,
    )

    col_btn, col_ex = st.columns([1, 4])
    with col_btn:
        analyze = st.button("🎬 Analyze")
    with col_ex:
        if st.button("Try an example"):
            review = "This film was absolutely incredible. The director crafted a masterpiece with stunning visuals and powerful performances that left me speechless."

    if analyze and review.strip():
        with st.spinner("🧠 Analyzing..."):
            tokenizer, model = load_model()
            label, score = predict(review, tokenizer, model)

        st.markdown("### Result")

        if label == "POSITIVE":
            st.markdown(f"""
            <div class='result-pos'>
                <div style='font-size:36px;margin-bottom:8px'>😊 POSITIVE</div>
                <div style='font-size:18px;color:#4CAF50;font-weight:700'>Confidence: {score*100:.1f}%</div>
                <div style='font-size:13px;color:#9A9A9A;margin-top:8px'>The model detects a positive audience sentiment in this review.</div>
            </div>
            """, unsafe_allow_html=True)
        elif label == "NEGATIVE":
            st.markdown(f"""
            <div class='result-neg'>
                <div style='font-size:36px;margin-bottom:8px'>😞 NEGATIVE</div>
                <div style='font-size:18px;color:#B52B2B;font-weight:700'>Confidence: {(1-score)*100:.1f}%</div>
                <div style='font-size:13px;color:#9A9A9A;margin-top:8px'>The model detects a negative audience sentiment in this review.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class='result-neu'>
                <div style='font-size:36px;margin-bottom:8px'>😐 MIXED</div>
                <div style='font-size:18px;color:#C9A84C;font-weight:700'>Score: {score*100:.1f}%</div>
                <div style='font-size:13px;color:#9A9A9A;margin-top:8px'>The model detects mixed or neutral sentiment in this review.</div>
            </div>
            """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("###")
        fig_bar = go.Figure(go.Bar(
            x=[score * 100, (1-score) * 100],
            y=["Positive", "Negative"],
            orientation="h",
            marker_color=["#4CAF50", "#B52B2B"],
            text=[f"{score*100:.1f}%", f"{(1-score)*100:.1f}%"],
            textposition="inside",
            textfont=dict(size=15, color="white", family="Georgia"),
        ))
        fig_bar.update_layout(
            plot_bgcolor="#0A0A0F", paper_bgcolor="#141420",
            font=dict(color="#F5F0E8", size=14, family="Georgia"),
            height=160, margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(range=[0,100], gridcolor="#1a1a25"),
            yaxis=dict(tickfont=dict(size=15)),
            showlegend=False,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    elif analyze:
        st.warning("Please paste a review first!")

# ════════════════════════════════════════════════════════
# PAGE 2 — BOX OFFICE DASHBOARD
# ════════════════════════════════════════════════════════
elif page == "📊 Box Office Dashboard":
    st.title("📊 Box Office Dashboard")
    st.markdown("*Interactive analytics across 1,678 films from TMDB*")
    st.markdown("---")

    df = load_data()

    # Stats row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Films", f"{len(df):,}")
    c2.metric("Avg Rating", f"{df['vote_average'].mean():.2f}")
    c3.metric("Genres", df["primary_genre"].nunique())
    c4.metric("Years Covered", f"{df['release_year'].min()}–{df['release_year'].max()}")

    st.markdown("###")

    # Chart 1 — Genre ratings
    genre_ratings = (
        df.groupby("primary_genre")["vote_average"]
        .agg(["mean","count"]).reset_index()
    )
    genre_ratings.columns = ["Genre","Avg Rating","Count"]
    genre_ratings = genre_ratings[genre_ratings["Count"] >= 10]
    genre_ratings = genre_ratings.sort_values("Avg Rating", ascending=False)

    fig1 = px.bar(
        genre_ratings, x="Genre", y="Avg Rating",
        color="Avg Rating",
        color_continuous_scale=[[0,"#8B1A1A"],[0.5,"#C9A84C"],[1,"#4CAF50"]],
        text="Avg Rating", title="Average Rating by Genre",
    )
    fig1.update_traces(
        texttemplate="<b>%{text:.2f}</b>", textposition="outside",
        textfont=dict(size=13, color="#F5F0E8"), marker_line_width=0,
    )
    fig1.update_layout(
        plot_bgcolor="#0A0A0F", paper_bgcolor="#141420",
        font=dict(color="#F5F0E8", size=14, family="Georgia"),
        title_font=dict(size=18, color="#C9A84C"),
        coloraxis_showscale=False, height=420,
        xaxis=dict(tickangle=-35, gridcolor="#1a1a25", tickfont=dict(size=12)),
        yaxis=dict(range=[0,9], gridcolor="#1a1a25"),
        margin=dict(t=50, b=80),
    )
    st.plotly_chart(fig1, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        # Chart 2 — Genre distribution
        genre_counts = df["primary_genre"].value_counts().reset_index()
        genre_counts.columns = ["Genre","Count"]
        fig2 = px.pie(
            genre_counts.head(10), values="Count", names="Genre",
            color_discrete_sequence=[
                "#C9A84C","#B52B2B","#5B4FE8","#4CAF50","#26C6DA",
                "#FF7043","#9C27B0","#F5F0E8","#607D8B","#FFC107"
            ],
            hole=0.4, title="Genre Distribution",
        )
        fig2.update_traces(textfont=dict(size=13, color="#F5F0E8"), textinfo="label+percent")
        fig2.update_layout(
            plot_bgcolor="#0A0A0F", paper_bgcolor="#141420",
            font=dict(color="#F5F0E8", size=13, family="Georgia"),
            title_font=dict(size=16, color="#C9A84C"),
            legend=dict(font=dict(size=12)),
            height=400, margin=dict(t=50, b=20),
        )
        st.plotly_chart(fig2, use_container_width=True)

    with col_right:
        # Chart 4 — Rating distribution
        fig4 = px.histogram(
            df[df["vote_average"] > 0],
            x="vote_average", nbins=20,
            color_discrete_sequence=["#C9A84C"],
            title="Rating Distribution",
            labels={"vote_average":"Rating"},
        )
        fig4.update_traces(marker_line_color="#0A0A0F", marker_line_width=2, opacity=0.9)
        fig4.update_layout(
            plot_bgcolor="#0A0A0F", paper_bgcolor="#141420",
            font=dict(color="#F5F0E8", size=13, family="Georgia"),
            title_font=dict(size=16, color="#C9A84C"),
            height=400, bargap=0.15, showlegend=False,
            xaxis=dict(title="Rating", gridcolor="#1a1a25", tickfont=dict(size=12)),
            yaxis=dict(title="Number of Films", gridcolor="#1a1a25"),
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Chart 3 — Rating per year
    yearly = (
        df[df["release_year"] >= 2000]
        .groupby("release_year")
        .agg(film_count=("id","count"), avg_rating=("vote_average","mean"))
        .reset_index()
    )
    yearly = yearly[yearly["film_count"] >= 5]
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=yearly["release_year"], y=yearly["avg_rating"].round(2),
        mode="lines+markers", name="Avg Rating",
        line=dict(color="#C9A84C", width=3),
        marker=dict(size=9, color="#C9A84C", line=dict(color="#F5F0E8", width=2)),
        fill="tozeroy", fillcolor="rgba(201,168,76,0.08)",
    ))
    fig3.update_layout(
        title="Average Film Rating Per Year (2000–Present)",
        plot_bgcolor="#0A0A0F", paper_bgcolor="#141420",
        font=dict(color="#F5F0E8", size=14, family="Georgia"),
        title_font=dict(size=18, color="#C9A84C"),
        height=380,
        xaxis=dict(title="Year", gridcolor="#1a1a25", tickmode="linear", dtick=2, tickfont=dict(size=12)),
        yaxis=dict(title="Avg Rating", gridcolor="#1a1a25", range=[5,8]),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Chart 5 — Popularity vs Rating
    df_f = df[
        (df["vote_average"] > 0) &
        (df["popularity"] < df["popularity"].quantile(0.95))
    ].copy()
    fig5 = px.scatter(
        df_f, x="vote_average", y="popularity",
        color="primary_genre",
        title="Popularity vs Rating by Genre",
        labels={"vote_average":"Rating","popularity":"Popularity"},
        hover_data=["title"], opacity=0.75,
    )
    fig5.update_traces(marker=dict(size=8, line=dict(width=0.5, color="#0A0A0F")))
    fig5.update_layout(
        plot_bgcolor="#0A0A0F", paper_bgcolor="#141420",
        font=dict(color="#F5F0E8", size=13, family="Georgia"),
        title_font=dict(size=18, color="#C9A84C"),
        height=450,
        xaxis=dict(title="Audience Rating", gridcolor="#1a1a25", tickfont=dict(size=12)),
        yaxis=dict(title="Popularity Score", gridcolor="#1a1a25"),
        legend=dict(font=dict(size=12), bgcolor="rgba(20,20,32,0.8)"),
        margin=dict(t=50, b=40),
    )
    st.plotly_chart(fig5, use_container_width=True)

# ════════════════════════════════════════════════════════
# PAGE 3 — FILM EXPLORER
# ════════════════════════════════════════════════════════
elif page == "🔍 Film Explorer":
    st.title("🔍 Film Explorer")
    st.markdown("*Search and explore films from the dataset*")
    st.markdown("---")

    df = load_data()

    col1, col2 = st.columns(2)
    with col1:
        genre_filter = st.selectbox(
            "Filter by genre",
            ["All"] + sorted(df["primary_genre"].unique().tolist())
        )
    with col2:
        year_range = st.slider(
            "Release year range",
            int(df["release_year"].min()),
            int(df["release_year"].max()),
            (2010, 2024)
        )

    search = st.text_input("🔍 Search film title", placeholder="e.g. Inception, Avatar...")

    filtered = df.copy()
    if genre_filter != "All":
        filtered = filtered[filtered["primary_genre"] == genre_filter]
    filtered = filtered[
        (filtered["release_year"] >= year_range[0]) &
        (filtered["release_year"] <= year_range[1])
    ]
    if search:
        filtered = filtered[filtered["title"].str.contains(search, case=False, na=False)]

    filtered = filtered.sort_values("vote_average", ascending=False)

    st.markdown(f"**{len(filtered)} films found**")
    st.markdown("###")

    display_cols = ["title","primary_genre","release_year","vote_average","vote_count","popularity"]
    st.dataframe(
        filtered[display_cols].rename(columns={
            "title":"Title", "primary_genre":"Genre",
            "release_year":"Year", "vote_average":"Rating",
            "vote_count":"Votes", "popularity":"Popularity"
        }).reset_index(drop=True),
        use_container_width=True,
        height=500,
    )