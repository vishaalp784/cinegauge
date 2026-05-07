import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import ast
import os

# ── STYLE ─────────────────────────────────────────────────
BG      = "#0A0A0F"
CARD    = "#141420"
GOLD    = "#C9A84C"
CREAM   = "#F5F0E8"
SILVER  = "#9A9A9A"

def style(fig, title):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=22, color=GOLD, family="Georgia"),
            x=0.03
        ),
        font=dict(color=CREAM, family="Georgia", size=15),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        legend=dict(
            font=dict(size=14, color=CREAM),
            bgcolor="rgba(20,20,32,0.8)",
            bordercolor=GOLD,
            borderwidth=1,
        ),
        margin=dict(t=70, l=60, r=40, b=60),
    )
    fig.update_xaxes(
        tickfont=dict(size=13, color=CREAM),
        title_font=dict(size=15, color=SILVER),
        gridcolor="#1a1a25",
        linecolor="#2a2a35",
    )
    fig.update_yaxes(
        tickfont=dict(size=13, color=CREAM),
        title_font=dict(size=15, color=SILVER),
        gridcolor="#1a1a25",
        linecolor="#2a2a35",
    )
    return fig

# ── LOAD DATA ─────────────────────────────────────────────
print("📊 Loading movie data...")
df = pd.read_csv("data/movies.csv")

GENRES = {
    28:"Action", 12:"Adventure", 16:"Animation", 35:"Comedy",
    80:"Crime", 99:"Documentary", 18:"Drama", 10751:"Family",
    14:"Fantasy", 36:"History", 27:"Horror", 10402:"Music",
    9648:"Mystery", 10749:"Romance", 878:"Sci-Fi",
    53:"Thriller", 10752:"War", 37:"Western"
}

def parse_genres(genre_ids):
    try:
        ids = ast.literal_eval(str(genre_ids))
        return [GENRES.get(i, "Other") for i in ids[:2]]
    except:
        return ["Other"]

df["genres"]        = df["genre_ids"].apply(parse_genres)
df["primary_genre"] = df["genres"].apply(lambda x: x[0] if x else "Other")
df["release_year"]  = pd.to_numeric(df["release_year"], errors="coerce")
df = df.dropna(subset=["release_year"])
df["release_year"]  = df["release_year"].astype(int)
print(f"✓ {len(df)} movies loaded")

os.makedirs("charts", exist_ok=True)

# ── CHART 1 — Average rating by genre ─────────────────────
print("🎨 Chart 1 — Genre ratings...")
genre_ratings = (
    df.groupby("primary_genre")["vote_average"]
    .agg(["mean","count"])
    .reset_index()
)
genre_ratings.columns = ["Genre","Avg Rating","Film Count"]
genre_ratings = genre_ratings[genre_ratings["Film Count"] >= 10]
genre_ratings = genre_ratings.sort_values("Avg Rating", ascending=False)

fig1 = px.bar(
    genre_ratings, x="Genre", y="Avg Rating",
    color="Avg Rating",
    color_continuous_scale=[
        [0.0, "#8B1A1A"],
        [0.5, "#C9A84C"],
        [1.0, "#4CAF50"]
    ],
    text="Avg Rating",
)
fig1.update_traces(
    texttemplate="<b>%{text:.2f}</b>",
    textposition="outside",
    textfont=dict(size=14, color=CREAM),
    marker_line_width=0,
)
fig1.update_layout(coloraxis_showscale=False)
fig1 = style(fig1, "🎬 Average Audience Rating by Genre")
fig1.update_xaxes(tickangle=-35)
fig1.update_yaxes(range=[0, 9])
fig1.write_html("charts/chart1_genre_ratings.html")
print("   ✓ Done")

# ── CHART 2 — Genre distribution pie ──────────────────────
print("🎨 Chart 2 — Genre distribution...")
genre_counts = df["primary_genre"].value_counts().reset_index()
genre_counts.columns = ["Genre","Count"]

fig2 = px.pie(
    genre_counts.head(10),
    values="Count", names="Genre",
    color_discrete_sequence=[
        "#C9A84C","#B52B2B","#5B4FE8","#4CAF50","#26C6DA",
        "#FF7043","#9C27B0","#F5F0E8","#607D8B","#FFC107"
    ],
    hole=0.4,
)
fig2.update_traces(
    textfont=dict(size=15, color=CREAM),
    textinfo="label+percent",
)
fig2 = style(fig2, "🎭 Film Distribution by Genre")
fig2.write_html("charts/chart2_genre_distribution.html")
print("   ✓ Done")

# ── CHART 3 — Average rating per year ─────────────────────
print("🎨 Chart 3 — Rating per year...")
yearly = (
    df[df["release_year"] >= 2000]
    .groupby("release_year")
    .agg(film_count=("id","count"), avg_rating=("vote_average","mean"))
    .reset_index()
)
yearly = yearly[yearly["film_count"] >= 5]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(
    x=yearly["release_year"],
    y=yearly["avg_rating"].round(2),
    mode="lines+markers",
    name="Avg Rating",
    line=dict(color=GOLD, width=3),
    marker=dict(
        size=9, color=GOLD,
        line=dict(color=CREAM, width=2)
    ),
    fill="tozeroy",
    fillcolor="rgba(201,168,76,0.08)",
))
fig3 = style(fig3, "⭐ Average Film Rating Per Year (2000–Present)")
fig3.update_xaxes(title="Year", tickmode="linear", dtick=2)
fig3.update_yaxes(title="Average Rating", range=[5, 8])
fig3.write_html("charts/chart3_avg_rating_per_year.html")
print("   ✓ Done")

# ── CHART 4 — Rating distribution ─────────────────────────
print("🎨 Chart 4 — Rating distribution...")
fig4 = px.histogram(
    df[df["vote_average"] > 0],
    x="vote_average", nbins=20,
    labels={"vote_average":"Rating", "count":"Number of Films"},
    color_discrete_sequence=[GOLD],
)
fig4.update_traces(
    marker_line_color=BG,
    marker_line_width=2,
    opacity=0.9,
)
fig4 = style(fig4, "⭐ Film Rating Distribution")
fig4.update_xaxes(title="Rating")
fig4.update_yaxes(title="Number of Films")
fig4.update_layout(bargap=0.15, showlegend=False)
fig4.write_html("charts/chart4_rating_distribution.html")
print("   ✓ Done")

# ── CHART 5 — Popularity vs Rating ────────────────────────
print("🎨 Chart 5 — Popularity vs Rating...")
df_f = df[
    (df["vote_average"] > 0) &
    (df["popularity"] < df["popularity"].quantile(0.95))
].copy()

fig5 = px.scatter(
    df_f,
    x="vote_average", y="popularity",
    color="primary_genre",
    labels={"vote_average":"Rating", "popularity":"Popularity Score"},
    hover_data=["title"],
    opacity=0.75,
)
fig5.update_traces(marker=dict(size=8, line=dict(width=0.5, color=BG)))
fig5 = style(fig5, "🌟 Popularity vs Rating by Genre")
fig5.update_xaxes(title="Audience Rating")
fig5.update_yaxes(title="Popularity Score")
fig5.write_html("charts/chart5_popularity_vs_rating.html")
print("   ✓ Done")

print("\n✅ All 5 charts rebuilt — bigger text, bolder labels!")
print("🎬 Open any .html file in your browser to see them!")