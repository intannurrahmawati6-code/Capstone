"""
============================================================
  EDA Dashboard - Fake vs Genuine Review Detection
  Dataset: master_dataset_interim.csv (Tokopedia Reviews)
  Jalankan di Google Colab dengan cara di bawah ini:
  
  !pip install streamlit wordcloud matplotlib seaborn plotly pyngrok
  !wget -q -O master_dataset_interim.csv <upload_link_csv>
  
  Kemudian jalankan:
  !streamlit run eda_dashboard.py &
  from pyngrok import ngrok
  public_url = ngrok.connect(8501)
  print(public_url)
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Review EDA Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 18px 20px;
        border-left: 4px solid #4F8EF7;
        margin-bottom: 10px;
    }
    .metric-card h3 { margin: 0; font-size: 28px; color: #1a1a2e; }
    .metric-card p  { margin: 4px 0 0; font-size: 13px; color: #666; }
    .section-title  { font-size: 20px; font-weight: 700; margin: 24px 0 8px; color: #1a1a2e; }
    .fake-badge  { background:#FFE0E0; color:#C0392B; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:600; }
    .genuine-badge { background:#D5F5E3; color:#1E8449; padding:3px 10px; border-radius:20px; font-size:13px; font-weight:600; }
    hr { border: none; border-top: 1px solid #eee; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# LOAD & CACHE DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # ── Feature Engineering ──────────────────
    df["word_count"]   = df["message"].str.split().str.len().fillna(0).astype(int)
    df["char_count"]   = df["message"].str.len().fillna(0).astype(int)
    df["has_exclaim"]  = df["message"].str.contains("!", na=False)
    df["has_emoji_kw"] = df["message"].str.contains(
        r"(haha|wkwk|:D|:v|mantap|oke|ok|kece)", na=False, case=False
    )

    # ── Fake Label Heuristic ─────────────────
    # Fake = sangat pendek (≤3 kata) + bintang 5 + anonymous
    df["fake_label"] = (
        (df["word_count"] <= 3) &
        (df["rating"] == 5) &
        (df["is_anonymous"] == True)
    ).map({True: "Fake", False: "Genuine"})

    # ── Sentiment (from rating) ───────────────
    df["sentiment"] = df["rating"].map({
        1: "Negatif", 2: "Negatif",
        3: "Netral",
        4: "Positif", 5: "Positif"
    })

    # ── Parse timestamp ke urutan relatif ────
    order_map = {
        "1 hari lalu": 1, "2 hari lalu": 2, "3 hari lalu": 3,
        "1 minggu lalu": 7, "2 minggu lalu": 14, "3 minggu lalu": 21,
        "1 bulan lalu": 30, "2 bulan lalu": 60, "3 bulan lalu": 90,
        "4 bulan lalu": 120, "5 bulan lalu": 150, "6 bulan lalu": 180,
        "7 bulan lalu": 210, "8 bulan lalu": 240, "9 bulan lalu": 270,
        "10 bulan lalu": 300, "11 bulan lalu": 330, "1 tahun lalu": 365,
        "Lebih dari 1 tahun lalu": 400
    }
    df["days_ago"] = df["timestamp"].map(order_map).fillna(200)

    return df


@st.cache_data
def get_word_freq(texts, stopwords, top_n=20):
    all_words = []
    for t in texts:
        words = re.findall(r"[a-z]+", str(t).lower())
        all_words.extend([w for w in words if w not in stopwords and len(w) > 2])
    return Counter(all_words).most_common(top_n)


# Stopwords bahasa Indonesia
STOPWORDS = {
    "yang","dan","di","ke","dari","ini","itu","dengan","untuk","adalah","ada",
    "pada","tidak","nya","saya","sudah","belum","sangat","banget","juga","tapi",
    "kalo","kalau","bisa","karena","jadi","karna","pas","ya","si","se","ga",
    "gak","nggak","bgt","sy","lg","dg","tp","dr","utk","dgn","sdh","blm","krn",
    "lagi","aja","sih","lebih","atau","pun","harga","produk","barang","pesanan",
    "pengiriman","beli","sesuai","packing","oke","ok","nya","yg","ke","di","an",
    "lah","deh","jg","udah","udh","emang","memang","mau","nih","kan","saja",
    "sama","itu","satu","dua","tiga","per","the","and","for","its","was","are",
    "with","that","this","have","from","they","will","but","not","all","been",
    "has","had","would","could","should","may","might","shall","than","then",
    "when","where","which","who","how","what","why","were","there","their",
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/80/star.png", width=60)
    st.title("Review EDA")
    st.markdown("---")

    uploaded = st.file_uploader(
        "📂 Upload CSV Dataset",
        type=["csv"],
        help="Format: master_dataset_interim.csv"
    )

    st.markdown("---")
    st.subheader("⚙️ Filter Data")

    # Placeholder — diisi setelah data dimuat
    filter_product = st.multiselect("Produk", options=[], key="prod_filter")
    filter_rating  = st.multiselect("Rating", options=[1,2,3,4,5], default=[1,2,3,4,5])
    filter_label   = st.radio("Label", ["Semua", "Fake", "Genuine"], horizontal=True)

    st.markdown("---")
    top_n = st.slider("Top N Kata", 10, 30, 20)

    st.markdown("---")
    st.caption("📊 Dashboard EDA · Fake Review Detection")

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
DEFAULT_PATH = "C:\Users\ASUS\Downloads\CAPSTONE\Data\master_dataset_interim.csv"

if uploaded:
    df_raw = load_data(uploaded)
else:
    try:
        df_raw = load_data(DEFAULT_PATH)
    except FileNotFoundError:
        st.error(
            "⚠️ File tidak ditemukan. "
            "Upload file CSV melalui sidebar, atau pastikan "
            f"**{DEFAULT_PATH}** ada di direktori yang sama."
        )
        st.stop()

# Update sidebar filter setelah data tersedia
products = df_raw["product_name"].unique().tolist()
with st.sidebar:
    filter_product = st.multiselect(
        "Produk", options=products, default=products, key="prod_filter2"
    )

# ── Apply Filters ───────────────────────────
df = df_raw[
    df_raw["product_name"].isin(filter_product) &
    df_raw["rating"].isin(filter_rating)
].copy()

if filter_label != "Semua":
    df = df[df["fake_label"] == filter_label]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🔍 EDA Dashboard — Fake vs Genuine Review")
st.caption(
    "Analisis eksploratori dataset ulasan produk Tokopedia · "
    f"Menampilkan **{len(df):,}** dari **{len(df_raw):,}** ulasan"
)
st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 1 — METRIC CARDS
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">📌 Ringkasan Dataset</p>', unsafe_allow_html=True)

n_fake    = (df["fake_label"] == "Fake").sum()
n_genuine = (df["fake_label"] == "Genuine").sum()
avg_rating = df["rating"].mean()
avg_words  = df["word_count"].mean()
pct_anon   = df["is_anonymous"].mean() * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Ulasan",  f"{len(df):,}")
c2.metric("🚨 Fake Review", f"{n_fake:,}",    f"{n_fake/len(df)*100:.1f}%")
c3.metric("✅ Genuine",     f"{n_genuine:,}", f"{n_genuine/len(df)*100:.1f}%")
c4.metric("⭐ Avg Rating",  f"{avg_rating:.2f}")
c5.metric("📝 Avg Kata",    f"{avg_words:.1f}")

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 2 — DISTRIBUSI FAKE VS GENUINE
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">🏷️ Distribusi Fake vs Genuine Review</p>', unsafe_allow_html=True)

col_a, col_b, col_c = st.columns([1, 1, 1])

with col_a:
    # Pie chart
    label_counts = df["fake_label"].value_counts()
    fig_pie = go.Figure(go.Pie(
        labels=label_counts.index,
        values=label_counts.values,
        marker_colors=["#E74C3C", "#2ECC71"],
        hole=0.45,
        textinfo="label+percent",
        textfont_size=14,
    ))
    fig_pie.update_layout(
        title="Proporsi Label",
        showlegend=False,
        margin=dict(t=40, b=10, l=10, r=10),
        height=300,
    )
    st.plotly_chart(fig_pie, use_container_width=True)

with col_b:
    # Bar: Fake vs Genuine per Rating
    cross = df.groupby(["rating", "fake_label"]).size().reset_index(name="count")
    fig_bar = px.bar(
        cross, x="rating", y="count", color="fake_label",
        color_discrete_map={"Fake": "#E74C3C", "Genuine": "#2ECC71"},
        barmode="group",
        title="Distribusi per Rating",
        labels={"rating": "Rating", "count": "Jumlah", "fake_label": "Label"},
    )
    fig_bar.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10),
                          legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_bar, use_container_width=True)

with col_c:
    # Bar: Fake vs Genuine per Produk
    prod_cross = df.groupby(["product_name", "fake_label"]).size().reset_index(name="count")
    prod_cross["short_name"] = prod_cross["product_name"].str[:25] + "…"
    fig_prod = px.bar(
        prod_cross, y="short_name", x="count", color="fake_label",
        color_discrete_map={"Fake":"#E74C3C","Genuine":"#2ECC71"},
        barmode="stack", orientation="h",
        title="Distribusi per Produk",
        labels={"short_name":"Produk","count":"Jumlah","fake_label":"Label"},
    )
    fig_prod.update_layout(height=300, margin=dict(t=40, b=10, l=10, r=10),
                           legend=dict(orientation="h", y=-0.3),
                           yaxis_title="")
    st.plotly_chart(fig_prod, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 3 — DISTRIBUSI RATING & SENTIMEN
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">⭐ Distribusi Rating & Sentimen</p>', unsafe_allow_html=True)

col_d, col_e = st.columns(2)

with col_d:
    rating_counts = df["rating"].value_counts().sort_index()
    fig_rating = go.Figure(go.Bar(
        x=rating_counts.index,
        y=rating_counts.values,
        marker_color=["#E74C3C","#E67E22","#F1C40F","#3498DB","#2ECC71"],
        text=rating_counts.values,
        textposition="outside",
    ))
    fig_rating.update_layout(
        title="Histogram Rating",
        xaxis_title="Rating", yaxis_title="Jumlah Ulasan",
        height=320, margin=dict(t=40,b=10,l=10,r=10),
        xaxis=dict(tickmode="linear", tick0=1, dtick=1),
    )
    st.plotly_chart(fig_rating, use_container_width=True)

with col_e:
    sent_counts = df["sentiment"].value_counts()
    fig_sent = go.Figure(go.Bar(
        x=sent_counts.index,
        y=sent_counts.values,
        marker_color=["#E74C3C","#F1C40F","#2ECC71"],
        text=sent_counts.values,
        textposition="outside",
    ))
    fig_sent.update_layout(
        title="Distribusi Sentimen (dari Rating)",
        xaxis_title="Sentimen", yaxis_title="Jumlah",
        height=320, margin=dict(t=40,b=10,l=10,r=10),
    )
    st.plotly_chart(fig_sent, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 4 — DISTRIBUSI PANJANG ULASAN
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">📏 Panjang Ulasan (Jumlah Kata)</p>', unsafe_allow_html=True)

col_f, col_g = st.columns(2)

with col_f:
    fig_hist = px.histogram(
        df, x="word_count", color="fake_label",
        color_discrete_map={"Fake":"#E74C3C","Genuine":"#2ECC71"},
        nbins=40, barmode="overlay",
        title="Distribusi Jumlah Kata per Ulasan",
        labels={"word_count":"Jumlah Kata","count":"Frekuensi","fake_label":"Label"},
        opacity=0.75,
    )
    fig_hist.add_vline(x=df["word_count"].mean(), line_dash="dash",
                       line_color="navy", annotation_text=f"Mean={df['word_count'].mean():.1f}")
    fig_hist.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10),
                            legend=dict(orientation="h", y=-0.25))
    st.plotly_chart(fig_hist, use_container_width=True)

with col_g:
    fig_box = px.box(
        df, x="fake_label", y="word_count", color="fake_label",
        color_discrete_map={"Fake":"#E74C3C","Genuine":"#2ECC71"},
        title="Boxplot Panjang Kata (Fake vs Genuine)",
        labels={"fake_label":"Label","word_count":"Jumlah Kata"},
        points="outliers",
    )
    fig_box.update_layout(height=320, margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)

# Word count stats table
st.markdown("**Statistik Deskriptif Panjang Kata per Label:**")
wc_stats = df.groupby("fake_label")["word_count"].describe().round(2)
st.dataframe(wc_stats, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 5 — POLA KATA (TOP WORDS)
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">🔤 Pola Kata Paling Sering Muncul</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🟢 Ulasan Positif", "🔴 Ulasan Negatif",
    "🚨 Ulasan Fake", "✅ Ulasan Genuine"
])

def word_bar_chart(texts, title, color, top_n):
    freq = get_word_freq(texts, STOPWORDS, top_n)
    if not freq:
        st.warning("Tidak cukup data.")
        return
    words, counts = zip(*freq)
    fig = go.Figure(go.Bar(
        x=list(counts)[::-1], y=list(words)[::-1],
        orientation="h",
        marker_color=color,
        text=[str(c) for c in counts][::-1],
        textposition="outside",
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Frekuensi", yaxis_title="",
        height=max(350, top_n * 22),
        margin=dict(t=40, b=10, l=10, r=10),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab1:
    pos_texts = df[df["rating"] >= 4]["message"]
    word_bar_chart(pos_texts, f"Top {top_n} Kata — Ulasan Positif (Rating ≥4)", "#2ECC71", top_n)

with tab2:
    neg_texts = df[df["rating"] <= 2]["message"]
    word_bar_chart(neg_texts, f"Top {top_n} Kata — Ulasan Negatif (Rating ≤2)", "#E74C3C", top_n)

with tab3:
    fake_texts = df[df["fake_label"] == "Fake"]["message"]
    word_bar_chart(fake_texts, f"Top {top_n} Kata — Ulasan FAKE", "#E67E22", top_n)

with tab4:
    genuine_texts = df[df["fake_label"] == "Genuine"]["message"]
    word_bar_chart(genuine_texts, f"Top {top_n} Kata — Ulasan GENUINE", "#3498DB", top_n)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 6 — WORD CLOUD (matplotlib)
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">☁️ Word Cloud Ulasan</p>', unsafe_allow_html=True)

try:
    from wordcloud import WordCloud

    def make_wc(texts, colormap, title):
        combined = " ".join(
            w for t in texts
            for w in re.findall(r"[a-z]+", str(t).lower())
            if w not in STOPWORDS and len(w) > 2
        )
        if not combined.strip():
            st.warning(f"Tidak cukup teks untuk {title}")
            return
        wc = WordCloud(
            width=600, height=300, background_color="white",
            colormap=colormap, max_words=120,
            collocations=False
        ).generate(combined)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, fontweight="bold", pad=10)
        st.pyplot(fig)
        plt.close()

    wc1, wc2 = st.columns(2)
    with wc1:
        make_wc(df[df["rating"] >= 4]["message"], "Greens", "Word Cloud — Ulasan Positif")
    with wc2:
        make_wc(df[df["rating"] <= 2]["message"], "Reds",  "Word Cloud — Ulasan Negatif")

    wc3, wc4 = st.columns(2)
    with wc3:
        make_wc(df[df["fake_label"] == "Fake"]["message"],    "Oranges", "Word Cloud — Fake Review")
    with wc4:
        make_wc(df[df["fake_label"] == "Genuine"]["message"], "Blues",   "Word Cloud — Genuine Review")

except ImportError:
    st.info(
        "📦 WordCloud belum terinstall. "
        "Jalankan: `!pip install wordcloud` lalu restart kernel."
    )

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 7 — ANALISIS FITUR LAIN
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">🔎 Analisis Fitur Tambahan</p>', unsafe_allow_html=True)

col_h, col_i, col_j = st.columns(3)

with col_h:
    # Proporsi Anonymous
    anon = df.groupby(["fake_label", "is_anonymous"]).size().reset_index(name="count")
    anon["anon_label"] = anon["is_anonymous"].map({True:"Anonim", False:"Teridentifikasi"})
    fig_anon = px.bar(
        anon, x="fake_label", y="count", color="anon_label",
        color_discrete_map={"Anonim":"#9B59B6","Teridentifikasi":"#1ABC9C"},
        barmode="group", title="Status Anonim",
        labels={"fake_label":"Label","count":"Jumlah","anon_label":"Tipe User"},
    )
    fig_anon.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10),
                            legend=dict(orientation="h",y=-0.3))
    st.plotly_chart(fig_anon, use_container_width=True)

with col_i:
    # Has Image per label
    img = df.groupby(["fake_label","has_image"]).size().reset_index(name="count")
    img["img_label"] = img["has_image"].map({True:"Ada Foto",False:"Tanpa Foto"})
    fig_img = px.bar(
        img, x="fake_label", y="count", color="img_label",
        color_discrete_map={"Ada Foto":"#3498DB","Tanpa Foto":"#BDC3C7"},
        barmode="group", title="Ulasan dengan Foto",
        labels={"fake_label":"Label","count":"Jumlah","img_label":"Foto"},
    )
    fig_img.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10),
                           legend=dict(orientation="h",y=-0.3))
    st.plotly_chart(fig_img, use_container_width=True)

with col_j:
    # Total likes
    fig_likes = px.box(
        df, x="fake_label", y="total_likes", color="fake_label",
        color_discrete_map={"Fake":"#E74C3C","Genuine":"#2ECC71"},
        title="Distribusi Total Likes",
        labels={"fake_label":"Label","total_likes":"Total Likes"},
        points="all",
    )
    fig_likes.update_layout(height=300, margin=dict(t=40,b=10,l=10,r=10), showlegend=False)
    st.plotly_chart(fig_likes, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 8 — TREND ULASAN (TIMESTAMP)
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">📅 Tren Ulasan Berdasarkan Waktu</p>', unsafe_allow_html=True)

ts_count = df.groupby(["timestamp","fake_label"]).size().reset_index(name="count")
ts_order = df.groupby("timestamp")["days_ago"].first().reset_index()
ts_count  = ts_count.merge(ts_order, on="timestamp")
ts_count  = ts_count.sort_values("days_ago")

fig_trend = px.line(
    ts_count, x="timestamp", y="count", color="fake_label",
    color_discrete_map={"Fake":"#E74C3C","Genuine":"#2ECC71"},
    markers=True,
    title="Jumlah Ulasan per Periode (Fake vs Genuine)",
    labels={"timestamp":"Periode","count":"Jumlah","fake_label":"Label"},
)
fig_trend.update_layout(
    height=380, margin=dict(t=40,b=80,l=10,r=10),
    xaxis_tickangle=-40,
    legend=dict(orientation="h", y=-0.35),
)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 9 — RAW DATA EXPLORER
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">🗂️ Jelajahi Data Mentah</p>', unsafe_allow_html=True)

view_cols = ["product_name","rating","fake_label","sentiment","word_count",
             "is_anonymous","has_image","total_likes","message"]
available  = [c for c in view_cols if c in df.columns]

search_kw  = st.text_input("🔎 Cari kata dalam ulasan:", placeholder="Contoh: kecewa, bagus, …")
n_display  = st.slider("Tampilkan N baris:", 10, 200, 50)

df_show = df[available].copy()
if search_kw.strip():
    df_show = df_show[df_show["message"].str.contains(search_kw, case=False, na=False)]

st.dataframe(df_show.head(n_display), use_container_width=True, height=380)

st.markdown("---")

# ─────────────────────────────────────────────
# SECTION 10 — INSIGHT RINGKAS
# ─────────────────────────────────────────────
st.markdown('<p class="section-title">💡 Insight Utama</p>', unsafe_allow_html=True)

pct_fake = n_fake / len(df) * 100
pct_anon = df["is_anonymous"].mean() * 100
avg_wc_fake    = df[df["fake_label"]=="Fake"]["word_count"].mean()
avg_wc_genuine = df[df["fake_label"]=="Genuine"]["word_count"].mean()
pct_5star = (df["rating"] == 5).mean() * 100

insights = [
    f"📌 **{pct_fake:.1f}%** ulasan terdeteksi sebagai **Fake** berdasarkan heuristik "
    f"(≤3 kata + bintang 5 + anonim).",
    f"👤 **{pct_anon:.1f}%** reviewer menggunakan mode **anonim**, "
    f"menyulitkan identifikasi akun palsu.",
    f"📝 Rata-rata panjang ulasan Fake: **{avg_wc_fake:.1f} kata**, "
    f"Genuine: **{avg_wc_genuine:.1f} kata** — perbedaan signifikan.",
    f"⭐ **{pct_5star:.1f}%** ulasan memberikan rating **bintang 5** — "
    f"sangat dominan dan perlu diwaspadai.",
    "🔤 Kata dominan ulasan positif: *bagus, cocok, semoga, alhamdulillah* — "
    "mayoritas singkat dan generik.",
    "🔤 Kata dominan ulasan negatif: *kecewa, habis, cas, warna, malah* — "
    "lebih spesifik dan detail.",
]

for ins in insights:
    st.markdown(f"- {ins}")

st.markdown("---")
st.caption(
    "© 2024 Review EDA Dashboard · "
    "Dataset: Tokopedia Product Reviews · "
    "Heuristik fake review: word_count ≤ 3 + rating = 5 + is_anonymous = True"
)
