# =====================================================================
# Section 0: Imports, Logging & Performance Styling
# =====================================================================
import datetime
import logging
import re
import feedparser
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("CatalystPulse")

st.set_page_config(page_title="Catalyst Pulse Pro", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.2rem; padding-bottom: 2rem; padding-left: 1.8rem; padding-right: 1.8rem; }
        header[data-testid="stHeader"] { display: none !important; }
        footer { visibility: hidden; }
        .score-pill-up { background-color: #d4edda; color: #155724; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
        .score-pill-down { background-color: #f8d7da; color: #721c24; padding: 4px 8px; border-radius: 6px; font-weight: bold; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Section 1: Universe Configuration & Regex Classifier
# =====================================================================
WATCHLIST_STOCKS = [
    {"ticker": "BEL.NS", "name": "Bharat Electronics", "category": "Defense / PSU"},
    {"ticker": "HAL.NS", "name": "Hindustan Aeronautics", "category": "Defense / PSU"},
    {"ticker": "LT.NS", "name": "Larsen & Toubro", "category": "Infrastructure"},
    {"ticker": "TATASTEEL.NS", "name": "Tata Steel", "category": "Metals"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "category": "Energy & Retail"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "category": "IT Services"},
    {"ticker": "INFY.NS", "name": "Infosys Ltd", "category": "IT Services"},
    {"ticker": "SBIN.NS", "name": "State Bank of India", "category": "PSU Banking"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "category": "Banking"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank", "category": "Banking"},
    {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel", "category": "Telecom"},
    {"ticker": "ITC.NS", "name": "ITC Ltd", "category": "FMCG"},
    {"ticker": "BSE.NS", "name": "BSE Limited", "category": "Capital Markets"},
    {"ticker": "DIXON.NS", "name": "Dixon Technologies", "category": "EMS / Electronics"},
    {"ticker": "TRENT.NS", "name": "Trent Ltd", "category": "Retail"},
]

CATALYST_RULES = {
    "Demerger / Value Unlock": {
        "regex": re.compile(r"(?:demerger|spin-?off|scheme of arrangement|value unlocking)", re.IGNORECASE),
        "impact_mult": 1.35,
        "fade_bias": False
    },
    "Order Win / Mega Contract": {
        "regex": re.compile(r"(?:bagged|awarded|receives?|secures?)\s+(?:an?\s+)?(?:order|contract|project|loi)", re.IGNORECASE),
        "impact_mult": 1.15,
        "fade_bias": False
    },
    "Capex / Expansion": {
        "regex": re.compile(r"(?:commercial production|capacity expansion|capex|new facility|new plant)", re.IGNORECASE),
        "impact_mult": 1.05,
        "fade_bias": False
    },
    "Dividend Payout": {
        "regex": re.compile(r"(?:interim dividend|final dividend|special dividend|dividend of rs)", re.IGNORECASE),
        "impact_mult": 0.85,
        "fade_bias": True
    },
    "Bonus / Stock Split": {
        "regex": re.compile(r"(?:sub-division|subdivision|split of face value|bonus issue|bonus shares)", re.IGNORECASE),
        "impact_mult": 0.70,
        "fade_bias": True
    },
    "Regulatory / Governance Risk": {
        "regex": re.compile(r"(?:resignation|auditor|search|seizure|enforcement|show cause|inspection)", re.IGNORECASE),
        "impact_mult": -1.50,
        "fade_bias": True
    }
}

RSS_FEEDS = {
    "BSE Announcements": "https://beta.bseindia.com/rss-feed.html",
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol Top News": "https://www.moneycontrol.com/rss/MCtopnews.xml"
}

# =====================================================================
# Section 2: Data Ingestion & Announcement Extraction
# =====================================================================
@st.cache_data(ttl=600)
def fetch_corporate_catalysts():
    news_items, matched_map = [], {}
    known_syms = [x["ticker"].replace(".NS", "") for x in WATCHLIST_STOCKS]

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:25]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                full_text = f"{title} {summary}"

                detected_catalyst, impact, is_fade = "General Market", 1.0, False
                for cat_name, meta in CATALYST_RULES.items():
                    if meta["regex"].search(full_text):
                        detected_catalyst = cat_name
                        impact = meta["impact_mult"]
                        is_fade = meta["fade_bias"]
                        break

                matched = [sym for sym in known_syms if re.search(rf"\b{sym}\b", full_text, re.IGNORECASE)]

                item = {
                    "source": source_name,
                    "title": title,
                    "summary": summary,
                    "link": entry.get("link", "#"),
                    "published": entry.get("published", str(datetime.date.today())),
                    "catalyst": detected_catalyst,
                    "impact": impact,
                    "fade_bias": is_fade,
                    "matched": matched
                }
                news_items.append(item)
                for sym in matched:
                    if sym not in matched_map:
                        matched_map[sym] = item
        except Exception as e:
            logger.warning(f"Error parsing feed {source_name}: {e}")

    return news_items, matched_map

@st.cache_data(ttl=300)
def load_market_data(tickers):
    download_list = list(tickers) + ["^NSEI", "^INDIAVIX"]
    try:
        return yf.download(download_list, period="1y", interval="1d", group_by="ticker", auto_adjust=True)
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        return pd.DataFrame()

# =====================================================================
# Section 3: Drift & Predictive Scoring Engine
# =====================================================================
def compute_predictive_catalyst_metrics(raw_data, news_map):
    if raw_data.empty:
        return pd.DataFrame()

    results = []
    for asset in WATCHLIST_STOCKS:
        sym = asset["ticker"]
        clean_sym = sym.replace(".NS", "")
        if sym not in raw_data.columns.levels[0]:
            continue

        df = raw_data[sym].dropna()
        if len(df) < 50:
            continue

        c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
        cmp = float(c.iloc[-1])
        d200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float(c.mean())
        dist_200 = ((cmp - d200) / d200) * 100.0

        # 5-Day Pre-Event Run-up
        price_5d_ago = float(c.iloc[-6]) if len(c) >= 6 else float(c.iloc[0])
        runup_5d = ((cmp - price_5d_ago) / price_5d_ago) * 100.0

        # Day-1 Volume vs 20-Day Average
        v_latest = float(v.iloc[-1])
        v_20d = float(v.rolling(20).mean().iloc[-1])
        vol_surge_ratio = round(v_latest / v_20d, 2) if v_20d > 0 else 1.0

        # Daily Candle Closing Position (0% = bottom low, 100% = top high)
        day_range = float(h.iloc[-1] - l.iloc[-1])
        close_pos_pct = round(((cmp - float(l.iloc[-1])) / day_range) * 100.0, 1) if day_range > 0 else 50.0

        # Active Catalyst Matching
        cat_info = news_map.get(clean_sym, None)
        if cat_info:
            cat_name = cat_info["catalyst"]
            impact_mult = cat_info["impact"]
            has_news = True
        else:
            cat_name = "⚡ Baseline Technicals"
            impact_mult = 1.0
            has_news = False

        # --- 5-Pillar Predictive Score Formula ---
        # 1. Expectation Score: Penalizes over-extended run-ups
        score_expectation = np.clip(100.0 - (runup_5d * 10.0), -50.0, 50.0)

        # 2. Volume Conviction Score: Rewards high volume with strong close
        score_volume = (vol_surge_ratio * (close_pos_pct - 50.0) * 0.4)

        # 3. Structural Trend: Bonus for CMP > 200 DMA
        score_trend = 15.0 if dist_200 > 0 else -15.0

        # Raw Score
        raw_score = (score_expectation * 0.35) + (score_volume * 0.45) + (score_trend * 0.20)
        final_catalyst_score = round(raw_score * impact_mult, 1)

        # Expected 1-2 Week Horizon Signal
        if final_catalyst_score >= 20.0 and close_pos_pct >= 55.0 and runup_5d <= 4.0:
            outlook = "🟢 Bullish Drift Expected (1-2W)"
            confidence = "High" if vol_surge_ratio >= 1.8 else "Moderate"
        elif final_catalyst_score <= -15.0 or (runup_5d >= 7.5 and close_pos_pct <= 45.0):
            outlook = "🔴 Distribution / Fade Expected (1-2W)"
            confidence = "High" if runup_5d >= 7.5 else "Moderate"
        else:
            outlook = "🟡 Neutral Consolidation"
            confidence = "Low"

        results.append({
            "Ticker": clean_sym,
            "Name": asset["name"],
            "Category": asset["category"],
            "CMP (₹)": round(cmp, 2),
            "Pre-Event RunUp 5D %": round(runup_5d, 2),
            "Vol Surge Ratio": vol_surge_ratio,
            "Close in Range %": close_pos_pct,
            "Dist 200DMA %": round(dist_200, 2),
            "Active Catalyst": cat_name,
            "Catalyst Score": final_catalyst_score,
            "1-2W Outlook": outlook,
            "Confidence": confidence,
            "Has_News": has_news
        })

    return pd.DataFrame(results)

# =====================================================================
# Section 4: Sidebar Controls & Navigation
# =====================================================================
st.sidebar.title("⚡ Catalyst Pulse Pro")
st.sidebar.caption("Event-Driven Quantitative Alpha Engine")

raw_market_data = load_market_data([x["ticker"] for x in WATCHLIST_STOCKS])
news_items_list, matched_news_map = fetch_corporate_catalysts()
catalyst_df = compute_predictive_catalyst_metrics(raw_market_data, matched_news_map)

active_view = st.sidebar.radio(
    "Modules",
    [
        "🎯 1-2 Week Predictive Screener",
        "🔬 Stock Catalyst Profiler",
        "📰 Exchange Disclosures & News Feed",
        "📖 Strategy & Scoring Handbook"
    ]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Active Universe: {len(WATCHLIST_STOCKS)} Stocks")
st.sidebar.caption(f"Captured Disclosures: {len(news_items_list)} Items")

# =====================================================================
# Section 5: Dynamic UI Views
# =====================================================================

# VIEW 1: Top 2-5 Screener
if active_view == "🎯 1-2 Week Predictive Screener":
    st.subheader("🎯 1 to 2-Week Dynamic Catalyst Screener")
    st.markdown("Identifies actionable candidates based on unpriced positive surprises versus post-spike distribution fades.")

    if not catalyst_df.empty:
        up_candidates = catalyst_df[catalyst_df["1-2W Outlook"].str.contains("Bullish")].sort_values(by="Catalyst Score", ascending=False).head(5)
        down_candidates = catalyst_df[catalyst_df["1-2W Outlook"].str.contains("Distribution")].sort_values(by="Catalyst Score", ascending=True).head(5)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🟢 Top Expected to Move UP (1-2 Weeks)")
            if not up_candidates.empty:
                st.dataframe(
                    up_candidates[["Ticker", "CMP (₹)", "Catalyst Score", "Pre-Event RunUp 5D %", "Vol Surge Ratio", "Active Catalyst", "Confidence"]].style.format({
                        "CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "Pre-Event RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x"
                    }),
                    use_container_width=True
                )
            else:
                st.info("No stocks currently meet all bullish drift parameters (RunUp <= 4%, High Volume & Clean Close).")

        with c2:
            st.markdown("#### 🔴 Top Expected to Move DOWN / Fade (1-2 Weeks)")
            if not down_candidates.empty:
                st.dataframe(
                    down_candidates[["Ticker", "CMP (₹)", "Catalyst Score", "Pre-Event RunUp 5D %", "Vol Surge Ratio", "Active Catalyst", "Confidence"]].style.format({
                        "CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "Pre-Event RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x"
                    }),
                    use_container_width=True
                )
            else:
                st.info("No high-conviction fade or distribution traps detected.")

        st.markdown("---")
        st.markdown("#### 🌐 Complete Evaluated Catalyst Universe")
        st.dataframe(
            catalyst_df[["Ticker", "Name", "CMP (₹)", "Catalyst Score", "1-2W Outlook", "Pre-Event RunUp 5D %", "Vol Surge Ratio", "Close in Range %", "Dist 200DMA %", "Active Catalyst"]].sort_values(by="Catalyst Score", ascending=False),
            use_container_width=True
        )

# VIEW 2: Stock Profiler
elif active_view == "🔬 Stock Catalyst Profiler":
    st.subheader("🔬 Single-Stock Reaction & Gap Profiler")
    selected_sym = st.selectbox("Select Stock to Inspect:", [x["ticker"].replace(".NS", "") for x in WATCHLIST_STOCKS])
    stock_row = catalyst_df[catalyst_df["Ticker"] == selected_sym].iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("CMP", f"₹{stock_row['CMP (₹)']:.2f}")
    k2.metric("Catalyst Score", f"{stock_row['Catalyst Score']:+.1f}")
    k3.metric("5D Pre-RunUp", f"{stock_row['Pre-Event RunUp 5D %']:+.2f}%")
    k4.metric("Volume Surge", f"{stock_row['Vol Surge Ratio']}x")

    st.markdown(f"**Current 1-2W Forecast:** {stock_row['1-2W Outlook']} (Confidence: {stock_row['Confidence']})")
    st.markdown(f"**Linked Announcement:** {stock_row['Active Catalyst']}")

# VIEW 3: Live News Feed
elif active_view == "📰 Exchange Disclosures & News Feed":
    st.subheader("📰 Authentic Exchange Filings & Media Stream")
    for item in news_items_list[:20]:
        with st.expander(f"[{item['catalyst']}] {item['title']}"):
            st.write(item["summary"] if item["summary"] else "Full disclosure available via exchange link.")
            if item["matched"]:
                st.markdown(f"**Tagged Watchlist Tickers:** `{', '.join(item['matched'])}`")
            st.caption(f"Source: {item['source']} | Published: {item['published']}")
            st.markdown(f"[View Document / Link]({item['link']})")

# VIEW 4: Strategy Handbook
elif active_view == "📖 Strategy & Scoring Handbook":
    st.subheader("📖 Event-Driven Alpha Methodology")
    st.markdown("""
    * **Pre-Event Run-up Trap:** When a stock gains over 7.5% over the 5 trading sessions prior to an announcement, smart money often distributes into the announcement.
    * **Volume-Delivery Ratio:** Authentic institutional accumulation requires volume to exceed 2.0x the 20-day average with a close in the upper 35% of the daily candle range.
    * **Closing Range Percentage:** Measures whether buyers held control into the close ($100\% = \text{Close at Day High}$, $0\% = \text{Close at Day Low}$).
    """)
