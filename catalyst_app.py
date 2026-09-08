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
logger = logging.getLogger("CatalystPulsePro")

st.set_page_config(page_title="Catalyst Pulse Pro (NIFTY 100)", page_icon="⚡", layout="wide")

st.markdown(
    """
    <style>
        .block-container { padding-top: 1.0rem !important; padding-bottom: 2rem !important; padding-left: 1.5rem !important; padding-right: 1.5rem !important; }
        header[data-testid="stHeader"] { display: none !important; }
        footer { visibility: hidden; }
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            background-color: #f1f3f5; padding: 5px; border-radius: 12px; display: flex; flex-wrap: wrap; gap: 5px; border: 1px solid #dee2e6;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            background-color: transparent; border-radius: 8px; padding: 5px 12px !important; font-weight: 600 !important; font-size: 0.88rem !important; color: #495057; cursor: pointer;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
            background-color: #1E88E5 !important; color: #ffffff !important; box-shadow: 0 2px 6px rgba(30, 136, 229, 0.35);
        }
        .vix-pulse-banner {
            background-color: #f8f9fa; border-left: 4px solid #1E88E5; padding: 8px 14px; border-radius: 4px; font-size: 0.88rem; margin-bottom: 0.8rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =====================================================================
# Section 1: Curated NIFTY 100 Universe & Reg 30 Regex Classifier
# =====================================================================
NIFTY_100_TICKERS = [
    {"ticker": "ABB.NS", "name": "ABB India", "category": "Capital Goods"},
    {"ticker": "ADANIENSOL.NS", "name": "Adani Energy Solutions", "category": "Power"},
    {"ticker": "ADANIENT.NS", "name": "Adani Enterprises", "category": "Conglomerate"},
    {"ticker": "ADANIGREEN.NS", "name": "Adani Green Energy", "category": "Renewable"},
    {"ticker": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "category": "Infrastructure"},
    {"ticker": "ADANIPOWER.NS", "name": "Adani Power", "category": "Power"},
    {"ticker": "AMBUJACEM.NS", "name": "Ambuja Cements", "category": "Cement"},
    {"ticker": "APOLLOHOSP.NS", "name": "Apollo Hospitals", "category": "Healthcare"},
    {"ticker": "ASIANPAINT.NS", "name": "Asian Paints", "category": "Consumer"},
    {"ticker": "AXISBANK.NS", "name": "Axis Bank", "category": "Banking"},
    {"ticker": "BAJAJ-AUTO.NS", "name": "Bajaj Auto", "category": "Automobile"},
    {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance", "category": "NBFC"},
    {"ticker": "BAJAJFINSV.NS", "name": "Bajaj Finserv", "category": "Financial Services"},
    {"ticker": "BAJAJHLDNG.NS", "name": "Bajaj Holdings", "category": "Finance"},
    {"ticker": "BANKBARODA.NS", "name": "Bank of Baroda", "category": "PSU Banking"},
    {"ticker": "BEL.NS", "name": "Bharat Electronics", "category": "Defense / PSU"},
    {"ticker": "BHEL.NS", "name": "Bharat Heavy Electricals", "category": "Capital Goods"},
    {"ticker": "BPCL.NS", "name": "BPCL", "category": "Oil & Gas"},
    {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel", "category": "Telecom"},
    {"ticker": "BOSCHLTD.NS", "name": "Bosch Ltd", "category": "Auto Ancillary"},
    {"ticker": "BRITANNIA.NS", "name": "Britannia Industries", "category": "FMCG"},
    {"ticker": "CANBK.NS", "name": "Canara Bank", "category": "PSU Banking"},
    {"ticker": "CHOLAFIN.NS", "name": "Cholamandalam Inv", "category": "NBFC"},
    {"ticker": "CIPLA.NS", "name": "Cipla", "category": "Pharma"},
    {"ticker": "COALINDIA.NS", "name": "Coal India", "category": "Mining / PSU"},
    {"ticker": "COLPAL.NS", "name": "Colgate Palmolive", "category": "FMCG"},
    {"ticker": "DLF.NS", "name": "DLF Ltd", "category": "Real Estate"},
    {"ticker": "DABUR.NS", "name": "Dabur India", "category": "FMCG"},
    {"ticker": "DIVISLAB.NS", "name": "Divi's Laboratories", "category": "Pharma"},
    {"ticker": "DIXON.NS", "name": "Dixon Technologies", "category": "EMS / Electronics"},
    {"ticker": "DRREDDY.NS", "name": "Dr. Reddy's Labs", "category": "Pharma"},
    {"ticker": "EICHERMOT.NS", "name": "Eicher Motors", "category": "Automobile"},
    {"ticker": "GAIL.NS", "name": "GAIL (India)", "category": "Gas / PSU"},
    {"ticker": "GODREJCP.NS", "name": "Godrej Consumer Products", "category": "FMCG"},
    {"ticker": "GRASIM.NS", "name": "Grasim Industries", "category": "Materials"},
    {"ticker": "HCLTECH.NS", "name": "HCL Technologies", "category": "IT Services"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "category": "Banking"},
    {"ticker": "HDFCLIFE.NS", "name": "HDFC Life Insurance", "category": "Insurance"},
    {"ticker": "HAVELLS.NS", "name": "Havells India", "category": "Consumer Electricals"},
    {"ticker": "HEROMOTOCO.NS", "name": "Hero MotoCorp", "category": "Automobile"},
    {"ticker": "HINDALCO.NS", "name": "Hindalco Industries", "category": "Metals"},
    {"ticker": "HAL.NS", "name": "Hindustan Aeronautics", "category": "Defense / PSU"},
    {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever", "category": "FMCG"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank", "category": "Banking"},
    {"ticker": "ICICIGI.NS", "name": "ICICI Lombard General Ins", "category": "Insurance"},
    {"ticker": "ICICIPRULI.NS", "name": "ICICI Prudential Life", "category": "Insurance"},
    {"ticker": "ITC.NS", "name": "ITC Ltd", "category": "FMCG"},
    {"ticker": "INDHOTEL.NS", "name": "Indian Hotels Company", "category": "Hospitality"},
    {"ticker": "IOC.NS", "name": "Indian Oil Corporation", "category": "Oil & Gas"},
    {"ticker": "IRCTC.NS", "name": "IRCTC", "category": "Railways / PSU"},
    {"ticker": "IRFC.NS", "name": "Indian Railway Finance", "category": "NBFC / PSU"},
    {"ticker": "INDUSINDBK.NS", "name": "IndusInd Bank", "category": "Banking"},
    {"ticker": "NAUKRI.NS", "name": "Info Edge (Naukri)", "category": "Internet"},
    {"ticker": "INFY.NS", "name": "Infosys Ltd", "category": "IT Services"},
    {"ticker": "INDIGO.NS", "name": "InterGlobe Aviation (IndiGo)", "category": "Aviation"},
    {"ticker": "JSWSTEEL.NS", "name": "JSW Steel", "category": "Metals"},
    {"ticker": "JINDALSTEL.NS", "name": "Jindal Steel & Power", "category": "Metals"},
    {"ticker": "JIOFIN.NS", "name": "Jio Financial Services", "category": "Financial Services"},
    {"ticker": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "category": "Banking"},
    {"ticker": "LTIM.NS", "name": "LTIMindtree", "category": "IT Services"},
    {"ticker": "LT.NS", "name": "Larsen & Toubro", "category": "Infrastructure"},
    {"ticker": "LUPIN.NS", "name": "Lupin Ltd", "category": "Pharma"},
    {"ticker": "M&M.NS", "name": "Mahindra & Mahindra", "category": "Automobile"},
    {"ticker": "MARICO.NS", "name": "Marico Ltd", "category": "FMCG"},
    {"ticker": "MARUTI.NS", "name": "Maruti Suzuki", "category": "Automobile"},
    {"ticker": "MAXHEALTH.NS", "name": "Max Healthcare", "category": "Healthcare"},
    {"ticker": "NTPC.NS", "name": "NTPC Ltd", "category": "Power / PSU"},
    {"ticker": "NESTLEIND.NS", "name": "Nestle India", "category": "FMCG"},
    {"ticker": "ONGC.NS", "name": "ONGC", "category": "Oil & Gas / PSU"},
    {"ticker": "PIDILITIND.NS", "name": "Pidilite Industries", "category": "Chemicals"},
    {"ticker": "PFC.NS", "name": "Power Finance Corp", "category": "NBFC / PSU"},
    {"ticker": "POWERGRID.NS", "name": "Power Grid Corp", "category": "Power / PSU"},
    {"ticker": "PNB.NS", "name": "Punjab National Bank", "category": "PSU Banking"},
    {"ticker": "RECLTD.NS", "name": "REC Ltd", "category": "NBFC / PSU"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "category": "Energy & Retail"},
    {"ticker": "SBICARD.NS", "name": "SBI Cards", "category": "Financial Services"},
    {"ticker": "SBILIFE.NS", "name": "SBI Life Insurance", "category": "Insurance"},
    {"ticker": "SRF.NS", "name": "SRF Ltd", "category": "Chemicals"},
    {"ticker": "MOTHERSON.NS", "name": "Samvardhana Motherson", "category": "Auto Ancillary"},
    {"ticker": "SHREECEM.NS", "name": "Shree Cement", "category": "Cement"},
    {"ticker": "SHRIRAMFIN.NS", "name": "Shriram Finance", "category": "NBFC"},
    {"ticker": "SIEMENS.NS", "name": "Siemens India", "category": "Capital Goods"},
    {"ticker": "SBIN.NS", "name": "State Bank of India", "category": "PSU Banking"},
    {"ticker": "SUNPHARMA.NS", "name": "Sun Pharma", "category": "Pharma"},
    {"ticker": "TVSMOTOR.NS", "name": "TVS Motor Company", "category": "Automobile"},
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "category": "IT Services"},
    {"ticker": "TATACONSUM.NS", "name": "Tata Consumer Products", "category": "FMCG"},
    {"ticker": "TATAMOTORS.NS", "name": "Tata Motors", "category": "Automobile"},
    {"ticker": "TATAPOWER.NS", "name": "Tata Power", "category": "Power"},
    {"ticker": "TATASTEEL.NS", "name": "Tata Steel", "category": "Metals"},
    {"ticker": "TECHM.NS", "name": "Tech Mahindra", "category": "IT Services"},
    {"ticker": "TITAN.NS", "name": "Titan Company", "category": "Consumer Discretionary"},
    {"ticker": "TORNTPOWER.NS", "name": "Torrent Power", "category": "Power"},
    {"ticker": "TRENT.NS", "name": "Trent Ltd", "category": "Retail"},
    {"ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement", "category": "Cement"},
    {"ticker": "UNITDSPR.NS", "name": "United Spirits", "category": "Beverages"},
    {"ticker": "VBL.NS", "name": "Varun Beverages", "category": "Beverages"},
    {"ticker": "VEDL.NS", "name": "Vedanta Ltd", "category": "Metals"},
    {"ticker": "WIPRO.NS", "name": "Wipro", "category": "IT Services"},
    {"ticker": "ZOMATO.NS", "name": "Zomato", "category": "Internet / Platform"},
    {"ticker": "ZYDUSLIFE.NS", "name": "Zydus Lifesciences", "category": "Pharma"},
]

CATALYST_RULES = {
    "Demerger / Value Unlock": {
        "regex": re.compile(r"(?:demerger|spin-?off|scheme of arrangement|value unlocking)", re.IGNORECASE),
        "impact_mult": 1.35, "base_p1w": 72, "base_p2w": 78
    },
    "Order Win / Mega Contract": {
        "regex": re.compile(r"(?:bagged|awarded|receives?|secures?)\s+(?:an?\s+)?(?:order|contract|project|loi)", re.IGNORECASE),
        "impact_mult": 1.15, "base_p1w": 65, "base_p2w": 68
    },
    "Capex / Expansion": {
        "regex": re.compile(r"(?:commercial production|capacity expansion|capex|new facility|new plant)", re.IGNORECASE),
        "impact_mult": 1.05, "base_p1w": 60, "base_p2w": 64
    },
    "Dividend Payout": {
        "regex": re.compile(r"(?:interim dividend|final dividend|special dividend|dividend of rs)", re.IGNORECASE),
        "impact_mult": 0.85, "base_p1w": 48, "base_p2w": 45
    },
    "Bonus / Stock Split": {
        "regex": re.compile(r"(?:sub-division|subdivision|split of face value|bonus issue|bonus shares)", re.IGNORECASE),
        "impact_mult": 0.70, "base_p1w": 42, "base_p2w": 38
    },
    "Regulatory / Governance Risk": {
        "regex": re.compile(r"(?:resignation|auditor|search|seizure|enforcement|show cause|inspection)", re.IGNORECASE),
        "impact_mult": -1.50, "base_p1w": 25, "base_p2w": 20
    }
}

RSS_FEEDS = {
    "BSE Corporate Announcements": "https://beta.bseindia.com/rss-feed.html",
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol Top News": "https://www.moneycontrol.com/rss/MCtopnews.xml"
}

# =====================================================================
# Section 2: Data Ingestion & Technical Math
# =====================================================================
@st.cache_data(ttl=600)
def fetch_corporate_catalysts(active_universe):
    news_items, matched_map = [], {}
    known_syms = [x["ticker"].replace(".NS", "") for x in active_universe]

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:35]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                full_text = f"{title} {summary}"

                detected_catalyst, impact, p1, p2 = "General Market", 1.0, 50, 50
                for cat_name, meta in CATALYST_RULES.items():
                    if meta["regex"].search(full_text):
                        detected_catalyst = cat_name
                        impact = meta["impact_mult"]
                        p1 = meta["base_p1w"]
                        p2 = meta["base_p2w"]
                        break

                matched = [sym for sym in known_syms if re.search(rf"\b{sym}\b", full_text, re.IGNORECASE)]

                item = {
                    "source": source_name, "title": title, "summary": summary,
                    "link": entry.get("link", "#"), "published": entry.get("published", str(datetime.date.today())),
                    "catalyst": detected_catalyst, "impact": impact, "p1w": p1, "p2w": p2, "matched": matched
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
        return yf.download(download_list, period="1y", interval="1d", group_by="ticker", auto_adjust=True, threads=True)
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        return pd.DataFrame()

# =====================================================================
# Section 3: Predictive Engine & Probability Calculation
# =====================================================================
def compute_predictive_catalyst_metrics(raw_data, news_map, universe):
    if raw_data.empty:
        return pd.DataFrame()

    results = []
    for asset in universe:
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

        # Pre-Event Run-up (5D)
        price_5d_ago = float(c.iloc[-6]) if len(c) >= 6 else float(c.iloc[0])
        runup_5d = ((cmp - price_5d_ago) / price_5d_ago) * 100.0

        # Volume Surge Ratio
        v_latest = float(v.iloc[-1])
        v_20d = float(v.rolling(20).mean().iloc[-1])
        vol_surge_ratio = round(v_latest / v_20d, 2) if v_20d > 0 else 1.0

        # Intraday Close in Range % (100% = Day High, 0% = Day Low)
        day_range = float(h.iloc[-1] - l.iloc[-1])
        close_pos_pct = round(((cmp - float(l.iloc[-1])) / day_range) * 100.0, 1) if day_range > 0 else 50.0

        # News Matching
        cat_info = news_map.get(clean_sym, None)
        if cat_info:
            cat_name = cat_info["catalyst"]
            impact_mult = cat_info["impact"]
            base_p1 = cat_info["p1w"]
            base_p2 = cat_info["p2w"]
        else:
            cat_name = "⚡ Technical Baseline"
            impact_mult = 1.0
            base_p1 = 50
            base_p2 = 50

        # Scoring Pillars
        score_runup = np.clip(100.0 - (runup_5d * 10.0), -50.0, 50.0)
        score_volume = (vol_surge_ratio * (close_pos_pct - 50.0) * 0.4)
        score_trend = 15.0 if dist_200 > 0 else -15.0

        raw_score = (score_runup * 0.35) + (score_volume * 0.45) + (score_trend * 0.20)
        final_catalyst_score = round(raw_score * impact_mult, 1)

        # Dynamic Probability Adjustment based on actual execution confirmation
        prob_adjustment = 0.0
        if runup_5d <= 2.5:
            prob_adjustment += 8.0
        elif runup_5d >= 7.5:
            prob_adjustment -= 18.0

        if vol_surge_ratio >= 2.0 and close_pos_pct >= 65.0:
            prob_adjustment += 10.0
        elif close_pos_pct <= 35.0:
            prob_adjustment -= 12.0

        if dist_200 > 0:
            prob_adjustment += 4.0
        else:
            prob_adjustment -= 8.0

        prob_1w = int(np.clip(base_p1 + prob_adjustment, 10, 92))
        prob_2w = int(np.clip(base_p2 + (prob_adjustment * 1.1), 10, 94))

        if final_catalyst_score >= 18.0 and prob_1w >= 65:
            outlook = "🟢 Bullish Drift (1-2W)"
        elif final_catalyst_score <= -15.0 or (runup_5d >= 7.5 and close_pos_pct <= 40.0) or prob_1w <= 40:
            outlook = "🔴 Distribution / Fade (1-2W)"
        else:
            outlook = "🟡 Neutral Consolidation"

        results.append({
            "Ticker": clean_sym,
            "Name": asset["name"],
            "Category": asset["category"],
            "CMP (₹)": round(cmp, 2),
            "Catalyst Score": final_catalyst_score,
            "P(1W) Drift %": prob_1w,
            "P(2W) Drift %": prob_2w,
            "Pre-RunUp 5D %": round(runup_5d, 2),
            "Vol Surge Ratio": vol_surge_ratio,
            "Close in Range %": close_pos_pct,
            "Dist 200DMA %": round(dist_200, 2),
            "Active Catalyst": cat_name,
            "1-2W Outlook": outlook,
        })

    return pd.DataFrame(results)

# =====================================================================
# Section 4: Sidebar Controls & Navigation
# =====================================================================
st.sidebar.title("⚡ Catalyst Pulse Pro")
st.sidebar.caption("Institutional News-Drift & Expectation Engine")

universe_choice = st.sidebar.selectbox("Active Stock Universe:", ["Curated NIFTY 100 (Full)", "Nifty Top 30 Large-Caps", "High-Beta Midcaps"], index=0)

if universe_choice == "Curated NIFTY 100 (Full)":
    ACTIVE_UNIVERSE = NIFTY_100_TICKERS
elif universe_choice == "Nifty Top 30 Large-Caps":
    ACTIVE_UNIVERSE = NIFTY_100_TICKERS[:30]
else:
    ACTIVE_UNIVERSE = NIFTY_100_TICKERS[30:70]

raw_market_data = load_market_data([x["ticker"] for x in ACTIVE_UNIVERSE])
news_items_list, matched_news_map = fetch_corporate_catalysts(ACTIVE_UNIVERSE)
catalyst_df = compute_predictive_catalyst_metrics(raw_market_data, matched_news_map, ACTIVE_UNIVERSE)

# Macro India VIX
curr_vix = 15.0
if "^INDIAVIX" in raw_market_data.columns.levels[0]:
    v_close = raw_market_data["^INDIAVIX"]["Close"].dropna()
    if not v_close.empty:
        curr_vix = float(v_close.iloc[-1])

st.sidebar.markdown("---")
st.sidebar.metric("India VIX Pulse", f"{curr_vix:.1f}", "Normal" if curr_vix < 20 else "High Volatility")
st.sidebar.caption(f"Evaluated Universe: {len(ACTIVE_UNIVERSE)} Stocks")

nav_choice = st.radio(
    "Navigation",
    [
        "🎯 Dynamic 1-2 Week Screener",
        "🔬 Single-Stock Deep Dive",
        "📰 Exchange Disclosures & Media Feed",
        "📖 Quantitative Strategy Handbook"
    ],
    label_visibility="collapsed",
    horizontal=True
)
st.markdown("<div style='margin-bottom: 0.6rem;'></div>", unsafe_allow_html=True)

# Helper function for relative Top 3 / Bottom 3 color-coding
def apply_top3_bot3_styling(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    
    # Highest is favorable (Top 3 Green, Bottom 3 Red)
    higher_is_better = ["Catalyst Score", "P(1W) Drift %", "P(2W) Drift %", "Vol Surge Ratio", "Close in Range %", "Dist 200DMA %"]
    for col in higher_is_better:
        if col in df.columns:
            t3 = df[col].nlargest(3).index
            b3 = df[col].nsmallest(3).index
            styles.loc[t3, col] = "background-color: #d4edda; color: #155724; font-weight: bold;"
            styles.loc[b3, col] = "background-color: #f8d7da; color: #721c24; font-weight: bold;"

    # Pre-RunUp: Lowest is favorable (Bottom 3 Lowest = Green, Top 3 Highest = Red)
    if "Pre-RunUp 5D %" in df.columns:
        best_runup = df["Pre-RunUp 5D %"].nsmallest(3).index
        worst_runup = df["Pre-RunUp 5D %"].nlargest(3).index
        styles.loc[best_runup, "Pre-RunUp 5D %"] = "background-color: #d4edda; color: #155724; font-weight: bold;"
        styles.loc[worst_runup, "Pre-RunUp 5D %"] = "background-color: #f8d7da; color: #721c24; font-weight: bold;"

    return styles

# =====================================================================
# Section 5: Dynamic Views
# =====================================================================

# VIEW 1: 1-2 Week Screener
if nav_choice == "🎯 Dynamic 1-2 Week Screener":
    st.subheader(f"🎯 Dynamic 1-2 Week Catalyst Screener ({universe_choice})")
    st.caption("Filters unpriced corporate catalysts vs overextended post-spike distribution fades.")

    if not catalyst_df.empty:
        up_candidates = catalyst_df[catalyst_df["1-2W Outlook"].str.contains("Bullish")].sort_values(by="Catalyst Score", ascending=False).head(5)
        down_candidates = catalyst_df[catalyst_df["1-2W Outlook"].str.contains("Distribution")].sort_values(by="Catalyst Score", ascending=True).head(5)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🟢 Top Expected to Move UP (1-2 Weeks)")
            if not up_candidates.empty:
                cols_u = ["Ticker", "CMP (₹)", "Catalyst Score", "P(1W) Drift %", "P(2W) Drift %", "Pre-RunUp 5D %", "Vol Surge Ratio", "Active Catalyst"]
                st.dataframe(
                    up_candidates[cols_u].style.apply(apply_top3_bot3_styling, axis=None).format({
                        "CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%",
                        "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x"
                    }),
                    use_container_width=True
                )
            else:
                st.info("No candidates qualify with unpriced conditions (RunUp <= 2.5%, Volume >= 2.0x, Close >= 65%).")

        with c2:
            st.markdown("#### 🔴 Top Expected to Move DOWN / Fade (1-2 Weeks)")
            if not down_candidates.empty:
                cols_d = ["Ticker", "CMP (₹)", "Catalyst Score", "P(1W) Drift %", "P(2W) Drift %", "Pre-RunUp 5D %", "Vol Surge Ratio", "Active Catalyst"]
                st.dataframe(
                    down_candidates[cols_d].style.apply(apply_top3_bot3_styling, axis=None).format({
                        "CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%",
                        "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x"
                    }),
                    use_container_width=True
                )
            else:
                st.info("No high-conviction distribution traps detected.")

        st.markdown("---")
        st.markdown(f"#### 🌐 Full Evaluated Universe ({len(catalyst_df)} Equities)")
        display_all = catalyst_df.sort_values(by="Catalyst Score", ascending=False).reset_index(drop=True)
        st.dataframe(
            display_all.style.apply(apply_top3_bot3_styling, axis=None).format({
                "CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%",
                "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x", "Close in Range %": "{:.1f}%",
                "Dist 200DMA %": "{:+.2f}%"
            }),
            use_container_width=True, height=520
        )

# VIEW 2: Stock Deep Dive
elif nav_choice == "🔬 Single-Stock Deep Dive":
    st.subheader("🔬 Single-Stock Expectation & Drift Profiler")
    all_syms = sorted([x["ticker"].replace(".NS", "") for x in ACTIVE_UNIVERSE])
    chosen_stock = st.selectbox("Select Stock to Inspect:", all_syms)

    stock_row = catalyst_df[catalyst_df["Ticker"] == chosen_stock].iloc[0]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("CMP", f"₹{stock_row['CMP (₹)']:.2f}")
    k2.metric("Catalyst Score", f"{stock_row['Catalyst Score']:+.1f}")
    k3.metric("P(1W) Drift", f"{stock_row['P(1W) Drift %']}%")
    k4.metric("P(2W) Drift", f"{stock_row['P(2W) Drift %']}%")
    k5.metric("Pre-RunUp 5D", f"{stock_row['Pre-RunUp 5D %']:+.2f}%")

    st.markdown(f"**Forecast Signal:** `{stock_row['1-2W Outlook']}`")
    st.markdown(f"**Linked Corporate Action:** `{stock_row['Active Catalyst']}`")
    st.markdown(f"**Intraday Candle Conviction:** Closed at **{stock_row['Close in Range %']}%** of the total daily range on **{stock_row['Vol Surge Ratio']}x** average volume.")

# VIEW 3: Live Feed
elif nav_choice == "📰 Exchange Disclosures & Media Feed":
    st.subheader("📰 Authentic Exchange Disclosures & Regulatory Stream")
    for item in news_items_list[:25]:
        with st.expander(f"[{item['catalyst']}] {item['title']}"):
            st.write(item["summary"] if item["summary"] else "Official disclosure notification via exchange stream.")
            if item["matched"]:
                st.markdown(f"**Associated Tickers:** `{', '.join(item['matched'])}`")
            st.caption(f"Source: {item['source']} | Published: {item['published']}")
            st.markdown(f"[Official Filing Link]({item['link']})")

# VIEW 4: Handbook
elif nav_choice == "📖 Quantitative Strategy Handbook":
    st.subheader("📖 Quantitative Strategy Handbook & Indicator Playbook")
    st.markdown("""
    This engine isolates **unpriced expectation divergence** from **'Sell the News' liquidation traps**.
    """)

    st.markdown("---")
    h1, h2 = st.columns(2)

    with h1:
        st.markdown("""
        ### 🔹 1. Pre-Event Run-Up (5D Lookback)
        * **Formula:** $\\frac{\\text{CMP} - P_{t-5}}{P_{t-5}} \\times 100$
        * **Meaning:** Measures how much the stock has already rallied into the announcement.
        * **Rule:** If $\\text{Run-up} > +7.5\\%$, institutions frequently dump into the news. If $\\text{Run-up} \\le +2.0\\%$, the move is unpriced and safe to enter.

        ### 🔹 2. Volume Surge Ratio
        * **Formula:** $\\frac{\\text{Day Volume}}{\\text{20D Average Volume}}$
        * **Meaning:** Confirms institutional participation.
        * **Rule:** True accumulation requires $\\ge 2.0\\times$ volume. Low volume on good news indicates retail-only participation.

        ### 🔹 3. Close in Range %
        * **Formula:** $\\frac{\\text{CMP} - \\text{Low}}{\\text{High} - \\text{Low}} \\times 100$
        * **Meaning:** Detects rejection wicks. If a stock surges at the open but closes near its lows, this value drops below $35\\%$, confirming an institutional exit trap.
        """)

    with h2:
        st.markdown("""
        ### 🔹 4. Probability Metrics: P(1W) & P(2W)
        * **Meaning:** Empirical probability that the asset will generate positive cumulative abnormal returns (CAR) over 5 trading sessions (1 week) and 10 trading sessions (2 weeks).
        * **Rule:**
          - $\\ge 70\\%$: High conviction for swing continuation.
          - $\\le 40\\%$: High probability of multi-day decay.

        ### 🔹 5. Scoring Weight Distribution
        * **Expectation Factor (Run-Up):** $35\\%$ weight
        * **Volume & Candle Closure:** $45\\%$ weight
        * **200 DMA Structural Trend:** $20\\%$ weight
        * **Multiplied by Event Accretion Factor:**
          - Demerger: $+1.35\\times$
          - Order Win / Capex: $+1.15\\times$
          - Bonus / Split: $+0.70\\times$ (Decay bias)
          - Regulatory Risk: $-1.50\\times$ (Negative drift)
        """)
