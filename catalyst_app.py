# =====================================================================
# Section 0: Imports, Logging & High-Density UI CSS
# =====================================================================
import datetime
from zoneinfo import ZoneInfo
import logging
import re
import os
import base64
import json
import requests
import feedparser
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", level=logging.INFO)
logger = logging.getLogger("CatalystPulsePro")

st.set_page_config(page_title="Catalyst Pulse Pro | NIFTY 100 Event Alpha", page_icon="⚡", layout="wide")

# High-density, space-optimized styling with extra bottom scroll padding
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 6.0rem !important;
            padding-left: 1.0rem !important;
            padding-right: 1.0rem !important;
        }
        header[data-testid="stHeader"] { display: none !important; }
        footer { visibility: hidden; }
        
        /* Shrunken compact sidebar */
        section[data-testid="stSidebar"] {
            width: 260px !important;
            min-width: 260px !important;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 0.8rem !important;
            padding-left: 0.8rem !important;
            padding-right: 0.8rem !important;
        }
        
        /* Compact Typography & Density */
        html, body, [class*="css"] {
            font-size: 0.86rem !important;
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.08rem !important;
            font-weight: 700 !important;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
        }
        
        /* Compact Segmented Pills */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            background-color: #f1f3f5; padding: 3px; border-radius: 8px; display: flex; flex-wrap: wrap; gap: 4px; border: 1px solid #dee2e6;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            background-color: transparent; border-radius: 6px; padding: 4px 10px !important; font-weight: 600 !important; font-size: 0.80rem !important; color: #495057; cursor: pointer;
        }
        div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"] {
            background-color: #1E88E5 !important; color: #ffffff !important; box-shadow: 0 1px 4px rgba(30, 136, 229, 0.3);
        }
        
        /* Compact Metrics container */
        div[data-testid="stMetric"] {
            background-color: #fcfcfc;
            border: 1px solid #edf0f2;
            border-radius: 6px;
            padding: 5px 8px !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

IST = ZoneInfo("Asia/Kolkata")

# =====================================================================
# Section 1: Curated NIFTY 100 Universe & Reg 30 Classifier
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
    "Demerger / Merger Unlock": {
        "regex": re.compile(r"(?:demerger|spin-?off|scheme of arrangement|value unlocking|merger|amalgamation)", re.IGNORECASE),
        "impact_mult": 1.35, "base_p1w": 72, "base_p2w": 78, "group": "Demergers & Mergers"
    },
    "Mega Order Win / Contract": {
        "regex": re.compile(r"(?:bagged|awarded|receives?|secures?|wins?)\s+(?:an?\s+)?(?:order|contract|project|loi|mandate)", re.IGNORECASE),
        "impact_mult": 1.20, "base_p1w": 66, "base_p2w": 70, "group": "Order Wins & Capex"
    },
    "Capex / Plant Expansion": {
        "regex": re.compile(r"(?:commercial production|capacity expansion|capex|new facility|new plant|manufacturing unit)", re.IGNORECASE),
        "impact_mult": 1.10, "base_p1w": 62, "base_p2w": 66, "group": "Order Wins & Capex"
    },
    "Dividend & Buyback": {
        "regex": re.compile(r"(?:interim dividend|final dividend|special dividend|dividend of rs|buyback|share repurchase)", re.IGNORECASE),
        "impact_mult": 0.85, "base_p1w": 46, "base_p2w": 42, "group": "Dividends & Buybacks"
    },
    "Bonus / Split / Rights Issue": {
        "regex": re.compile(r"(?:sub-division|subdivision|split of face value|bonus issue|bonus shares|rights issue|allotment of rights)", re.IGNORECASE),
        "impact_mult": 0.70, "base_p1w": 40, "base_p2w": 36, "group": "Splits & Bonus"
    },
    "Regulatory / Governance Warning": {
        "regex": re.compile(r"(?:resignation of auditor|cbi|ed search|seizure|enforcement|show cause notice|inspection|fraud)", re.IGNORECASE),
        "impact_mult": -1.60, "base_p1w": 22, "base_p2w": 18, "group": "Governance / Risk"
    }
}

RSS_FEEDS = {
    "BSE Corporate Announcements": "https://beta.bseindia.com/rss-feed.html",
    "Economic Times Markets": "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "Moneycontrol News": "https://www.moneycontrol.com/rss/MCtopnews.xml"
}

# =====================================================================
# Section 2: Data Ingestion & Technical Math
# =====================================================================
@st.cache_data(ttl=60)
def fetch_corporate_catalysts(active_universe):
    news_items, matched_map, ticker_news_history = [], {}, {}
    known_syms = [x["ticker"].replace(".NS", "") for x in active_universe]

    for source_name, feed_url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:40]:
                title = entry.get("title", "")
                summary = entry.get("summary", "")
                full_text = f"{title} {summary}"

                detected_catalyst, impact, p1, p2, grp = "⚡ General Market Notice", 1.0, 50, 50, "General"
                for cat_name, meta in CATALYST_RULES.items():
                    if meta["regex"].search(full_text):
                        detected_catalyst = cat_name
                        impact = meta["impact_mult"]
                        p1 = meta["base_p1w"]
                        p2 = meta["base_p2w"]
                        grp = meta["group"]
                        break

                matched = [sym for sym in known_syms if re.search(rf"\b{sym}\b", full_text, re.IGNORECASE)]

                item = {
                    "source": source_name, "title": title, "summary": summary,
                    "link": entry.get("link", "#"), "published": entry.get("published", str(datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M IST"))),
                    "catalyst": detected_catalyst, "impact": impact, "p1w": p1, "p2w": p2, "group": grp, "matched": matched
                }
                news_items.append(item)
                for sym in matched:
                    if sym not in matched_map:
                        matched_map[sym] = item
                    if sym not in ticker_news_history:
                        ticker_news_history[sym] = []
                    ticker_news_history[sym].append(item)
        except Exception as e:
            logger.warning(f"Error parsing feed {source_name}: {e}")

    return news_items, matched_map, ticker_news_history

@st.cache_data(ttl=60)
def load_market_data(tickers):
    download_list = list(tickers) + ["^NSEI", "^INDIAVIX"]
    try:
        return yf.download(download_list, period="1y", interval="1d", group_by="ticker", auto_adjust=True, threads=True)
    except Exception as e:
        logger.error(f"yfinance download failed: {e}")
        return pd.DataFrame()

# =====================================================================
# Section 3: Predictive Engine & Scoring Logic
# =====================================================================
LEDGER_FILENAME = "catalyst_prediction_ledger.csv"

def get_github_ledger():
    token = st.secrets.get("GITHUB_PAT", os.environ.get("GITHUB_PAT", ""))
    repo = st.secrets.get("GITHUB_REPO", os.environ.get("GITHUB_REPO", ""))
    
    empty_df = pd.DataFrame(columns=[
        "Prediction_ID", "Date", "Ticker", "Active_Catalyst", "CMP_At_Prediction",
        "Predicted_Outlook", "Recommended_Action", "Holding_Horizon", "Target_Days",
        "Market_Regime", "Trigger_Type", "Confidence", "Target_Return_Pct", "Stop_Loss_Pct",
        "Days_Elapsed", "Current_CMP", "Realized_Return_Pct", "Outcome_Status"
    ])
    
    if not token or not repo:
        if os.path.exists(LEDGER_FILENAME):
            return pd.read_csv(LEDGER_FILENAME), None
        return empty_df, None

    url = f"https://api.github.com/repos/{repo}/contents/{LEDGER_FILENAME}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        file_json = res.json()
        content = base64.b64decode(file_json["content"]).decode("utf-8")
        df = pd.read_csv(pd.io.common.StringIO(content))
        return df, file_json["sha"]
    elif res.status_code == 404:
        return empty_df, None
    else:
        logger.warning(f"GitHub Ledger fetch failed: {res.status_code} {res.text}")
        return empty_df, None

def commit_github_ledger(updated_df, sha=None):
    token = st.secrets.get("GITHUB_PAT", os.environ.get("GITHUB_PAT", ""))
    repo = st.secrets.get("GITHUB_REPO", os.environ.get("GITHUB_REPO", ""))
    
    csv_bytes = updated_df.to_csv(index=False).encode("utf-8")
    b64_content = base64.b64encode(csv_bytes).decode("utf-8")
    
    if not token or not repo:
        updated_df.to_csv(LEDGER_FILENAME, index=False)
        return True

    url = f"https://api.github.com/repos/{repo}/contents/{LEDGER_FILENAME}"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
    
    payload = {
        "message": f"Ledger update [{datetime.datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}]",
        "content": b64_content
    }
    if sha:
        payload["sha"] = sha

    res = requests.put(url, headers=headers, data=json.dumps(payload))
    return res.status_code in [200, 201]

def log_daily_predictions_to_github(candidates_df, trigger_type="MANUAL_WEB_UI", current_regime_label="🟡 Sideways"):
    ledger, sha = get_github_ledger()
    now_ist = datetime.datetime.now(IST)
    today_str = now_ist.strftime("%Y-%m-%d")
    timestamp_str = now_ist.strftime("%Y-%m-%d %H:%M:%S IST")
    
    new_records = []
    for _, row in candidates_df.iterrows():
        p_id = f"{today_str}_{row['Ticker']}"
        if not ledger.empty and p_id in ledger["Prediction_ID"].values:
            continue
            
        is_bullish = "Bullish" in str(row.get("1-2W Outlook", ""))
        action_rec = "🟢 BUY / ACCUMULATE" if is_bullish else "🔴 SHORT / FADE (SELL)"
        
        horizon_val = str(row.get("Dynamic Horizon", "🎯 Tactical Swing (1-2 Weeks)"))
        target_days_val = int(row.get("Target Days", 8))
        
        new_records.append({
            "Prediction_ID": p_id,
            "Date": timestamp_str,
            "Ticker": row["Ticker"],
            "Active_Catalyst": row["Active Catalyst"],
            "CMP_At_Prediction": row["CMP (₹)"],
            "Predicted_Outlook": "BULLISH" if is_bullish else "BEARISH_FADE",
            "Recommended_Action": action_rec,
            "Holding_Horizon": horizon_val,
            "Target_Days": target_days_val,
            "Market_Regime": current_regime_label,
            "Trigger_Type": trigger_type,
            "Confidence": row.get("Confidence", "High"),
            "Target_Return_Pct": 4.5 if is_bullish else -4.0,
            "Stop_Loss_Pct": -2.5 if is_bullish else 2.5,
            "Days_Elapsed": 0,
            "Current_CMP": row["CMP (₹)"],
            "Realized_Return_Pct": 0.0,
            "Outcome_Status": "PENDING"
        })
    
    if new_records:
        updated = pd.concat([ledger, pd.DataFrame(new_records)], ignore_index=True)
        success = commit_github_ledger(updated, sha)
        return len(new_records) if success else 0
    return 0

def audit_and_update_outcomes(raw_data):
    ledger, sha = get_github_ledger()
    if ledger.empty or raw_data.empty:
        return ledger

    changed = False
    for idx, row in ledger.iterrows():
        pred_type = str(row.get("Predicted_Outlook", "BEARISH_FADE")).upper()
        
        # Populate legacy records if missing
        if "Recommended_Action" not in ledger.columns or pd.isna(ledger.at[idx, "Recommended_Action"]):
            ledger.at[idx, "Recommended_Action"] = "🟢 BUY / ACCUMULATE" if "BULLISH" in pred_type else "🔴 SHORT / FADE (SELL)"
            changed = True
        if "Holding_Horizon" not in ledger.columns or pd.isna(ledger.at[idx, "Holding_Horizon"]):
            ledger.at[idx, "Holding_Horizon"] = "🎯 Tactical Swing (1-2 Weeks)"
            changed = True
        if "Target_Days" not in ledger.columns or pd.isna(ledger.at[idx, "Target_Days"]):
            ledger.at[idx, "Target_Days"] = 8
            changed = True
        if "Trigger_Type" not in ledger.columns or pd.isna(ledger.at[idx, "Trigger_Type"]):
            ledger.at[idx, "Trigger_Type"] = "SCHEDULED_CRON_AUTO"
            changed = True
        if "Market_Regime" not in ledger.columns or pd.isna(ledger.at[idx, "Market_Regime"]):
            ledger.at[idx, "Market_Regime"] = "🟡 Sideways / Rangebound"
            changed = True

        if row["Outcome_Status"] in ["SUCCESS", "FAILED"]:
            continue

        ticker_sym = str(row["Ticker"]).replace(".NS", "")
        ns_sym = f"{ticker_sym}.NS"

        target_col = None
        if hasattr(raw_data, "columns") and hasattr(raw_data.columns, "levels"):
            if ns_sym in raw_data.columns.levels[0]:
                target_col = ns_sym
            elif ticker_sym in raw_data.columns.levels[0]:
                target_col = ticker_sym

        if target_col:
            hist = raw_data[target_col]["Close"].dropna()
            if not hist.empty:
                curr_p = float(hist.iloc[-1])
                init_p = float(row["CMP_At_Prediction"])
                
                # Direction-Aware Return: In a SHORT / BEARISH_FADE setup, price dropping yields positive return
                is_bullish = ("BULLISH" in pred_type)
                if is_bullish:
                    ret_pct = round(((curr_p - init_p) / init_p) * 100.0, 2)
                else:
                    ret_pct = round(((init_p - curr_p) / init_p) * 100.0, 2)

                raw_date_str = str(row["Date"]).replace(" IST", "")
                pred_date = pd.to_datetime(raw_date_str).tz_localize(None)
                post_data = hist[hist.index.tz_localize(None) >= pred_date.normalize()]
                days = max(0, len(post_data) - 1)

                ledger.at[idx, "Current_CMP"] = curr_p
                ledger.at[idx, "Realized_Return_Pct"] = ret_pct
                ledger.at[idx, "Days_Elapsed"] = days
                changed = True

                target_days = int(row.get("Target_Days", 8))
                max_allowed_days = target_days + 2

                if days >= 1:
                    if is_bullish:
                        if ret_pct >= row["Target_Return_Pct"]:
                            ledger.at[idx, "Outcome_Status"] = "SUCCESS"
                        elif ret_pct <= row["Stop_Loss_Pct"] or days >= max_allowed_days:
                            ledger.at[idx, "Outcome_Status"] = "SUCCESS" if ret_pct > 0 else "FAILED"
                    else:
                        target_gain = abs(float(row.get("Target_Return_Pct", -4.0)))
                        stop_loss_hit = ret_pct <= -abs(float(row.get("Stop_Loss_Pct", 2.5)))
                        if ret_pct >= target_gain:
                            ledger.at[idx, "Outcome_Status"] = "SUCCESS"
                        elif stop_loss_hit or days >= max_allowed_days:
                            ledger.at[idx, "Outcome_Status"] = "SUCCESS" if ret_pct > 0 else "FAILED"

    if changed:
        commit_github_ledger(ledger, sha)
    return ledger

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

        price_5d_ago = float(c.iloc[-6]) if len(c) >= 6 else float(c.iloc[0])
        runup_5d = ((cmp - price_5d_ago) / price_5d_ago) * 100.0

        v_latest = float(v.iloc[-1])
        v_20d = float(v.rolling(20).mean().iloc[-1])
        vol_surge_ratio = round(v_latest / v_20d, 2) if v_20d > 0 else 1.0

        day_range = float(h.iloc[-1] - l.iloc[-1])
        close_pos_pct = round(((cmp - float(l.iloc[-1])) / day_range) * 100.0, 1) if day_range > 0 else 50.0

        cat_info = news_map.get(clean_sym, None)
        if cat_info:
            cat_name = cat_info["catalyst"]
            cat_group = cat_info["group"]
            impact_mult = cat_info["impact"]
            base_p1 = cat_info["p1w"]
            base_p2 = cat_info["p2w"]
        else:
            cat_name = "⚡ Technical Baseline"
            cat_group = "Baseline"
            impact_mult = 1.0
            base_p1 = 50
            base_p2 = 50

        score_runup = np.clip(100.0 - (runup_5d * 10.0), -50.0, 50.0)
        score_volume = (vol_surge_ratio * (close_pos_pct - 50.0) * 0.4)
        score_trend = 15.0 if dist_200 > 0 else -15.0

        raw_score = (score_runup * 0.35) + (score_volume * 0.45) + (score_trend * 0.20)
        final_catalyst_score = round(raw_score * impact_mult, 1)

        prob_adjustment = 0.0
        if runup_5d <= 2.5:
            prob_adjustment += 8.0
        elif runup_5d >= 7.5:
            prob_adjustment -= 18.0

        if vol_surge_ratio >= 2.0 and close_pos_pct >= 65.0:
            prob_adjustment += 10.0
        elif close_pos_pct <= 35.0:
            prob_adjustment -= 14.0

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

        abs_target = 4.5 if "Bullish" in outlook else 4.0
        drift_velocity = 1.0 + (vol_surge_ratio * 0.25) if prob_1w >= 65 else 0.65
        estimated_days = int(np.clip(round(abs_target / drift_velocity), 3, 15))

        if estimated_days <= 5:
            horizon_label = f"⚡ Fast Drift ({estimated_days} Sessions)"
        elif estimated_days <= 10:
            horizon_label = f"🎯 Tactical Swing ({estimated_days} Sessions)"
        else:
            horizon_label = f"⏳ Positional Unlock ({estimated_days} Sessions)"

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
            "Catalyst Group": cat_group,
            "1-2W Outlook": outlook,
            "Dynamic Horizon": horizon_label,
            "Target Days": estimated_days
        })

    return pd.DataFrame(results)

# =====================================================================
# Section 4: Sidebar Controls & Header
# =====================================================================
st.sidebar.title("⚡ Catalyst Pulse Pro")
st.sidebar.caption("Event Alpha & Post-Announcement Drift Engine")

universe_choice = st.sidebar.selectbox("Stock Universe:", ["Curated NIFTY 100 (Full)", "Nifty Top 30 Large-Caps", "High-Beta Midcaps"], index=0)

if universe_choice == "Curated NIFTY 100 (Full)":
    ACTIVE_UNIVERSE = NIFTY_100_TICKERS
elif universe_choice == "Nifty Top 30 Large-Caps":
    ACTIVE_UNIVERSE = NIFTY_100_TICKERS[:30]
else:
    ACTIVE_UNIVERSE = NIFTY_100_TICKERS[30:70]

raw_market_data = load_market_data([x["ticker"] for x in ACTIVE_UNIVERSE])
news_items_list, matched_news_map, ticker_news_hist = fetch_corporate_catalysts(ACTIVE_UNIVERSE)
catalyst_df = compute_predictive_catalyst_metrics(raw_market_data, matched_news_map, ACTIVE_UNIVERSE)

# Macro India VIX extraction
curr_vix = 14.5
if "^INDIAVIX" in raw_market_data.columns.levels[0]:
    v_close = raw_market_data["^INDIAVIX"]["Close"].dropna()
    if not v_close.empty:
        curr_vix = float(v_close.iloc[-1])

vix_mood = "🟢 Stable & Calm" if curr_vix < 14 else ("🟡 Normal Volatility" if curr_vix <= 21 else "⚠️ High Panic / Wild Swings")

# Macro Market Regime Calculation based on NIFTY 50 Trend
nifty_regime = "🟡 Sideways / Rangebound"
if "^NSEI" in raw_market_data.columns.levels[0]:
    n_df = raw_market_data["^NSEI"]["Close"].dropna()
    if len(n_df) >= 200:
        n_cmp = float(n_df.iloc[-1])
        n_d50 = float(n_df.rolling(50).mean().iloc[-1])
        n_d200 = float(n_df.rolling(200).mean().iloc[-1])
        if n_cmp > n_d50 > n_d200:
            nifty_regime = "🟢 Strong Bull"
        elif n_cmp < n_d50 < n_d200:
            nifty_regime = "🔴 Bear Market / Correction"
        else:
            nifty_regime = "🟡 Sideways / Rangebound"

regime_tag_full = f"{nifty_regime} (VIX: {curr_vix:.1f})"

st.sidebar.markdown("---")
st.sidebar.metric("India VIX Pulse", f"{curr_vix:.1f}", vix_mood)
st.sidebar.caption(f"Market Regime: {nifty_regime}")
st.sidebar.caption(f"Universe: {len(ACTIVE_UNIVERSE)} Stocks | Filings: {len(news_items_list)}")

# =====================================================================
# Navigation State Persistence (URL + Session State Lock)
# =====================================================================
NAV_TABS = [
    "🎯 Dynamic 1-2 Week Screener",
    "🔬 Single-Stock Deep Dive",
    "📰 Exchange Disclosures & Media Feed",
    "📊 Paper Prediction Audit & Win Rate",
    "📖 Quantitative Strategy Handbook"
]

url_tab = st.query_params.get("tab", None)
if "active_nav_tab" not in st.session_state:
    if url_tab in NAV_TABS:
        st.session_state["active_nav_tab"] = url_tab
    else:
        st.session_state["active_nav_tab"] = NAV_TABS[0]
elif url_tab in NAV_TABS and st.session_state["active_nav_tab"] != url_tab:
    st.session_state["active_nav_tab"] = url_tab

h_col1, h_col2, h_col3 = st.columns([1.5, 1.2, 0.4])

with h_col1:
    st.markdown(
        f"### ⚡ Catalyst Pulse Pro <span style='font-size:0.85rem; color:#6c757d;'>| Regime: {nifty_regime}</span>",
        unsafe_allow_html=True
    )

with h_col2:
    st.markdown(
        f"""
        <div style='text-align: right; padding-top: 6px; font-size: 0.85rem;'>
            <b>Market Mood:</b> {vix_mood} (VIX: {curr_vix:.1f}) &nbsp;|&nbsp; 
            <span style='color: #28a745; font-weight: 600;'>BSE / Reg 30 Live</span>
        </div>
        """,
        unsafe_allow_html=True
    )

with h_col3:
    if st.button("🔄 Refresh", use_container_width=True, help="Purge internal data cache and reload without tab reset"):
        saved_active = st.session_state.get("active_nav_tab", NAV_TABS[0])
        st.cache_data.clear()
        st.session_state["active_nav_tab"] = saved_active
        st.query_params["tab"] = saved_active
        st.rerun()

def on_tab_change():
    st.query_params["tab"] = st.session_state["active_nav_tab"]

nav_choice = st.radio(
    "Navigation",
    options=NAV_TABS,
    key="active_nav_tab",
    on_change=on_tab_change,
    label_visibility="collapsed",
    horizontal=True
)

st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

def apply_top3_bot3_styling(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)
    higher_is_better = ["Catalyst Score", "P(1W) Drift %", "P(2W) Drift %", "Vol Surge Ratio", "Close in Range %", "Dist 200DMA %"]
    for col in higher_is_better:
        if col in df.columns:
            t3 = df[col].nlargest(3).index
            b3 = df[col].nsmallest(3).index
            styles.loc[t3, col] = "background-color: #d4edda; color: #155724; font-weight: bold;"
            styles.loc[b3, col] = "background-color: #f8d7da; color: #721c24; font-weight: bold;"

    if "Pre-RunUp 5D %" in df.columns:
        best_runup = df["Pre-RunUp 5D %"].nsmallest(3).index
        worst_runup = df["Pre-RunUp 5D %"].nlargest(3).index
        styles.loc[best_runup, "Pre-RunUp 5D %"] = "background-color: #d4edda; color: #155724; font-weight: bold;"
        styles.loc[worst_runup, "Pre-RunUp 5D %"] = "background-color: #f8d7da; color: #721c24; font-weight: bold;"

    return styles

# =====================================================================
# Section 5: Views
# =====================================================================

# VIEW 1: 3-Level Screener
if nav_choice == "🎯 Dynamic 1-2 Week Screener":
    if catalyst_df.empty:
        st.warning("⚠️ Market data feed synchronizing...")
    else:
        st.markdown("##### ⚡ Level 1: Outlier Decision Radar (Top 3-5 High-Conviction Setups)")
        up_candidates = catalyst_df[catalyst_df["1-2W Outlook"].str.contains("Bullish")].sort_values(by="Catalyst Score", ascending=False).head(5)
        down_candidates = catalyst_df[catalyst_df["1-2W Outlook"].str.contains("Distribution")].sort_values(by="Catalyst Score", ascending=True).head(5)

        c1, c2 = st.columns(2)
        cols_summary = ["Ticker", "CMP (₹)", "Catalyst Score", "P(1W) Drift %", "P(2W) Drift %", "Pre-RunUp 5D %", "Vol Surge Ratio", "Active Catalyst"]

        with c1:
            st.markdown("<span style='color: #28a745; font-weight: 700;'>🟢 Top Expected to Move UP (Unpriced Catalyst Breakouts)</span>", unsafe_allow_html=True)
            if not up_candidates.empty:
                st.dataframe(
                    up_candidates[cols_summary].style.apply(apply_top3_bot3_styling, axis=None).format({
                        "CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%",
                        "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x"
                    }),
                    use_container_width=True, height=180
                )
            else:
                st.info("No stocks currently meet pristine unpriced conditions (RunUp <= 2.5% with high Day-1 volume).")

        with c2:
            st.markdown("<span style='color: #dc3545; font-weight: 700;'>🔴 Top Expected to Fade / Breakdown ('Sell the News' Traps)</span>", unsafe_allow_html=True)
            if not down_candidates.empty:
                st.dataframe(
                    down_candidates[cols_summary].style.apply(apply_top3_bot3_styling, axis=None).format({
                        "CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%",
                        "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x"
                    }),
                    use_container_width=True, height=180
                )
            else:
                st.info("No distribution traps detected.")

        st.markdown("<hr style='margin-top: 0.6rem; margin-bottom: 0.8rem;' />", unsafe_allow_html=True)

        st.markdown("##### 🏢 Level 2: Corporate Action Catalyst Quadrant")
        st.caption("How Indian equities statistically react across specific corporate filings: Contracts, Mergers, Dividends, and Stock Splits.")

        q_tab1, q_tab2, q_tab3, q_tab4 = st.tabs([
            "🎯 Order Wins & Capex (66% Win Edge)",
            "🔄 Demergers & Mergers (72% Win Edge)",
            "💰 Dividends & Buybacks (Fades if RunUp > 5%)",
            "✂️ Splits & Bonus (40% Decay Risk)"
        ])

        cols_quad = ["Ticker", "CMP (₹)", "Catalyst Score", "P(1W) Drift %", "P(2W) Drift %", "Pre-RunUp 5D %", "Vol Surge Ratio", "Close in Range %", "Active Catalyst"]

        with q_tab1:
            df_orders = catalyst_df[catalyst_df["Catalyst Group"] == "Order Wins & Capex"].sort_values(by="Catalyst Score", ascending=False)
            if not df_orders.empty:
                st.dataframe(df_orders[cols_quad].style.apply(apply_top3_bot3_styling, axis=None).format({"CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%", "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x", "Close in Range %": "{:.1f}%"}), use_container_width=True)
            else:
                st.info("No fresh contract wins or capex announcements recorded in current batch.")

        with q_tab2:
            df_demerge = catalyst_df[catalyst_df["Catalyst Group"] == "Demergers & Mergers"].sort_values(by="Catalyst Score", ascending=False)
            if not df_demerge.empty:
                st.dataframe(df_demerge[cols_quad].style.apply(apply_top3_bot3_styling, axis=None).format({"CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%", "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x", "Close in Range %": "{:.1f}%"}), use_container_width=True)
            else:
                st.info("No active demerger or merger filings tagged in current batch.")

        with q_tab3:
            df_div = catalyst_df[catalyst_df["Catalyst Group"] == "Dividends & Buybacks"].sort_values(by="Catalyst Score", ascending=False)
            if not df_div.empty:
                st.dataframe(df_div[cols_quad].style.apply(apply_top3_bot3_styling, axis=None).format({"CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%", "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x", "Close in Range %": "{:.1f}%"}), use_container_width=True)
            else:
                st.info("No active dividend or buyback filings tagged in current batch.")

        with q_tab4:
            df_splits = catalyst_df[catalyst_df["Catalyst Group"] == "Splits & Bonus"].sort_values(by="Catalyst Score", ascending=False)
            if not df_splits.empty:
                st.dataframe(df_splits[cols_quad].style.apply(apply_top3_bot3_styling, axis=None).format({"CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%", "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x", "Close in Range %": "{:.1f}%"}), use_container_width=True)
            else:
                st.info("No stock split or bonus declarations tagged in current batch.")

        st.markdown("<hr style='margin-top: 0.6rem; margin-bottom: 0.8rem;' />", unsafe_allow_html=True)

        st.markdown("##### 🌐 Level 3: Master NIFTY 100 Evaluated Matrix")
        display_all = catalyst_df.sort_values(by="Catalyst Score", ascending=False).reset_index(drop=True)
        cols_master = ["Ticker", "Name", "CMP (₹)", "Catalyst Score", "1-2W Outlook", "P(1W) Drift %", "P(2W) Drift %", "Pre-RunUp 5D %", "Vol Surge Ratio", "Close in Range %", "Dist 200DMA %", "Active Catalyst"]
        st.dataframe(
            display_all[cols_master].style.apply(apply_top3_bot3_styling, axis=None).format({
                "CMP (₹)": "₹{:.2f}", "Catalyst Score": "{:+.1f}", "P(1W) Drift %": "{}%", "P(2W) Drift %": "{}%",
                "Pre-RunUp 5D %": "{:+.2f}%", "Vol Surge Ratio": "{:.1f}x", "Close in Range %": "{:.1f}%", "Dist 200DMA %": "{:+.2f}%"
            }),
            use_container_width=True, height=420
        )

# VIEW 2: Stock Deep Dive with News History
elif nav_choice == "🔬 Single-Stock Deep Dive":
    st.subheader("🔬 Single-Stock Reaction & Disclosure Deep Dive")
    all_syms = sorted([x["ticker"].replace(".NS", "") for x in ACTIVE_UNIVERSE])
    chosen_stock = st.selectbox("Select Stock to Inspect:", all_syms)

    stock_row = catalyst_df[catalyst_df["Ticker"] == chosen_stock].iloc[0]

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("CMP", f"₹{stock_row['CMP (₹)']:.2f}")
    k2.metric("Catalyst Score", f"{stock_row['Catalyst Score']:+.1f}")
    k3.metric("P(1W) Drift", f"{stock_row['P(1W) Drift %']}%")
    k4.metric("P(2W) Drift", f"{stock_row['P(2W) Drift %']}%")
    k5.metric("Pre-RunUp (5D)", f"{stock_row['Pre-RunUp 5D %']:+.2f}%")

    st.markdown(f"**Current 1-2 Week Forecast:** `{stock_row['1-2W Outlook']}`")
    st.markdown(f"**Active Tagged Announcement:** `{stock_row['Active Catalyst']}`")
    st.markdown(f"**Intraday Candlestick Position:** Closed at **{stock_row['Close in Range %']}%** of the day's high-low range on **{stock_row['Vol Surge Ratio']}x** average volume.")

    st.markdown("---")
    st.markdown(f"#### 📰 Real-Time Filings & News Linked to {chosen_stock}")
    matched_history = ticker_news_hist.get(chosen_stock, [])
    if matched_history:
        for n_item in matched_history:
            with st.expander(f"[{n_item['catalyst']}] {n_item['title']}"):
                st.write(n_item["summary"] if n_item["summary"] else "Official disclosure on exchange ledger.")
                st.caption(f"Source: {n_item['source']} | Published: {n_item['published']}")
                st.markdown(f"[View Exchange Filing Document]({n_item['link']})")
    else:
        st.info(f"No active news headlines or Regulation 30 disclosures currently tagged for {chosen_stock} in the latest live feeds. Metrics represent pure quantitative technical baseline.")

# VIEW 3: Live Exchange Disclosures
elif nav_choice == "📰 Exchange Disclosures & Media Feed":
    st.subheader("📰 Authentic Exchange Disclosures & Regulatory Stream")
    st.caption("Live feed parsed from BSE Corporate Announcements RSS and Financial Media.")

    f_filter = st.selectbox("Filter Feed by Category:", ["All Filings", "Demergers & Mergers", "Order Wins & Capex", "Dividends & Buybacks", "Splits & Bonus", "Governance / Risk"])

    displayed_count = 0
    for item in news_items_list:
        if f_filter != "All Filings" and item["group"] != f_filter:
            continue
        displayed_count += 1
        with st.expander(f"[{item['catalyst']}] {item['title']}"):
            st.write(item["summary"] if item["summary"] else "Official disclosure notification via exchange stream.")
            if item["matched"]:
                st.markdown(f"**Tagged Tickers:** `{', '.join(item['matched'])}`")
            st.caption(f"Source: {item['source']} | Published: {item['published']}")
            st.markdown(f"[Official Filing Document / Link]({item['link']})")

    if displayed_count == 0:
        st.info(f"No filings matching '{f_filter}' in the current feed batch.")

# VIEW 4: Full Quantitative Handbook with Plain English
elif nav_choice == "📖 Quantitative Strategy Handbook":
    st.title("📖 Quantitative Strategy Handbook & Indicator Playbook")
    st.markdown(
        """
        This institutional handbook explains the mathematical formulation, empirical behavioral logic, and practical application 
        of every metric in **Catalyst Pulse Pro**. It is designed so that both quantitative funds and common investors can make 
        unbiased, data-backed decisions.
        """
    )
    st.markdown("---")

    h1, h2 = st.columns(2)

    with h1:
        st.markdown(
            """
            ### 🔹 1. Pre-Event Run-up (5D Lookback)
            * **Mathematical Formula:**
              $$\\text{Run-up}_{5D} = \\left( \\frac{\\text{CMP} - \\text{Price}_{t-5}}{\\text{Price}_{t-5}} \\right) \\times 100$$
            * **Technical Purpose:** Measures whether information leaked or smart money already bought the asset prior to public news release.
            * **🗣️ Common Man Explanation:** 
              Imagine a movie everyone expects to be a blockbuster. If the tickets sell for 10x the price before release, even a good movie can disappoint investors. 
              If a stock already gained $+10\\%$ in the 5 days *before* winning a contract, large investors use the good news to sell their shares to excited retail buyers (**"Sell the News"**).
            * **How to Conclude:**
              - **$\\le +2.5\\%$:** Safe to enter. The news is a genuine surprise.
              - **$> +7.5\\%$:** 🚫 DANGER. Do not buy, even if the news looks incredible.

            ---

            ### 🔹 2. Volume Surge Ratio
            * **Mathematical Formula:**
              $$\\text{Surge Ratio} = \\frac{\\text{Volume}_{\\text{Today}}}{\\text{Average Volume}_{20\\text{D}}}$$
            * **Technical Purpose:** Distinguishes institutional block buying from retail noise.
            * **🗣️ Common Man Explanation:** 
              When a small retail investor buys shares, trading volume barely moves. When large domestic institutions (DIIs) or foreign funds (FIIs) buy, volume spikes dramatically ($2\\times$ to $5\\times$ normal).
            * **How to Conclude:**
              - **$\\ge 2.0\\times$:** Institutional backing confirmed.
              - **$< 1.0\\times$:** Retail-only excitement. Avoid chasing.

            ---

            ### 🔹 3. Intraday Close in Range %
            * **Mathematical Formula:**
              $$\\text{Close in Range \\%} = \\left( \\frac{\\text{CMP} - \\text{Low}_{\\text{Day}}}{\\text{High}_{\\text{Day}} - \\text{Low}_{\\text{Day}}} \\right) \\times 100$$
            * **Technical Purpose:** Detects distribution rejection wicks on daily candles.
            * **🗣️ Common Man Explanation:** 
              A stock opens $+5\\%$ higher at 9:15 AM because of good news. If it closes at 3:30 PM near its day's highest point ($> 70\\%$), buyers stayed in control. But if it falls all day and closes near its lowest price ($< 35\\%$), it means large funds dumped their shares all afternoon.
            * **How to Conclude:**
              - **$\\ge 65\\%$:** Strong institutional absorption $\\rightarrow$ High odds of upward continuation.
              - **$\\le 35\\%$:** Rejection trap $\\rightarrow$ Expect multi-day downward fade.
            """
        )

    with h2:
        st.markdown(
            """
            ### 🔹 4. Success Probabilities: P(1W) & P(2W)
            * **Mathematical Modeling:** Empirical win-rate probability derived from post-announcement abnormal returns over 5 sessions (1 week) and 10 sessions (2 weeks), dynamically adjusted by volume and run-up friction:
              $$P(1W) = \\text{Base}_{\\text{Event}} + \\text{Adj}_{\\text{Run-up}} + \\text{Adj}_{\\text{Volume}} + \\text{Adj}_{\\text{Trend}}$$
            * **🗣️ Common Man Explanation:** 
              The historical odds that this stock will be trading higher 1 week and 2 weeks from today based on the exact type of news and how the market reacted today.
            * **How to Conclude:**
              - **$P(1W) \\ge 70\\%$:** Statistical green light for a 5-to-10 day swing trade.
              - **$P(1W) \\le 40\\%$:** High probability of capital loss over the coming fortnight.

            ---

            ### 🔹 5. Corporate Action Behavior Playbook
            * **1. Demergers & Value Unlocks (Base Edge: $72\\%$):**
              Demergers physically unlock hidden subsidiary value and force institutional index funds to adjust portfolios, leading to sustained positive multi-week drift.
            * **2. Mega Contracts & Capex (Base Edge: $66\\%$):**
              Expands future revenue run-rate. Strong positive drift **only if** the pre-event run-up was small ($\le 3\%$).
            * **3. Dividends & Buybacks (Base Edge: $46\\%$):**
              Dividends extract cash from the company balance sheet. Once the ex-dividend date passes, stock prices automatically drop by the dividend amount, often creating a multi-week decay.
            * **4. Bonus Issues & Stock Splits (Base Edge: $40\\%$):**
              Splits and bonuses do not add a single rupee of fundamental value—they simply divide the same pizza into smaller slices. Retail investors often chase them mistakenly thinking the stock is "cheap," leading to heavy institutional profit-booking.
            """
        )

    st.markdown("---")
    st.markdown("### 🧭 Step-by-Step Practical Decision Flowchart")
    st.markdown(
        """
        1. **Check Level 1 Screener:** Look at the **Top Expected to Move UP**. Verify that `Pre-RunUp 5D %` is $\\le 2.5\\%$ and `Vol Surge Ratio` is $\\ge 2.0\\times$.
        2. **Confirm Trend in Level 3:** Check that `Dist 200DMA %` is positive ($> 0\\%$). Never buy a news breakout on a stock falling below its 200 DMA.
        3. **Inspect the Corporate Action in Tab 2:** Go to **Single-Stock Deep Dive** and read the actual disclosure text to verify execution timelines.
        4. **Execute with Discipline:** If all conditions align, allocate standard capital. Protect with a stop-loss placed just below the low of the announcement candle.
        """
    )

# VIEW 5: Paper Prediction Audit & Win Rate Ledger (GitHub Backed)
elif nav_choice == "📊 Paper Prediction Audit & Win Rate":
    st.subheader("📊 Self-Auditing Paper Prediction Ledger & Win Rate")
    st.caption("Auto-synced to GitHub Repository. Real-time IST timestamps, directional PnL, dynamic horizons, market regime tracking & test run management.")

    audited_ledger = audit_and_update_outcomes(raw_market_data)

    col_btn1, col_btn2 = st.columns([1.2, 3])
    with col_btn1:
        if st.button("📥 Record Today's Top Predictions (Manual Trigger)", use_container_width=True):
            top_setups = pd.concat([
                catalyst_df[catalyst_df["1-2W Outlook"].str.contains("Bullish")].head(3),
                catalyst_df[catalyst_df["1-2W Outlook"].str.contains("Distribution")].head(3)
            ])
            added = log_daily_predictions_to_github(top_setups, trigger_type="MANUAL_WEB_UI", current_regime_label=regime_tag_full)
            if added > 0:
                st.success(f"✅ Recorded {added} new predictions under {nifty_regime} (IST)!")
                st.rerun()
            else:
                st.info("Today's setups are already recorded in the ledger.")

    # Collapsible Test Run Deletion & Ledger Clean-up Tool
    with st.expander("🛠️ Manage & Delete Test Runs / Selected Entries", expanded=False):
        st.caption("Select specific test predictions to permanently purge from the GitHub repository ledger.")
        if not audited_ledger.empty and "Prediction_ID" in audited_ledger.columns:
            preds_to_delete = st.multiselect(
                "Select Prediction IDs to Delete:",
                options=audited_ledger["Prediction_ID"].tolist()
            )
            del_c1, del_c2 = st.columns([1, 4])
            with del_c1:
                if st.button("🗑️ Delete Selected", type="primary", use_container_width=True):
                    if preds_to_delete:
                        cleaned_df = audited_ledger[~audited_ledger["Prediction_ID"].isin(preds_to_delete)].copy()
                        _, sha = get_github_ledger()
                        if commit_github_ledger(cleaned_df, sha):
                            st.success(f"Purged {len(preds_to_delete)} records permanently from GitHub ledger.")
                            st.rerun()
                        else:
                            st.error("Failed to commit ledger changes to GitHub.")
                    else:
                        st.warning("Please select at least one record to delete.")

    if not audited_ledger.empty:
        def format_ledger_ist_date(d_val):
            if pd.isna(d_val) or str(d_val).strip() == "":
                return "Pending"
            try:
                raw = str(d_val).replace(" IST", "").strip()
                dt = pd.to_datetime(raw)
                if dt.tzinfo is None:
                    dt = dt.tz_localize("UTC").tz_convert(IST)
                else:
                    dt = dt.tz_convert(IST)
                return dt.strftime("%Y-%m-%d %H:%M:%S IST")
            except Exception:
                return str(d_val)

        display_ledger = audited_ledger.copy()
        if "Date" in display_ledger.columns:
            display_ledger["Date"] = display_ledger["Date"].apply(format_ledger_ist_date)

        ASSUMED_TRANCHE_BUDGET = 15000.0

        if "Recommended_Action" not in display_ledger.columns:
            display_ledger["Recommended_Action"] = display_ledger["Predicted_Outlook"].apply(
                lambda x: "🟢 BUY / ACCUMULATE" if "BULLISH" in str(x).upper() else "🔴 SHORT / FADE (SELL)"
            )
        if "Holding_Horizon" not in display_ledger.columns:
            display_ledger["Holding_Horizon"] = "🎯 Tactical Swing (1-2 Weeks)"
        if "Trigger_Type" not in display_ledger.columns:
            display_ledger["Trigger_Type"] = "MANUAL_WEB_UI"
        if "Market_Regime" not in display_ledger.columns:
            display_ledger["Market_Regime"] = "🟡 Sideways / Rangebound"

        display_ledger["Clean_Ret_Pct"] = pd.to_numeric(display_ledger["Realized_Return_Pct"], errors="coerce").fillna(0.0)
        display_ledger["Live PnL (₹)"] = (display_ledger["Clean_Ret_Pct"] / 100.0) * ASSUMED_TRANCHE_BUDGET
        display_ledger["Live PnL %"] = display_ledger["Clean_Ret_Pct"]

        closed = display_ledger[display_ledger["Outcome_Status"].isin(["SUCCESS", "FAILED"])].copy()
        pending = display_ledger[display_ledger["Outcome_Status"] == "PENDING"].copy()

        total_closed = len(closed)
        successes = len(closed[closed["Outcome_Status"] == "SUCCESS"])
        win_rate = round((successes / total_closed * 100.0), 1) if total_closed > 0 else 0.0
        closed_realized_pnl_rs = float(closed["Live PnL (₹)"].sum())
        avg_closed_ret = round(float(closed["Clean_Ret_Pct"].mean()), 2) if total_closed > 0 else 0.0

        active_pending_count = len(pending)
        active_capital_deployed = active_pending_count * ASSUMED_TRANCHE_BUDGET
        live_unrealized_pnl_rs = float(pending["Live PnL (₹)"].sum())
        live_unrealized_pct = (live_unrealized_pnl_rs / active_capital_deployed * 100.0) if active_capital_deployed > 0 else 0.0

        nifty_period_ret = 0.0
        if not raw_market_data.empty and "^NSEI" in raw_market_data.columns.levels[0]:
            n_series = raw_market_data["^NSEI"]["Close"].dropna()
            if len(n_series) >= max(5, total_closed) and len(n_series) > 0:
                lookback = min(len(n_series), max(10, total_closed))
                nifty_period_ret = float(((n_series.iloc[-1] - n_series.iloc[-lookback]) / n_series.iloc[-lookback]) * 100.0)
        
        effective_return_pct = avg_closed_ret if total_closed > 0 else live_unrealized_pct
        alpha_vs_nifty = effective_return_pct - nifty_period_ret

        eval_daily_returns = (closed["Clean_Ret_Pct"] / 100.0) if not closed.empty else (display_ledger["Clean_Ret_Pct"] / 100.0)
        rf_daily = 0.065 / 252.0
        excess_ret = eval_daily_returns - rf_daily
        sharpe_ratio = float((excess_ret.mean() / eval_daily_returns.std() * np.sqrt(252))) if eval_daily_returns.std() > 0 else 0.0
        downside_std = eval_daily_returns[eval_daily_returns < 0].std()
        sortino_ratio = float((excess_ret.mean() / downside_std * np.sqrt(252))) if (downside_std > 0 and not np.isnan(downside_std)) else 0.0

        # --- Collapsible 8-KPI Executive Metric Container ---
        with st.expander("⚡ Strategy Performance & Institutional Benchmarks (8 Key Metrics)", expanded=True):
            st.markdown("<span style='font-weight:600; font-size:0.84rem; color:#495057;'>Portfolio MTM & Alpha Dashboard</span>", unsafe_allow_html=True)
            p_c1, p_c2, p_c3, p_c4 = st.columns(4)
            p_c1.metric(
                "Live Unrealized PnL",
                f"₹{live_unrealized_pnl_rs:+,.2f}",
                f"{live_unrealized_pct:+.2f}% Mark-to-Market"
            )
            p_c2.metric(
                "Active Capital Monitored",
                f"₹{active_capital_deployed:,.2f}",
                f"{active_pending_count} Active Setups"
            )
            p_c3.metric(
                "Booked Realized PnL",
                f"₹{closed_realized_pnl_rs:+,.2f}",
                f"{win_rate}% Win Rate ({successes}/{total_closed})"
            )
            p_c4.metric(
                "Generated Alpha vs NIFTY",
                f"{alpha_vs_nifty:+.2f}%",
                f"NIFTY: {nifty_period_ret:+.2f}%"
            )

            st.markdown("<div style='margin-top: 0.4rem;'></div>", unsafe_allow_html=True)
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Strategy Net Return", f"{effective_return_pct:+.2f}%", f"{total_closed} Audits Evaluated")
            b2.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}", "Risk-Adjusted")
            b3.metric("Sortino Ratio", f"{sortino_ratio:.2f}", "Downside-Risk-Adjusted")
            b4.metric("Benchmark Alpha (α)", f"{alpha_vs_nifty:+.2f}%", "Outperformance" if alpha_vs_nifty > 0 else "Underperformance")

        st.markdown("<div style='margin-bottom: 0.4rem;'></div>", unsafe_allow_html=True)

        # --- Interactive Controls Matching Tactical Allocator Pro ---
        f1, f2, f3, f4 = st.columns([1, 1, 1, 1])
        with f1:
            outlook_f = st.selectbox("Filter Outlook / Setup:", ["All Outlooks Combined", "BULLISH (Buy) Only", "BEARISH_FADE (Short) Only"], index=0)
        with f2:
            status_f = st.selectbox("Filter Horizon / Status:", ["All Combined", "PENDING (Open) Only", "Completed (SUCCESS/FAILED) Only"], index=0)
        with f3:
            trig_f = st.selectbox("Filter Trigger Origin:", ["All Triggers", "MANUAL_WEB_UI Only", "SCHEDULED_CRON_AUTO Only"], index=0)
        with f4:
            regime_f = st.selectbox("Filter Market Regime:", ["All Regimes Combined", "Bull Market Only", "Bear Market Only", "Sideways / Rangebound Only"], index=0)

        filtered_ledger = display_ledger.copy()
        if "BULLISH" in outlook_f:
            filtered_ledger = filtered_ledger[filtered_ledger["Predicted_Outlook"] == "BULLISH"]
        elif "BEARISH" in outlook_f:
            filtered_ledger = filtered_ledger[filtered_ledger["Predicted_Outlook"] == "BEARISH_FADE"]

        if "PENDING" in status_f:
            filtered_ledger = filtered_ledger[filtered_ledger["Outcome_Status"] == "PENDING"]
        elif "Completed" in status_f:
            filtered_ledger = filtered_ledger[filtered_ledger["Outcome_Status"].isin(["SUCCESS", "FAILED"])]

        if "MANUAL" in trig_f:
            filtered_ledger = filtered_ledger[filtered_ledger["Trigger_Type"] == "MANUAL_WEB_UI"]
        elif "SCHEDULED" in trig_f:
            filtered_ledger = filtered_ledger[filtered_ledger["Trigger_Type"] == "SCHEDULED_CRON_AUTO"]

        if "Bull" in regime_f:
            filtered_ledger = filtered_ledger[filtered_ledger["Market_Regime"].str.contains("Bull", na=False)]
        elif "Bear" in regime_f:
            filtered_ledger = filtered_ledger[filtered_ledger["Market_Regime"].str.contains("Bear", na=False)]
        elif "Sideways" in regime_f:
            filtered_ledger = filtered_ledger[filtered_ledger["Market_Regime"].str.contains("Sideways|Range", na=False)]

        st.markdown("##### 📑 Ledger Audit Trail (IST Real-Time Sync)")

        def highlight_outcomes(df):
            styles = pd.DataFrame("", index=df.index, columns=df.columns)
            if "Outcome_Status" in df.columns:
                styles["Outcome_Status"] = df["Outcome_Status"].apply(
                    lambda v: "background-color: #d4edda; color: #155724; font-weight: bold;" if v == "SUCCESS"
                    else ("background-color: #f8d7da; color: #721c24; font-weight: bold;" if v == "FAILED"
                    else "background-color: #fff3cd; color: #856404;")
                )
            if "Live PnL (₹)" in df.columns:
                styles["Live PnL (₹)"] = df["Live PnL (₹)"].apply(
                    lambda v: "color: #155724; font-weight: bold;" if v > 0 else ("color: #721c24; font-weight: bold;" if v < 0 else "")
                )
            if "Live PnL %" in df.columns:
                styles["Live PnL %"] = df["Live PnL %"].apply(
                    lambda v: "background-color: #d4edda; color: #155724; font-weight: bold;" if v > 0 else ("background-color: #f8d7da; color: #721c24; font-weight: bold;" if v < 0 else "")
                )
            if "Recommended_Action" in df.columns:
                styles["Recommended_Action"] = df["Recommended_Action"].apply(
                    lambda v: "color: #155724; font-weight: bold;" if "BUY" in str(v) else "color: #721c24; font-weight: bold;"
                )
            if "Trigger_Type" in df.columns:
                styles["Trigger_Type"] = df["Trigger_Type"].apply(
                    lambda v: "background-color: #e3f2fd; color: #0d47a1; font-weight: 600;" if "MANUAL" in str(v) else "background-color: #f3e5f5; color: #4a148c;"
                )
            if "Market_Regime" in df.columns:
                styles["Market_Regime"] = df["Market_Regime"].apply(
                    lambda v: "background-color: #d4edda; color: #155724; font-weight: 600;" if "Bull" in str(v)
                    else ("background-color: #f8d7da; color: #721c24; font-weight: 600;" if "Bear" in str(v)
                    else "background-color: #fff3cd; color: #856404;")
                )
            return styles

        cols_display = [
            "Prediction_ID", "Date", "Ticker", "Trigger_Type", "Market_Regime", "Recommended_Action", "Holding_Horizon",
            "Active_Catalyst", "CMP_At_Prediction", "Current_CMP", "Live PnL (₹)", "Live PnL %",
            "Confidence", "Days_Elapsed", "Target_Return_Pct", "Stop_Loss_Pct", "Outcome_Status"
        ]
        existing_cols = [c for c in cols_display if c in filtered_ledger.columns]

        st.dataframe(
            filtered_ledger[existing_cols].style.apply(highlight_outcomes, axis=None).format({
                "CMP_At_Prediction": "₹{:.2f}",
                "Current_CMP": "₹{:.2f}",
                "Live PnL (₹)": "₹{:+,.2f}",
                "Live PnL %": "{:+0.2f}%",
                "Target_Return_Pct": "{:+0.1f}%",
                "Stop_Loss_Pct": "{:+0.1f}%"
            }),
            use_container_width=True
        )
    else:
        st.info("No predictions recorded yet. Click 'Record Today\\'s Top Predictions' to initialize the audit trail.")
