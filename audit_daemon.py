import os
import re
import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import yfinance as yf
import feedparser
import numpy as np

IST = ZoneInfo("Asia/Kolkata")
file_path = "catalyst_prediction_ledger.csv"

# =====================================================================
# 1. Audit Existing Pending Trades
# =====================================================================
ledger = pd.read_csv(file_path) if os.path.exists(file_path) else pd.DataFrame()
changed = False

if not ledger.empty and "Outcome_Status" in ledger.columns:
    pending = ledger[ledger["Outcome_Status"] == "PENDING"]
    if not pending.empty:
        clean_tickers = [str(t).replace(".NS", "").strip() for t in pending["Ticker"].unique()]
        ticker_list = [f"{t}.NS" for t in clean_tickers]

        print(f"Auditing open trades for: {ticker_list}")
        raw_audit = yf.download(ticker_list, period="1mo", interval="1d", group_by="ticker", auto_adjust=True)

        if not raw_audit.empty:
            for idx, row in ledger.iterrows():
                if row["Outcome_Status"] != "PENDING":
                    continue

                clean_t = str(row["Ticker"]).replace(".NS", "").strip()
                ns_sym = f"{clean_t}.NS"

                hist = pd.Series(dtype=float)
                if isinstance(raw_audit.columns, pd.MultiIndex):
                    for sym_cand in [ns_sym, clean_t]:
                        if sym_cand in raw_audit.columns.levels[0]:
                            hist = raw_audit[sym_cand]["Close"].dropna()
                            break
                else:
                    if "Close" in raw_audit.columns:
                        hist = raw_audit["Close"].dropna()

                if not hist.empty:
                    curr_p = float(hist.iloc[-1])
                    init_p = float(row["CMP_At_Prediction"])

                    pred_type = str(row.get("Predicted_Outlook", "BEARISH_FADE")).upper()
                    is_bullish = ("BULLISH" in pred_type)

                    if is_bullish:
                        ret_pct = round(((curr_p - init_p) / init_p) * 100.0, 2)
                    else:
                        ret_pct = round(((init_p - curr_p) / init_p) * 100.0, 2)

                    clean_date_str = str(row["Date"]).replace(" IST", "").strip()[:10]
                    hist_date_strs = hist.index.strftime("%Y-%m-%d")
                    post = hist[hist_date_strs >= clean_date_str]
                    days = max(0, len(post) - 1)

                    ledger.at[idx, "Current_CMP"] = curr_p
                    ledger.at[idx, "Realized_Return_Pct"] = ret_pct
                    ledger.at[idx, "Days_Elapsed"] = days
                    changed = True

                    target_days = int(row.get("Target_Days", 8)) if not pd.isna(row.get("Target_Days")) else 8
                    max_allowed_days = target_days + 2

                    if days >= 1:
                        if is_bullish:
                            target_ret = float(row.get("Target_Return_Pct", 4.5))
                            stop_ret = float(row.get("Stop_Loss_Pct", -2.5))
                            if ret_pct >= target_ret:
                                ledger.at[idx, "Outcome_Status"] = "SUCCESS"
                            elif ret_pct <= stop_ret or days >= max_allowed_days:
                                ledger.at[idx, "Outcome_Status"] = "SUCCESS" if ret_pct > 0 else "FAILED"
                        else:
                            target_gain = abs(float(row.get("Target_Return_Pct", -4.0)))
                            stop_loss_hit = ret_pct <= -abs(float(row.get("Stop_Loss_Pct", 2.5)))
                            if ret_pct >= target_gain:
                                ledger.at[idx, "Outcome_Status"] = "SUCCESS"
                            elif stop_loss_hit or days >= max_allowed_days:
                                ledger.at[idx, "Outcome_Status"] = "SUCCESS" if ret_pct > 0 else "FAILED"

# =====================================================================
# 2. Daily Automatic Screening & Prediction Logging
# =====================================================================
now_ist = datetime.datetime.now(IST)
today_str = now_ist.strftime("%Y-%m-%d")
timestamp_str = now_ist.strftime("%Y-%m-%d %H:%M:%S IST")

# Avoid duplicate runs on the same date
existing_preds = set(ledger["Prediction_ID"].values) if not ledger.empty and "Prediction_ID" in ledger.columns else set()

NIFTY_UNIVERSE = [
    "ABB", "ADANIENSOL", "ADANIENT", "ADANIGREEN", "ADANIPORTS", "ADANIPOWER",
    "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BAJAJHLDNG", "BANKBARODA", "BEL", "BHEL", "BPCL", "BHARTIARTL",
    "BOSCHLTD", "BRITANNIA", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA", "COLPAL",
    "DLF", "DABUR", "DIVISLAB", "DIXON", "DRREDDY", "EICHERMOT", "GAIL", "GODREJCP",
    "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HAVELLS", "HEROMOTOCO", "HINDALCO",
    "HAL", "HINDUNILVR", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ITC", "INDHOTEL",
    "IOC", "IRCTC", "IRFC", "INDUSINDBK", "NAUKRI", "INFY", "INDIGO", "JSWSTEEL",
    "JINDALSTEL", "JIOFIN", "KOTAKBANK", "LTIM", "LT", "LUPIN", "M&M", "MARICO",
    "MARUTI", "MAXHEALTH", "NTPC", "NESTLEIND", "ONGC", "PIDILITIND", "PFC",
    "POWERGRID", "PNB", "RECLTD", "RELIANCE", "SBICARD", "SBILIFE", "SRF",
    "MOTHERSON", "SHREECEM", "SHRIRAMFIN", "SIEMENS", "SBIN", "SUNPHARMA",
    "TVSMOTOR", "TCS", "TATACONSUM", "TATAMOTORS", "TATAPOWER", "TATASTEEL",
    "TECHM", "TITAN", "TORNTPOWER", "TRENT", "ULTRACEMCO", "UNITDSPR", "VBL",
    "VEDL", "WIPRO", "ZOMATO", "ZYDUSLIFE"
]

CATALYST_RULES = {
    "Demerger / Merger Unlock": (re.compile(r"(?:demerger|spin-?off|scheme of arrangement|merger)", re.IGNORECASE), 1.35, 72),
    "Mega Order Win / Contract": (re.compile(r"(?:bagged|awarded|receives?|secures?|wins?)\s+(?:an?\s+)?(?:order|contract|project)", re.IGNORECASE), 1.20, 66),
    "Capex / Plant Expansion": (re.compile(r"(?:commercial production|capacity expansion|capex|new plant)", re.IGNORECASE), 1.10, 62),
    "Dividend & Buyback": (re.compile(r"(?:interim dividend|final dividend|special dividend|buyback)", re.IGNORECASE), 0.85, 46),
    "Bonus / Split / Rights Issue": (re.compile(r"(?:sub-division|split of face value|bonus issue|bonus shares)", re.IGNORECASE), 0.70, 40),
}

RSS_FEEDS = [
    "https://beta.bseindia.com/rss-feed.html",
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/MCtopnews.xml"
]

# Ingest RSS
matched_news = {}
for url in RSS_FEEDS:
    try:
        f = feedparser.parse(url)
        for entry in f.entries[:30]:
            txt = f"{entry.get('title', '')} {entry.get('summary', '')}"
            for sym in NIFTY_UNIVERSE:
                if re.search(rf"\b{sym}\b", txt, re.IGNORECASE):
                    badge, mult, p1 = "⚡ Technical Baseline", 1.0, 50
                    for c_name, (rgx, m, p) in CATALYST_RULES.items():
                        if rgx.search(txt):
                            badge, mult, p1 = c_name, m, p
                            break
                    matched_news[sym] = {"catalyst": badge, "mult": mult, "p1": p1}
    except Exception:
        pass

print("Fetching universe market data...")
dl_list = [f"{s}.NS" for s in NIFTY_UNIVERSE]
raw_market = yf.download(dl_list, period="1y", interval="1d", group_by="ticker", auto_adjust=True)

candidate_rows = []
for sym in NIFTY_UNIVERSE:
    ns_sym = f"{sym}.NS"
    if ns_sym not in raw_market.columns.levels[0]:
        continue

    df = raw_market[ns_sym].dropna()
    if len(df) < 50:
        continue

    c, h, l, v = df["Close"], df["High"], df["Low"], df["Volume"]
    cmp = float(c.iloc[-1])
    d200 = float(c.rolling(200).mean().iloc[-1]) if len(c) >= 200 else float(c.mean())
    dist_200 = ((cmp - d200) / d200) * 100.0

    p_5d = float(c.iloc[-6]) if len(c) >= 6 else float(c.iloc[0])
    runup_5d = ((cmp - p_5d) / p_5d) * 100.0

    v_latest = float(v.iloc[-1])
    v_20d = float(v.rolling(20).mean().iloc[-1])
    vol_surge = round(v_latest / v_20d, 2) if v_20d > 0 else 1.0

    day_range = float(h.iloc[-1] - l.iloc[-1])
    close_pos = round(((cmp - float(l.iloc[-1])) / day_range) * 100.0, 1) if day_range > 0 else 50.0

    cat_data = matched_news.get(sym, {"catalyst": "⚡ Technical Baseline", "mult": 1.0, "p1": 50})

    s_runup = np.clip(100.0 - (runup_5d * 10.0), -50.0, 50.0)
    s_vol = (vol_surge * (close_pos - 50.0) * 0.4)
    s_trend = 15.0 if dist_200 > 0 else -15.0
    final_score = round(((s_runup * 0.35) + (s_vol * 0.45) + (s_trend * 0.20)) * cat_data["mult"], 1)

    prob_adj = 0.0
    if runup_5d <= 2.5: prob_adj += 8.0
    elif runup_5d >= 7.5: prob_adj -= 18.0

    if vol_surge >= 2.0 and close_pos >= 65.0: prob_adj += 10.0
    elif close_pos <= 35.0: prob_adj -= 14.0

    prob_adj += 4.0 if dist_200 > 0 else -8.0
    prob_1w = int(np.clip(cat_data["p1"] + prob_adj, 10, 92))

    if final_score >= 18.0 and prob_1w >= 65:
        outlook = "BULLISH"
    elif final_score <= -15.0 or (runup_5d >= 7.5 and close_pos <= 40.0) or prob_1w <= 40:
        outlook = "BEARISH_FADE"
    else:
        outlook = "NEUTRAL"

    if outlook != "NEUTRAL":
        abs_target = 4.5 if outlook == "BULLISH" else 4.0
        drift_velocity = 1.0 + (vol_surge * 0.25) if prob_1w >= 65 else 0.65
        est_days = int(np.clip(round(abs_target / drift_velocity), 3, 15))

        if est_days <= 5:
            horizon_lbl = f"⚡ Fast Drift ({est_days} Sessions)"
        elif est_days <= 10:
            horizon_lbl = f"🎯 Tactical Swing ({est_days} Sessions)"
        else:
            horizon_lbl = f"⏳ Positional Unlock ({est_days} Sessions)"

        candidate_rows.append({
            "Ticker": sym, "CMP": cmp, "Score": final_score,
            "Outlook": outlook, "Catalyst": cat_data["catalyst"],
            "Horizon": horizon_lbl, "Target_Days": est_days
        })

c_df = pd.DataFrame(candidate_rows)
new_predictions = []

if not c_df.empty:
    top_bullish = c_df[c_df["Outlook"] == "BULLISH"].sort_values(by="Score", ascending=False).head(3)
    top_bearish = c_df[c_df["Outlook"] == "BEARISH_FADE"].sort_values(by="Score", ascending=True).head(3)
    selected_today = pd.concat([top_bullish, top_bearish])

    for _, row in selected_today.iterrows():
        p_id = f"{today_str}_{row['Ticker']}"
        if p_id in existing_preds:
            continue

        is_bull = (row["Outlook"] == "BULLISH")
        new_predictions.append({
            "Prediction_ID": p_id,
            "Date": timestamp_str,
            "Ticker": row["Ticker"],
            "Active_Catalyst": row["Catalyst"],
            "CMP_At_Prediction": round(row["CMP"], 2),
            "Predicted_Outlook": row["Outlook"],
            "Recommended_Action": "🟢 BUY / ACCUMULATE" if is_bull else "🔴 SHORT / FADE (SELL)",
            "Holding_Horizon": row["Horizon"],
            "Target_Days": row["Target_Days"],
            "Trigger_Type": "SCHEDULED_CRON_AUTO",
            "Confidence": "High",
            "Target_Return_Pct": 4.5 if is_bull else -4.0,
            "Stop_Loss_Pct": -2.5 if is_bull else 2.5,
            "Days_Elapsed": 0,
            "Current_CMP": round(row["CMP"], 2),
            "Realized_Return_Pct": 0.0,
            "Outcome_Status": "PENDING"
        })

if new_predictions:
    print(f"Logging {len(new_predictions)} automated daily predictions...")
    new_df = pd.DataFrame(new_predictions)
    ledger = pd.concat([ledger, new_df], ignore_index=True) if not ledger.empty else new_df
    changed = True

if changed:
    ledger.to_csv(file_path, index=False)
    print("CSV updated and saved successfully.")
else:
    print("No changes required.")
