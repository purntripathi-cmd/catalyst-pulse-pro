import os
import pandas as pd
import yfinance as yf

file_path = "catalyst_prediction_ledger.csv"
if not os.path.exists(file_path):
    print("No ledger found. Skipping.")
    exit(0)

ledger = pd.read_csv(file_path)
pending = ledger[ledger["Outcome_Status"] == "PENDING"]
if pending.empty:
    print("No pending predictions to audit.")
    exit(0)

clean_tickers = [str(t).replace(".NS", "").strip() for t in pending["Ticker"].unique()]
ticker_list = [f"{t}.NS" for t in clean_tickers]

print(f"Auditing tickers: {ticker_list}")
raw = yf.download(ticker_list, period="1mo", interval="1d", group_by="ticker", auto_adjust=True)

changed = False
for idx, row in ledger.iterrows():
    if row["Outcome_Status"] != "PENDING":
        continue

    clean_t = str(row["Ticker"]).replace(".NS", "").strip()
    ns_sym = f"{clean_t}.NS"

    target_col = None
    if hasattr(raw, "columns") and hasattr(raw.columns, "levels"):
        if ns_sym in raw.columns.levels[0]:
            target_col = ns_sym
        elif clean_t in raw.columns.levels[0]:
            target_col = clean_t

    if target_col:
        hist = raw[target_col]["Close"].dropna()
        if not hist.empty:
            curr_p = float(hist.iloc[-1])
            init_p = float(row["CMP_At_Prediction"])

            pred_type = str(row.get("Predicted_Outlook", "BEARISH_FADE")).upper()
            is_bullish = ("BULLISH" in pred_type)

            # Direction-Aware Return
            if is_bullish:
                ret_pct = round(((curr_p - init_p) / init_p) * 100.0, 2)
            else:
                ret_pct = round(((init_p - curr_p) / init_p) * 100.0, 2)

            # Strips IST string and normalizes to avoid timezone-naive comparison crash
            raw_date_str = str(row["Date"]).replace(" IST", "").strip()
            pred_date = pd.to_datetime(raw_date_str).tz_localize(None)
            post = hist[hist.index.tz_localize(None) >= pred_date.normalize()]
            days = max(0, len(post) - 1)

            ledger.at[idx, "Current_CMP"] = curr_p
            ledger.at[idx, "Realized_Return_Pct"] = ret_pct
            ledger.at[idx, "Days_Elapsed"] = days
            changed = True

            target_days = int(row.get("Target_Days", 8)) if not pd.isna(row.get("Target_Days")) else 8
            max_allowed_days = target_days + 2

            if days >= 1:
                if is_bullish:
                    if ret_pct >= float(row.get("Target_Return_Pct", 4.5)):
                        ledger.at[idx, "Outcome_Status"] = "SUCCESS"
                    elif ret_pct <= float(row.get("Stop_Loss_Pct", -2.5)) or days >= max_allowed_days:
                        ledger.at[idx, "Outcome_Status"] = "SUCCESS" if ret_pct > 0 else "FAILED"
                else:
                    target_gain = abs(float(row.get("Target_Return_Pct", -4.0)))
                    stop_loss_hit = ret_pct <= -abs(float(row.get("Stop_Loss_Pct", 2.5)))
                    if ret_pct >= target_gain:
                        ledger.at[idx, "Outcome_Status"] = "SUCCESS"
                    elif stop_loss_hit or days >= max_allowed_days:
                        ledger.at[idx, "Outcome_Status"] = "SUCCESS" if ret_pct > 0 else "FAILED"

if changed:
    ledger.to_csv(file_path, index=False)
    print("Ledger audited and written to CSV successfully.")
else:
    print("No changes required.")
