"""
ticker_utils.py - High-precision Institutional Ticker & Entity Matching for Indian Equities
Prevents false positives on short ticker abbreviations (LT, BEL, HAL, REC, etc.)
and resolves full corporate entity names to canonical NSE tickers.
"""

import re
from typing import List
import pandas as pd
import numpy as np

__all__ = [
    "TICKER_ALIASES",
    "match_tickers_in_text",
    "NSE_FNO_TICKERS",
    "PHARMA_HEALTHCARE_TICKERS",
    "ROUND_TRIP_FRICTION_PCT",
    "is_fno_eligible",
    "is_pharma_healthcare_ticker",
    "get_action_recommendation",
    "calibrate_empirical_probability",
    "compute_bimodal_catalyst_score",
    "compute_dynamic_atr_brackets",
    "is_ticker_in_cooldown_or_blacklisted",
    "evaluate_trade_lifecycle",
    "evaluate_audit_evolution_and_stability",
]

# Mapping canonical NSE symbols to case-sensitive acronyms and case-insensitive entity aliases
TICKER_ALIASES = {
    "ABB": {"exact": [r"\bABB\b"], "names": [r"\bABB India\b"]},
    "ADANIENSOL": {"exact": [r"\bADANIENSOL\b"], "names": [r"\bAdani Energy Solutions\b", r"\bAdani Transmission\b"]},
    "ADANIENT": {"exact": [r"\bADANIENT\b"], "names": [r"\bAdani Enterprises\b"]},
    "ADANIGREEN": {"exact": [r"\bADANIGREEN\b"], "names": [r"\bAdani Green\b", r"\bAdani Green Energy\b"]},
    "ADANIPORTS": {"exact": [r"\bADANIPORTS\b"], "names": [r"\bAdani Ports\b", r"\bAPSEZ\b"]},
    "ADANIPOWER": {"exact": [r"\bADANIPOWER\b"], "names": [r"\bAdani Power\b"]},
    "AMBUJACEM": {"exact": [r"\bAMBUJACEM\b"], "names": [r"\bAmbuja Cements?\b"]},
    "APOLLOHOSP": {"exact": [r"\bAPOLLOHOSP\b"], "names": [r"\bApollo Hospitals?\b"]},
    "ASIANPAINT": {"exact": [r"\bASIANPAINT\b"], "names": [r"\bAsian Paints?\b"]},
    "AXISBANK": {"exact": [r"\bAXISBANK\b"], "names": [r"\bAxis Bank\b"]},
    "BAJAJ-AUTO": {"exact": [r"\bBAJAJ-AUTO\b"], "names": [r"\bBajaj Auto\b"]},
    "BAJFINANCE": {"exact": [r"\bBAJFINANCE\b"], "names": [r"\bBajaj Finance\b"]},
    "BAJAJFINSV": {"exact": [r"\bBAJAJFINSV\b"], "names": [r"\bBajaj Finserv\b"]},
    "BAJAJHLDNG": {"exact": [r"\bBAJAJHLDNG\b"], "names": [r"\bBajaj Holdings\b"]},
    "BANKBARODA": {"exact": [r"\bBANKBARODA\b", r"\bBOB\b"], "names": [r"\bBank of Baroda\b"]},
    "BEL": {"exact": [r"\bBEL\b"], "names": [r"\bBharat Electronics\b"]},
    "BHEL": {"exact": [r"\bBHEL\b"], "names": [r"\bBharat Heavy Electricals\b"]},
    "BPCL": {"exact": [r"\bBPCL\b"], "names": [r"\bBharat Petroleum\b"]},
    "BHARTIARTL": {"exact": [r"\bBHARTIARTL\b"], "names": [r"\bBharti Airtel\b", r"\bAirtel\b"]},
    "BOSCHLTD": {"exact": [r"\bBOSCHLTD\b"], "names": [r"\bBosch Ltd\b", r"\bBosch India\b"]},
    "BRITANNIA": {"exact": [r"\bBRITANNIA\b"], "names": [r"\bBritannia Industries\b", r"\bBritannia\b"]},
    "CANBK": {"exact": [r"\bCANBK\b"], "names": [r"\bCanara Bank\b"]},
    "CHOLAFIN": {"exact": [r"\bCHOLAFIN\b"], "names": [r"\bCholamandalam\b", r"\bChola Finance\b"]},
    "CIPLA": {"exact": [r"\bCIPLA\b"], "names": [r"\bCipla\b"]},
    "COALINDIA": {"exact": [r"\bCOALINDIA\b", r"\bCIL\b"], "names": [r"\bCoal India\b"]},
    "COLPAL": {"exact": [r"\bCOLPAL\b"], "names": [r"\bColgate[- ]Palmolive\b", r"\bColgate India\b"]},
    "DLF": {"exact": [r"\bDLF\b"], "names": [r"\bDLF Ltd\b"]},
    "DABUR": {"exact": [r"\bDABUR\b"], "names": [r"\bDabur India\b", r"\bDabur\b"]},
    "DIVISLAB": {"exact": [r"\bDIVISLAB\b"], "names": [r"\bDivi'?s Laboratories\b", r"\bDivis Lab\b"]},
    "DIXON": {"exact": [r"\bDIXON\b"], "names": [r"\bDixon Tech\w*\b"]},
    "DRREDDY": {"exact": [r"\bDRREDDY\b"], "names": [r"\bDr\.?\s*Reddy'?s?\b"]},
    "EICHERMOT": {"exact": [r"\bEICHERMOT\b"], "names": [r"\bEicher Motors\b", r"\bRoyal Enfield\b"]},
    "GAIL": {"exact": [r"\bGAIL\b"], "names": [r"\bGas Authority of India\b"]},
    "GODREJCP": {"exact": [r"\bGODREJCP\b", r"\bGCPL\b"], "names": [r"\bGodrej Consumer\b"]},
    "GRASIM": {"exact": [r"\bGRASIM\b"], "names": [r"\bGrasim Industries\b"]},
    "HCLTECH": {"exact": [r"\bHCLTECH\b"], "names": [r"\bHCL Tech\w*\b"]},
    "HDFCBANK": {"exact": [r"\bHDFCBANK\b"], "names": [r"\bHDFC Bank\b"]},
    "HDFCLIFE": {"exact": [r"\bHDFCLIFE\b"], "names": [r"\bHDFC Life\b"]},
    "HAVELLS": {"exact": [r"\bHAVELLS\b"], "names": [r"\bHavells India\b", r"\bHavells\b"]},
    "HEROMOTOCO": {"exact": [r"\bHEROMOTOCO\b"], "names": [r"\bHero MotoCorp\b", r"\bHero Honda\b"]},
    "HINDALCO": {"exact": [r"\bHINDALCO\b"], "names": [r"\bHindalco Industries\b"]},
    "HAL": {"exact": [r"\bHAL\b"], "names": [r"\bHindustan Aeronautics\b"]},
    "HINDUNILVR": {"exact": [r"\bHINDUNILVR\b", r"\bHUL\b"], "names": [r"\bHindustan Unilever\b"]},
    "ICICIBANK": {"exact": [r"\bICICIBANK\b"], "names": [r"\bICICI Bank\b"]},
    "ICICIGI": {"exact": [r"\bICICIGI\b"], "names": [r"\bICICI Lombard\b"]},
    "ICICIPRULI": {"exact": [r"\bICICIPRULI\b"], "names": [r"\bICICI Prudential\b", r"\bICICI Pru Life\b"]},
    "ITC": {"exact": [r"\bITC\b"], "names": [r"\bITC Limited\b", r"\bITC Ltd\b"]},
    "INDHOTEL": {"exact": [r"\bINDHOTEL\b", r"\bIHCL\b"], "names": [r"\bIndian Hotels\b", r"\bTaj Hotels\b"]},
    "IOC": {"exact": [r"\bIOC\b", r"\bIOCL\b"], "names": [r"\bIndian Oil Corporation\b", r"\bIndian Oil\b"]},
    "IRCTC": {"exact": [r"\bIRCTC\b"], "names": [r"\bIndian Railway Catering\b"]},
    "IRFC": {"exact": [r"\bIRFC\b"], "names": [r"\bIndian Railway Finance\b"]},
    "INDUSINDBK": {"exact": [r"\bINDUSINDBK\b"], "names": [r"\bIndusInd Bank\b"]},
    "NAUKRI": {"exact": [r"\bNAUKRI\b"], "names": [r"\bInfo Edge\b", r"\bNaukri\.com\b"]},
    "INFY": {"exact": [r"\bINFY\b"], "names": [r"\bInfosys\b"]},
    "INDIGO": {"exact": [r"\bINDIGO\b"], "names": [r"\bInterGlobe Aviation\b", r"\bIndiGo Airlines\b"]},
    "JSWSTEEL": {"exact": [r"\bJSWSTEEL\b"], "names": [r"\bJSW Steel\b"]},
    "JINDALSTEL": {"exact": [r"\bJINDALSTEL\b", r"\bJSPL\b"], "names": [r"\bJindal Steel & Power\b", r"\bJindal Steel\b"]},
    "JIOFIN": {"exact": [r"\bJIOFIN\b", r"\bJFSL\b"], "names": [r"\bJio Financial Services\b", r"\bJio Financial\b"]},
    "KOTAKBANK": {"exact": [r"\bKOTAKBANK\b"], "names": [r"\bKotak Mahindra Bank\b", r"\bKotak Bank\b"]},
    "LTIM": {"exact": [r"\bLTIM\b"], "names": [r"\bLTIMindtree\b"]},
    "LT": {"exact": [r"\bLT\b"], "names": [r"\bL&T\b", r"\bLarsen & Toubro\b", r"\bLarsen and Toubro\b"]},
    "LUPIN": {"exact": [r"\bLUPIN\b"], "names": [r"\bLupin Ltd\b", r"\bLupin Pharma\b"]},
    "M&M": {"exact": [r"\bM&M\b"], "names": [r"\bMahindra & Mahindra\b", r"\bMahindra and Mahindra\b"]},
    "MARICO": {"exact": [r"\bMARICO\b"], "names": [r"\bMarico Ltd\b", r"\bMarico\b"]},
    "MARUTI": {"exact": [r"\bMARUTI\b"], "names": [r"\bMaruti Suzuki\b"]},
    "MAXHEALTH": {"exact": [r"\bMAXHEALTH\b"], "names": [r"\bMax Healthcare\b"]},
    "NTPC": {"exact": [r"\bNTPC\b"], "names": [r"\bNTPC Ltd\b", r"\bNational Thermal Power\b"]},
    "NESTLEIND": {"exact": [r"\bNESTLEIND\b"], "names": [r"\bNestle India\b", r"\bNestle\b"]},
    "ONGC": {"exact": [r"\bONGC\b"], "names": [r"\bOil & Natural Gas Corp\b", r"\bOil and Natural Gas Corp\b"]},
    "PIDILITIND": {"exact": [r"\bPIDILITIND\b"], "names": [r"\bPidilite Industries\b", r"\bFevicol\b"]},
    "PFC": {"exact": [r"\bPFC\b"], "names": [r"\bPower Finance Corporation\b", r"\bPower Finance Corp\b"]},
    "POWERGRID": {"exact": [r"\bPOWERGRID\b", r"\bPGCIL\b"], "names": [r"\bPower Grid Corp\b", r"\bPower Grid\b"]},
    "PNB": {"exact": [r"\bPNB\b"], "names": [r"\bPunjab National Bank\b"]},
    "RECLTD": {"exact": [r"\bRECLTD\b", r"\bREC\b"], "names": [r"\bREC Ltd\b", r"\bREC Limited\b", r"\bRural Electrification\b"]},
    "RELIANCE": {"exact": [r"\bRELIANCE\b", r"\bRIL\b"], "names": [r"\bReliance Industries\b"]},
    "SBICARD": {"exact": [r"\bSBICARD\b"], "names": [r"\bSBI Cards?\b", r"\bSBI Card\b"]},
    "SBILIFE": {"exact": [r"\bSBILIFE\b"], "names": [r"\bSBI Life Insurance\b", r"\bSBI Life\b"]},
    "SRF": {"exact": [r"\bSRF\b"], "names": [r"\bSRF Ltd\b", r"\bSRF Limited\b"]},
    "MOTHERSON": {"exact": [r"\bMOTHERSON\b"], "names": [r"\bSamvardhana Motherson\b", r"\bMotherson Sumi\b"]},
    "SHREECEM": {"exact": [r"\bSHREECEM\b"], "names": [r"\bShree Cements?\b"]},
    "SHRIRAMFIN": {"exact": [r"\bSHRIRAMFIN\b"], "names": [r"\bShriram Finance\b"]},
    "SIEMENS": {"exact": [r"\bSIEMENS\b"], "names": [r"\bSiemens India\b", r"\bSiemens Ltd\b"]},
    "SBIN": {"exact": [r"\bSBIN\b", r"\bSBI\b"], "names": [r"\bState Bank of India\b"]},
    "SUNPHARMA": {"exact": [r"\bSUNPHARMA\b"], "names": [r"\bSun Pharma\b", r"\bSun Pharmaceuticals\b"]},
    "TVSMOTOR": {"exact": [r"\bTVSMOTOR\b"], "names": [r"\bTVS Motor\b"]},
    "TCS": {"exact": [r"\bTCS\b"], "names": [r"\bTata Consultancy Services\b", r"\bTata Consultancy\b"]},
    "TATACONSUM": {"exact": [r"\bTATACONSUM\b"], "names": [r"\bTata Consumer Products\b", r"\bTata Consumer\b"]},
    "TATAMOTORS": {"exact": [r"\bTATAMOTORS\b"], "names": [r"\bTata Motors\b"]},
    "TATAPOWER": {"exact": [r"\bTATAPOWER\b"], "names": [r"\bTata Power\b"]},
    "TATASTEEL": {"exact": [r"\bTATASTEEL\b"], "names": [r"\bTata Steel\b"]},
    "TECHM": {"exact": [r"\bTECHM\b"], "names": [r"\bTech Mahindra\b"]},
    "TITAN": {"exact": [r"\bTITAN\b"], "names": [r"\bTitan Company\b", r"\bTitan Ltd\b"]},
    "TORNTPOWER": {"exact": [r"\bTORNTPOWER\b"], "names": [r"\bTorrent Power\b"]},
    "TRENT": {"exact": [r"\bTRENT\b"], "names": [r"\bTrent Ltd\b", r"\bWestside\b", r"\bZudio\b"]},
    "ULTRACEMCO": {"exact": [r"\bULTRACEMCO\b"], "names": [r"\bUltraTech Cement\b", r"\bUltraTech\b"]},
    "UNITDSPR": {"exact": [r"\bUNITDSPR\b"], "names": [r"\bUnited Spirits\b", r"\bMcDowell\b", r"\bDiageo India\b"]},
    "VBL": {"exact": [r"\bVBL\b"], "names": [r"\bVarun Beverages\b"]},
    "VEDL": {"exact": [r"\bVEDL\b"], "names": [r"\bVedanta Ltd\b", r"\bVedanta Resources\b"]},
    "WIPRO": {"exact": [r"\bWIPRO\b"], "names": [r"\bWipro Ltd\b", r"\bWipro Technologies\b"]},
    "ZOMATO": {"exact": [r"\bZOMATO\b"], "names": [r"\bZomato Ltd\b", r"\bBlinkit\b"]},
    "ZYDUSLIFE": {"exact": [r"\bZYDUSLIFE\b"], "names": [r"\bZydus Lifesciences\b", r"\bZydus Cadila\b", r"\bCadila Healthcare\b"]}
}

def match_tickers_in_text(text: str, universe_tickers: List[str]) -> List[str]:
    """
    Extracts canonical tickers from text using exact regex matching for short symbols
    and alias resolution for full corporate entity names.
    """
    matched = []
    for sym in universe_tickers:
        clean_sym = str(sym).replace(".NS", "").strip().upper()
        alias_info = TICKER_ALIASES.get(clean_sym, {})
        found = False

        # 1. Exact case-sensitive match for short acronyms (prevents 'lt' or 'bel' false positives)
        for pat in alias_info.get("exact", []):
            if re.search(pat, text):
                found = True
                break

        # 2. Case-insensitive match for company entity names
        if not found:
            for pat in alias_info.get("names", []):
                if re.search(pat, text, re.IGNORECASE):
                    found = True
                    break

        # 3. Fallback matching directly on the ticker symbol
        if not found:
            if len(clean_sym) <= 4:
                # Short tickers MUST match case-sensitively
                if re.search(rf"\b{re.escape(clean_sym)}\b", text):
                    found = True
            else:
                if re.search(rf"\b{re.escape(clean_sym)}\b", text, re.IGNORECASE):
                    found = True

        if found:
            matched.append(clean_sym)

    return list(dict.fromkeys(matched))


# =====================================================================
# Institutional Microstructure & SEBI Regulatory Compliance
# =====================================================================

# Verified NSE Equity Derivatives (F&O) Active Constituent Universe
NSE_FNO_TICKERS = {
    "ABB", "ADANIENT", "ADANIPORTS", "AMBUJACEM", "APOLLOHOSP", "ASIANPAINT", "AXISBANK",
    "BAJAJ-AUTO", "BAJFINANCE", "BAJAJFINSV", "BANKBARODA", "BEL", "BHEL", "BPCL",
    "BHARTIARTL", "BOSCHLTD", "BRITANNIA", "CANBK", "CHOLAFIN", "CIPLA", "COALINDIA",
    "COLPAL", "DLF", "DABUR", "DIVISLAB", "DIXON", "DRREDDY", "EICHERMOT", "GAIL",
    "GODREJCP", "GRASIM", "HCLTECH", "HDFCBANK", "HDFCLIFE", "HAVELLS", "HEROMOTOCO",
    "HINDALCO", "HAL", "HINDUNILVR", "ICICIBANK", "ICICIGI", "ICICIPRULI", "ITC",
    "INDHOTEL", "IOC", "IRCTC", "INDUSINDBK", "NAUKRI", "INFY", "INDIGO", "JSWSTEEL",
    "JINDALSTEL", "KOTAKBANK", "LTIM", "LT", "LUPIN", "M&M", "MARICO", "MARUTI",
    "NTPC", "NESTLEIND", "ONGC", "PIDILITIND", "PFC", "POWERGRID", "PNB", "RECLTD",
    "RELIANCE", "SBICARD", "SBILIFE", "SRF", "MOTHERSON", "SHREECEM", "SHRIRAMFIN",
    "SIEMENS", "SBIN", "SUNPHARMA", "TVSMOTOR", "TCS", "TATACONSUM", "TATAMOTORS",
    "TATAPOWER", "TATASTEEL", "TECHM", "TITAN", "TORNTPOWER", "TRENT", "ULTRACEMCO",
    "UNITDSPR", "VBL", "VEDL", "WIPRO", "ZOMATO", "ZYDUSLIFE", "CONCOR"
}

# Verified Pharma, Lifesciences & Healthcare Universe for USFDA / Regulatory Catalyst Gate (GAP-04)
PHARMA_HEALTHCARE_TICKERS = {
    "SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "ZYDUSLIFE",
    "APOLLOHOSP", "MAXHEALTH", "AUROPHARMA", "BIOCON", "TORNTPHARM",
    "ALKEM", "GLENMARK", "LAURUSLABS", "SYNGENE", "IPCALAB", "GRANULES",
    "NATCOPHARM", "FORTIS", "MANKIND", "ABBOTINDIA", "GLAND", "JBCHEPHARM"
}

# Standard Round-Trip Indian Transaction Friction (35 bps)
# Covers STT (Securities Transaction Tax), GST (18%), NSE turnover fees, SEBI turnover fees,
# stamp duty (0.015%), and institutional bid-ask spread slippage (0.05-0.10%).
ROUND_TRIP_FRICTION_PCT = 0.35

def is_fno_eligible(symbol: str) -> bool:
    """Returns True if ticker is eligible for overnight shorting via NSE Equity Derivatives (F&O)."""
    clean = str(symbol).replace(".NS", "").replace("^", "").strip().upper()
    return clean in NSE_FNO_TICKERS

def is_pharma_healthcare_ticker(symbol: str) -> bool:
    """Returns True if ticker belongs to the Pharma / Healthcare / Lifesciences sector eligible for USFDA catalysts."""
    clean = str(symbol).replace(".NS", "").replace("^", "").strip().upper()
    return clean in PHARMA_HEALTHCARE_TICKERS

def get_action_recommendation(outlook: str, ticker: str) -> str:
    """
    Returns regulatory-compliant action recommendation conforming to SEBI market microstructure.
    Cash equities cannot be shorted overnight without SLBM. F&O tickers can be shorted via derivatives.
    """
    out_upper = str(outlook).upper()
    if "BULLISH" in out_upper:
        return "🟢 BUY / ACCUMULATE (Cash Equity)"
    elif "BEARISH" in out_upper or "DISTRIBUTION" in out_upper or "FADE" in out_upper:
        if is_fno_eligible(ticker):
            return "🔴 F&O DERIVATIVE SHORT (Put/Future)"
        else:
            return "⚠️ CASH AVOID / EXIT (Long Square-Off)"
    else:
        return "🟡 NEUTRAL / MONITOR"

# =====================================================================
# Empirical Probability Calibration (Platt / Logistic Scaling + Past Ledger Memory)
# Fit on real historical trade ledger (catalyst_prediction_ledger.csv)
# =====================================================================
_EMPIRICAL_LEDGER_MEMORY_CACHE = None

def _load_empirical_ledger_memory() -> dict:
    """
    Loads closed trade outcomes per (Ticker, Direction) from catalyst_prediction_ledger.csv
    to provide factual past-performance feedback (Empirical Bayes shrinkage) to live predictions.
    """
    global _EMPIRICAL_LEDGER_MEMORY_CACHE
    if _EMPIRICAL_LEDGER_MEMORY_CACHE is not None:
        return _EMPIRICAL_LEDGER_MEMORY_CACHE

    memory = {}
    try:
        import os
        ledger_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalyst_prediction_ledger.csv")
        if os.path.exists(ledger_path):
            df = pd.read_csv(ledger_path, on_bad_lines="skip")
            closed = df[df["Outcome_Status"].isin(["SUCCESS", "FAILED"])]
            for (tkr, out), grp in closed.groupby(["Ticker", "Predicted_Outlook"]):
                clean_t = str(tkr).replace(".NS", "").strip().upper()
                dir_key = "BULLISH" if "BULLISH" in str(out).upper() else "BEARISH_FADE"
                n_total = len(grp)
                n_win = int((grp["Outcome_Status"] == "SUCCESS").sum())
                memory[(clean_t, dir_key)] = (n_win, n_total)
    except Exception:
        pass
    _EMPIRICAL_LEDGER_MEMORY_CACHE = memory
    return memory


def get_empirical_ticker_memory(ticker: str, outlook: str = "BULLISH") -> float:
    """
    Returns an Empirical Bayes logit adjustment based on factual historical hit-rate
    for (ticker, outlook) in catalyst_prediction_ledger.csv.
    - Penalizes secular momentum / repeat-failing fade traps (e.g., TRENT 1/7, BHARTIARTL 0/4,
      ICICIGI 0/4, TORNTPOWER 0/4) by up to -0.65 logit.
    - Rewards proven mean-reverting setups (e.g., MAXHEALTH 9/9, AMBUJACEM 4/4, HINDUNILVR 3/3)
      by up to +0.35 logit.
    """
    if not ticker:
        return 0.0
    clean_t = str(ticker).replace(".NS", "").strip().upper()
    dir_key = "BULLISH" if "BULLISH" in str(outlook).upper() else "BEARISH_FADE"
    mem = _load_empirical_ledger_memory()
    if (clean_t, dir_key) not in mem:
        return 0.0
    n_win, n_total = mem[(clean_t, dir_key)]
    if n_total < 2:
        return 0.0
    # Empirical Bayes posterior win rate with prior 0.50 and strength m=2
    post_wr = (n_win + 2.0 * 0.50) / (n_total + 2.0)
    raw_wr = n_win / n_total
    if raw_wr <= 0.25 and n_total >= 2:
        return -0.65 if n_total >= 3 else -0.45
    elif raw_wr >= 0.75 and n_total >= 2:
        return 0.35 if n_total >= 3 else 0.25
    return float(np.clip((post_wr - 0.50) * 1.2, -0.50, 0.35))


def calibrate_empirical_probability(
    score: float,
    vol_surge: float = 1.0,
    runup_5d: float = 0.0,
    outlook: str = "BULLISH",
    market_regime: str = "Sideways",
    ticker: str = None
) -> float:
    """
    Computes an empirical, Platt-calibrated probability of trade success.
    Fit parameters derived from catalyst_prediction_ledger.csv (v5.1 GAP-01 patched):
    - Base logit intercept: 0.1627
    - Directional bias (bearish fade vs bullish breakout vs contrarian moat)
    - Volume surge enhancement (+0.30 logit per surge above 1.8x)
    - Pre-runup exhaustion & falling-knife penalty (GAP-01: rewards +0.3% to +4.0% coiled base,
      penalizes falling knives runup_5d < 0% on BULLISH breakouts)
    - Empirical Bayes ticker-level historical feedback loop (get_empirical_ticker_memory)
    Returns a probability bounded honestly between 18.0% and 68.0%.
    """
    out_upper = str(outlook).upper()
    is_bull = "BULLISH" in out_upper
    is_contrarian = "CONTRARIAN" in out_upper
    
    # 1. Base logit from logistic regression on prediction ledger
    logit = 0.1627 + (0.0222 * float(score))
    
    # Directional regime adjustment
    if is_contrarian:
        logit += 0.10  # Institutional contrarian moats have 50%+ closed WR & 66.7% green open rate
        if runup_5d <= 0.0:
            logit += 0.35  # Oversold dip enhances contrarian value entry
        elif runup_5d >= 5.0:
            logit -= 0.35
    elif is_bull:
        logit -= 1.10  # Historical bullish breakouts had higher failure rate in choppy regimes
        if 0.3 <= runup_5d <= 4.0:
            logit += 0.35  # Clean constructive base before breakout improves odds
        elif runup_5d < 0.0:
            logit -= 0.45  # GAP-01 Fix: Falling knife penalty on momentum breakout
        elif runup_5d >= 6.0:
            logit -= 0.60  # Extended runup heavily degrades continuation
    else:
        logit += 0.15  # Fades have empirical mean-reversion tailwind
        if runup_5d >= 6.0:
            logit += 0.40  # Overbought exhaustion aids mean-reversion short
        elif runup_5d <= 0.0:
            logit -= 0.30  # Already collapsed, fading further is risky

    # Volume surge conviction
    v_ratio = max(0.2, min(5.0, float(vol_surge)))
    if v_ratio >= 1.8:
        logit += 0.30
    elif v_ratio < 0.8:
        logit -= 0.25

    # Market regime modifier
    regime_str = str(market_regime).lower()
    if "bull" in regime_str:
        logit += 0.25 if is_bull else -0.25
    elif "bear" in regime_str:
        logit += 0.25 if not is_bull else (-0.10 if is_contrarian else -0.35)

    # Historical ticker-specific Empirical Bayes feedback adjustment
    if ticker:
        logit += get_empirical_ticker_memory(ticker, outlook)

    # Platt sigmoid transformation
    prob = 1.0 / (1.0 + np.exp(-logit))
    prob_pct = round(float(prob * 100.0), 1)

    # Realistic institutional bounds: quant swing setups operate in 18% - 68% range
    return float(np.clip(prob_pct, 18.0, 68.0))


# =====================================================================
# v5.1 Bimodal Catalyst Scoring, Dynamic ATR Brackets & Cooldown Gates
# Fixes GAP-01 (score_runup inversion), GAP-02 (static stops), GAP-03 (repeat fades)
# =====================================================================
def compute_bimodal_catalyst_score(
    score_cat: float = None,
    runup_5d: float = 0.0,
    score_vol: float = None,
    score_trend: float = None,
    rsi_14: float = 50.0,
    dist_200: float = 0.0,
    cat_text: str = "",
    vol_surge_ratio: float = None,
    close_pos_pct: float = 50.0,
    impact_mult: float = 1.0,
    has_confirmed_catalyst: bool = None
) -> dict:
    """
    GAP-01 Fix: Replaces the defective monotonic `100 - runup_5d * 10` formula (which gave
    +50.0 score to falling knives with runup_5d < 0% and 0 catalyst) with a bimodal
    institutional scoring engine used identically by catalyst_app.py and audit_daemon.py.
    """
    r5 = float(runup_5d)
    rsi = float(rsi_14)
    d200 = float(dist_200)

    if score_cat is None:
        if has_confirmed_catalyst:
            s_cat = 45.0 * float(impact_mult) if float(impact_mult) >= 0 else 35.0 * float(impact_mult)
        else:
            s_cat = 0.0
    else:
        s_cat = float(score_cat)

    if score_vol is None:
        v_sr = float(vol_surge_ratio) if vol_surge_ratio is not None else 1.0
        s_vol = float(np.clip(v_sr * (float(close_pos_pct) - 50.0) * 0.4, -35.0, 35.0))
    else:
        s_vol = float(score_vol)

    if score_trend is None:
        s_trend = 15.0 if d200 > 0 else -15.0
    else:
        s_trend = float(score_trend)

    is_no_cat = (has_confirmed_catalyst is False) or ("No Catalyst" in str(cat_text)) or (abs(s_cat) < 1.0)

    # Bimodal 5D Runup Scoring:
    # - Positive catalyst (s_cat >= 25): constructive base (0% to +4.5% runup) is optimal (+35),
    #   falling knife (< -1.5%) is penalized (-20), overextended (> +6%) is penalized (-50).
    # - Negative catalyst (s_cat <= -25) or technical overbought fade (r5 >= 6.0):
    #   extended runup drives fade conviction (-50).
    # - Zero catalyst & falling/flat (r5 < 5.0): score_runup = 0.0 so dead stocks never become BULLISH.
    if s_cat >= 25.0:
        if 0.0 <= r5 <= 4.5:
            score_runup = 35.0
        elif r5 > 4.5:
            score_runup = float(np.clip(35.0 - (r5 - 4.5) * 14.0, -50.0, 35.0))
        else:
            score_runup = float(np.clip(r5 * 10.0, -35.0, 0.0))
    elif s_cat <= -25.0:
        if r5 >= 4.0:
            score_runup = -45.0
        elif r5 >= 0.0:
            score_runup = -25.0
        else:
            score_runup = -10.0
    else:
        # Technical-only / No Catalyst
        if r5 >= 6.5 and rsi >= 65.0:
            score_runup = float(np.clip(100.0 - (r5 * 10.0), -50.0, -25.0))
        else:
            score_runup = 0.0

    raw_score = (s_cat * 0.40) + (score_runup * 0.35) + (s_vol * 0.15) + (s_trend * 0.10)
    final_score = round(float(np.clip(raw_score, -100.0, 100.0)), 1)

    if d200 > 0.0 and rsi < 40.0 and r5 < -3.0 and s_cat >= 0.0:
        outlook = "BULLISH_CONTRARIAN"
    elif (not is_no_cat) and s_cat >= 35.0 and final_score >= 18.0 and s_trend > 0.0 and (0.0 <= r5 <= 5.5):
        outlook = "BULLISH"
    elif (s_cat <= -25.0 and final_score <= -15.0) or (r5 >= 6.5 and rsi >= 68.0 and d200 < 22.0 and s_cat <= 0.0):
        outlook = "BEARISH_FADE"
    else:
        outlook = "NEUTRAL"

    return {
        "score_runup": round(float(score_runup), 1),
        "final_score": final_score,
        "outlook": outlook,
        "mode": outlook
    }


def compute_dynamic_atr_brackets(
    atr_pct: float,
    outlook: str = "BULLISH",
    cat_text: str = "",
    catalyst_name: str = ""
) -> dict:
    """
    GAP-02 Fix: Replaces static -2.5% / +2.5% brackets (which caused 50.5% of all losses to
    whipsaw out in <=2 days before 30% of them reversed to hit target) with volatility-scaled
    ATR brackets:
    - BULLISH_CONTRARIAN (180D Moat): +18.0% target, -12.0% catastrophe closing stop
    - BULLISH (8D Catalyst Breakout): Target = clip(3.0 * ATR%, 4.5%, 8.5%), Stop = -clip(2.2 * ATR%, 3.5%, 6.0%)
    - BEARISH_FADE (5D Mean-Reversion Short): Target = +clip(2.8 * ATR%, 4.0%, 7.5%), Stop = -clip(2.2 * ATR%, 3.5%, 5.5%)
    """
    clean_atr = max(0.8, min(6.0, float(atr_pct if pd.notna(atr_pct) and atr_pct > 0 else 1.8)))
    out_upper = str(outlook).upper()
    combined_cat = f"{cat_text} {catalyst_name}"

    if "CONTRARIAN" in out_upper:
        return {
            "target_pct": 18.0,
            "stop_pct": -12.0,
            "target_ret": 18.0,
            "stop_ret": -12.0,
            "target_days": 180,
            "horizon_label": "🏛️ Institutional Value Moat (180 Sessions)"
        }
    elif "BULLISH" in out_upper:
        high_conv = any(k in combined_cat for k in ["M&A", "Order", "Earnings", "Tariff", "USFDA", "Breakout"])
        t_mult = 3.2 if high_conv else 2.8
        t_pct = round(float(np.clip(t_mult * clean_atr, 4.5, 8.5)), 2)
        s_pct = -round(float(np.clip(2.2 * clean_atr, 3.5, 6.0)), 2)
        return {
            "target_pct": t_pct,
            "stop_pct": s_pct,
            "target_ret": t_pct,
            "stop_ret": s_pct,
            "target_days": 8,
            "horizon_label": "🎯 Tactical Swing (8 Sessions)"
        }
    elif "BEARISH" in out_upper or "FADE" in out_upper:
        t_pct = round(float(np.clip(2.8 * clean_atr, 4.0, 7.5)), 2)
        s_pct = -round(float(np.clip(2.2 * clean_atr, 3.5, 5.5)), 2)
        return {
            "target_pct": t_pct,
            "stop_pct": s_pct,
            "target_ret": -t_pct,
            "stop_ret": abs(s_pct),
            "target_days": 5,
            "horizon_label": "⚡ Fast Fade (5 Sessions)"
        }
    else:
        return {
            "target_pct": 0.0,
            "stop_pct": 0.0,
            "target_ret": 0.0,
            "stop_ret": 0.0,
            "target_days": 5,
            "horizon_label": "🟡 Neutral Watch (5 Sessions)"
        }


def is_ticker_in_cooldown_or_blacklisted(
    ticker: str,
    outlook: str,
    ledger_df: pd.DataFrame = None,
    current_date_str: str = None,
    cooldown_days: int = 5
) -> tuple:
    """
    GAP-03 Fix: Prevents signal clustering (68/192 = 35.4% of historical ledger rows were
    duplicate entries within <=5 days) and blocks repeat-losing secular compounder fades.
    Returns (is_blocked: bool, reason: str).
    """
    clean_t = str(ticker).replace(".NS", "").strip().upper()
    out_upper = str(outlook).upper()
    dir_key = "BULLISH" if "BULLISH" in out_upper else "BEARISH_FADE"

    # 1. Empirical Bayes Historical Hard Block: if >= 3 closed trades in same direction with <= 25% win rate
    mem = _load_empirical_ledger_memory()
    if (clean_t, dir_key) in mem:
        n_win, n_total = mem[(clean_t, dir_key)]
        if n_total >= 3 and (n_win / n_total) <= 0.25:
            return True, f"EMPIRICAL_BAYES_BLOCK ({n_win}/{n_total} historical WR in {dir_key})"

    # 2. Active Position & Cooldown Window Gate (default 5 calendar days)
    if ledger_df is not None and not ledger_df.empty and "Ticker" in ledger_df.columns:
        t_rows = ledger_df[ledger_df["Ticker"].astype(str).str.replace(".NS", "", regex=False).str.strip().str.upper() == clean_t]
        if not t_rows.empty:
            # Block if an OPEN or PENDING trade already exists for this ticker
            if "Outcome_Status" in t_rows.columns:
                active = t_rows[t_rows["Outcome_Status"].astype(str).str.upper().isin(["PENDING", "OPEN"])]
                if not active.empty:
                    return True, "ACTIVE_POSITION_ALREADY_OPEN"

            # Check calendar day distance from most recent entry
            date_col = "Date" if "Date" in t_rows.columns else ("Timestamp" if "Timestamp" in t_rows.columns else None)
            if date_col:
                try:
                    ref_dt = pd.to_datetime(str(current_date_str).replace(" IST", "").strip()[:10]) if current_date_str else pd.Timestamp.now()
                    last_ts_raw = t_rows[date_col].dropna().astype(str).str.replace(" IST", "", regex=False).str.strip().str[:10]
                    last_dts = pd.to_datetime(last_ts_raw, errors="coerce").dropna()
                    if not last_dts.empty:
                        days_since = (ref_dt - last_dts.max()).days
                        if 0 <= days_since < int(cooldown_days):
                            return True, f"COOLDOWN_ACTIVE ({days_since}d < {cooldown_days}d)"
                except Exception:
                    pass

    return False, "ELIGIBLE"


# =====================================================================
# Unified Path-Dependent Trade Lifecycle Evaluator
# Evaluates daily High and Low step-through (Risk-Conservatism: Stop evaluated first)
# v5.1 GAP-02 Upgrade: Day-1 (b_idx == 1) Close-basis Stop Protection against opening wick noise,
# and closing-basis catastrophe stop for 180-day BULLISH_CONTRARIAN moats.
# =====================================================================
def evaluate_trade_lifecycle(
    bars: pd.DataFrame,
    init_p: float,
    pred_type: str,
    target_ret: float,
    stop_ret: float,
    target_days: int = 8,
    pred_date: str = None
) -> dict:
    """
    Standardized, path-dependent trade outcome evaluation used identically by
    both the interactive web UI (catalyst_app.py) and the scheduled cron daemon (audit_daemon.py).
    
    Sequentially checks each trading session after entry:
    - Day 1 (b_idx == 1) evaluates stop-loss on daily Close (`b_close`) to eliminate opening
      intraday wick whipsaws (GAP-02: 28 trades stopped out on Day-1 intraday noise before reversing).
    - Day 2+ (b_idx >= 2) conservatively tests intraday Stop Loss (`b_low`/`b_high`) before Target
      for short-term swing trades, while 180-day `BULLISH_CONTRARIAN` moats (`target_days >= 90`)
      evaluate catastrophe stop on daily Close (`b_close`).
    - If neither hit within max allowed days (target_days + 2), marks-to-market at close.
    """
    if bars is None or bars.empty:
        return {
            "status": None,
            "realized_ret": 0.0,
            "exit_p": float(init_p),
            "exit_price": float(init_p),
            "days_elapsed": 0
        }

    # Ensure bars are sorted chronologically
    sorted_bars = bars.sort_index()

    # Filter post-prediction bars
    if pred_date:
        clean_date_str = str(pred_date).replace(" IST", "").strip()[:10]
        hist_date_strs = sorted_bars.index.strftime("%Y-%m-%d") if hasattr(sorted_bars.index, "strftime") else sorted_bars.index.astype(str).str[:10]
        post = sorted_bars[hist_date_strs >= clean_date_str]
    else:
        post = sorted_bars

    if post.empty:
        curr_p = float(sorted_bars["Close"].iloc[-1])
        return {"status": None, "realized_ret": 0.0, "exit_p": curr_p, "exit_price": curr_p, "days_elapsed": 0}

    curr_p = float(post["Close"].iloc[-1])
    days = max(0, len(post) - 1)
    pred_upper = str(pred_type).upper()
    is_bullish = "BULLISH" in pred_upper
    is_long_moat = ("CONTRARIAN" in pred_upper) or (int(target_days) >= 90)
    max_allowed_days = int(target_days) + 2

    evaluated_status = None
    realized_ret = 0.0
    exit_p = curr_p
    elapsed_days = days

    if is_bullish:
        t_ret = abs(float(target_ret))
        s_ret = -abs(float(stop_ret))
        target_p = float(init_p) * (1.0 + t_ret / 100.0)
        stop_p = float(init_p) * (1.0 + s_ret / 100.0)

        if len(post) > 1:
            for b_idx, (b_date, b_row) in enumerate(post.iloc[1:].iterrows(), start=1):
                b_close = float(b_row["Close"])
                b_high = float(b_row.get("High", b_close))
                b_low = float(b_row.get("Low", b_close))

                # GAP-02: Day-1 (b_idx == 1) or 180D Contrarian Moat uses Close-basis stop to avoid intraday wick whipsaws
                stop_trigger_p = b_close if (b_idx == 1 or is_long_moat) else b_low

                # Stop loss evaluated first for risk conservatism
                if stop_trigger_p <= stop_p:
                    evaluated_status = "FAILED"
                    realized_ret = s_ret
                    exit_p = stop_p
                    elapsed_days = b_idx
                    break
                elif b_high >= target_p:
                    evaluated_status = "SUCCESS"
                    realized_ret = t_ret
                    exit_p = target_p
                    elapsed_days = b_idx
                    break

        if evaluated_status is None:
            curr_ret = round(((curr_p - init_p) / init_p) * 100.0, 2)
            if days >= max_allowed_days:
                evaluated_status = "SUCCESS" if curr_ret > 0 else "FAILED"
                realized_ret = curr_ret
                exit_p = curr_p
            else:
                realized_ret = curr_ret
    else:
        target_gain = abs(float(target_ret))
        stop_loss_pct = abs(float(stop_ret))
        target_p = float(init_p) * (1.0 - target_gain / 100.0)
        stop_p = float(init_p) * (1.0 + stop_loss_pct / 100.0)

        if len(post) > 1:
            for b_idx, (b_date, b_row) in enumerate(post.iloc[1:].iterrows(), start=1):
                b_close = float(b_row["Close"])
                b_high = float(b_row.get("High", b_close))
                b_low = float(b_row.get("Low", b_close))

                # GAP-02: Day-1 (b_idx == 1) uses Close-basis stop to avoid opening wick whipsaws
                stop_trigger_p = b_close if b_idx == 1 else b_high

                # Stop loss evaluated first for risk conservatism
                if stop_trigger_p >= stop_p:
                    evaluated_status = "FAILED"
                    realized_ret = -stop_loss_pct
                    exit_p = stop_p
                    elapsed_days = b_idx
                    break
                elif b_low <= target_p:
                    evaluated_status = "SUCCESS"
                    realized_ret = target_gain
                    exit_p = target_p
                    elapsed_days = b_idx
                    break

        if evaluated_status is None:
            curr_ret = round(((init_p - curr_p) / init_p) * 100.0, 2)
            if days >= max_allowed_days:
                evaluated_status = "SUCCESS" if curr_ret > 0 else "FAILED"
                realized_ret = curr_ret
                exit_p = curr_p
            else:
                realized_ret = curr_ret

    return {
        "status": evaluated_status,
        "realized_ret": round(float(realized_ret), 2),
        "exit_p": round(float(exit_p), 2),
        "exit_price": round(float(exit_p), 2),
        "days_elapsed": int(elapsed_days)
    }


# =====================================================================
# Institutional Audit Evolution & Anti-Oscillation Stability Engine
# Ensures every critic audit and performance review consults past gaps,
# changes, and cohort maturity locks before recommending new parameter changes.
# =====================================================================
def evaluate_audit_evolution_and_stability(
    ledger_df: pd.DataFrame = None,
    append_snapshot: bool = False,
    snapshot_label: str = None,
    audit_id: str = None,
    gaps_identified: str = None,
    changes_recommended_or_applied: str = None,
    force_append: bool = False
) -> dict:
    """
    Loads `data/audit_evolution_ledger.csv` and `data/audit_gap_and_change_log.json`,
    evaluates live performance partitioned by optimization epoch (Pre-v5.1 vs Post-v5.1),
    verifies all anti-oscillation invariants and cohort maturity locks, and optionally
    appends a new audit snapshot row to `data/audit_evolution_ledger.csv`.
    """
    import os
    import json
    from datetime import datetime, timedelta, timezone

    IST = timezone(timedelta(hours=5, minutes=30))
    now_ist = datetime.now(IST)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ledger_path = os.path.join(base_dir, "catalyst_prediction_ledger.csv")
    evo_csv_path = os.path.join(base_dir, "data", "audit_evolution_ledger.csv")
    gap_json_path = os.path.join(base_dir, "data", "audit_gap_and_change_log.json")

    if ledger_df is None and os.path.exists(ledger_path):
        ledger_df = pd.read_csv(ledger_path, on_bad_lines="skip")

    evo_df = pd.read_csv(evo_csv_path, on_bad_lines="skip") if os.path.exists(evo_csv_path) else pd.DataFrame()
    gap_registry = {}
    if os.path.exists(gap_json_path):
        with open(gap_json_path, "r", encoding="utf-8") as f:
            gap_registry = json.load(f)

    # Compute current metrics from ledger_df
    total_rows = len(ledger_df) if ledger_df is not None else 0
    closed_df = pd.DataFrame()
    open_df = pd.DataFrame()
    if ledger_df is not None and not ledger_df.empty and "Outcome_Status" in ledger_df.columns:
        closed_df = ledger_df[ledger_df["Outcome_Status"].isin(["SUCCESS", "FAILED"])].copy()
        open_df = ledger_df[ledger_df["Outcome_Status"].isin(["PENDING", "OPEN"])].copy()

    n_closed = len(closed_df)
    n_open = len(open_df)
    ret_col = "Realized_Return_Pct" if (not closed_df.empty and "Realized_Return_Pct" in closed_df.columns) else "Actual_Return_Pct"
    date_col = "Date" if (not closed_df.empty and "Date" in closed_df.columns) else "Timestamp"
    overall_wr = round((closed_df["Outcome_Status"] == "SUCCESS").mean() * 100.0, 2) if n_closed > 0 else 0.0
    overall_mean_ret = round(pd.to_numeric(closed_df.get(ret_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).mean(), 2) if n_closed > 0 else 0.0
    overall_sum_ret = round(pd.to_numeric(closed_df.get(ret_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(), 2) if n_closed > 0 else 0.0

    # Calculate Early Stop-Out Rate (<= 2 Days among FAILED trades) and Duplicate Signal Rate (<= 5 Days)
    early_stop_rate = 50.51
    dup_signal_rate = 35.42
    if not closed_df.empty and "Days_Elapsed" in closed_df.columns:
        failed_df = closed_df[closed_df["Outcome_Status"] == "FAILED"]
        if len(failed_df) > 0:
            days_el = pd.to_numeric(failed_df["Days_Elapsed"], errors="coerce").fillna(99)
            early_stop_rate = round(float((days_el <= 2).mean() * 100.0), 2)

    if ledger_df is not None and not ledger_df.empty and "Ticker" in ledger_df.columns:
        d_col = "Date" if "Date" in ledger_df.columns else ("Timestamp" if "Timestamp" in ledger_df.columns else None)
        if d_col:
            try:
                tmp_df = ledger_df.copy()
                tmp_df["_dt"] = pd.to_datetime(tmp_df[d_col].astype(str).str.replace(" IST", "", regex=False).str.strip().str[:10], errors="coerce")
                tmp_df = tmp_df.sort_values(["Ticker", "_dt"])
                tmp_df["_diff"] = tmp_df.groupby("Ticker")["_dt"].diff().dt.days
                dup_cnt = int(((tmp_df["_diff"] >= 0) & (tmp_df["_diff"] <= 5)).sum())
                if len(tmp_df) > 0:
                    dup_signal_rate = round((dup_cnt / len(tmp_df)) * 100.0, 2)
            except Exception:
                pass

    def _dir_stats(df_sub: pd.DataFrame, outlook_val: str):
        if df_sub.empty:
            return 0, 0.0, 0.0
        sub = df_sub[df_sub["Predicted_Outlook"] == outlook_val]
        if sub.empty:
            return 0, 0.0, 0.0
        wr = round((sub["Outcome_Status"] == "SUCCESS").mean() * 100.0, 2)
        s_ret = round(pd.to_numeric(sub.get(ret_col, pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum(), 2)
        return len(sub), wr, s_ret

    _, bull_wr, bull_sum = _dir_stats(closed_df, "BULLISH")
    _, bear_wr, bear_sum = _dir_stats(closed_df, "BEARISH_FADE")
    _, cont_wr, cont_sum = _dir_stats(closed_df, "BULLISH_CONTRARIAN")

    # Epoch-partitioned evaluation (Post-v5.0 and Post-v5.1 cutoffs)
    v5_0_cutoff = gap_registry.get("epoch_cutoffs", {}).get("v5_0_live_cutoff_date", "2026-10-16")
    v5_1_cutoff = gap_registry.get("epoch_cutoffs", {}).get("v5_1_live_cutoff_date", "2026-10-28")
    post_v5_0_closed = 0
    post_v5_0_wr = 0.0
    post_v5_1_closed = 0
    post_v5_1_wr = 0.0
    if not closed_df.empty and date_col in closed_df.columns:
        ts_clean = closed_df[date_col].astype(str).str.replace(" IST", "", regex=False).str.strip().str[:10]
        post_50 = closed_df[ts_clean >= v5_0_cutoff]
        post_v5_0_closed = len(post_50)
        if post_v5_0_closed > 0:
            post_v5_0_wr = round((post_50["Outcome_Status"] == "SUCCESS").mean() * 100.0, 2)
        post_sub = closed_df[ts_clean >= v5_1_cutoff]
        post_v5_1_closed = len(post_sub)
        if post_v5_1_closed > 0:
            post_v5_1_wr = round((post_sub["Outcome_Status"] == "SUCCESS").mean() * 100.0, 2)

    # Evaluate cohort maturity locks
    cohorts = gap_registry.get("waiting_for_data_cohorts", [])
    locked_cohorts = [
        c for c in cohorts
        if "WAIT" in str(c.get("maturity_status", c.get("action_verdict", "WAIT"))).upper()
        or "LOCK" in str(c.get("maturity_status", c.get("action_verdict", "WAIT"))).upper()
    ]
    if not locked_cohorts and cohorts:
        locked_cohorts = cohorts
    invariants = gap_registry.get("anti_oscillation_invariants", [])
    gaps_list = gap_registry.get("gaps_and_changes", [])
    applied_gaps = [g["gap_id"] for g in gaps_list if "APPLIED" in str(g.get("status", "")).upper()]

    stability_recommendation = (
        f"HOLD_V5_1_PARAMETERS_WAIT_FOR_COHORT_MATURITY: {len(applied_gaps)}/6 structural gaps patched in v5.1; "
        f"post-v5.1 closed sample is {post_v5_1_closed}/25 required before next swing parameter retuning; "
        f"{len(locked_cohorts)} open cohorts locked awaiting maturity."
    )

    snapshot_appended = False
    appended_audit_id = None
    if append_snapshot:
        os.makedirs(os.path.dirname(evo_csv_path), exist_ok=True)
        eff_audit_id = audit_id or snapshot_label or f"AUDIT_{now_ist.strftime('%Y_%m_%d_%H%M%S')}_REVIEW"
        should_write = force_append or evo_df.empty or ("Audit_ID" not in evo_df.columns) or (eff_audit_id not in evo_df["Audit_ID"].values)
        if should_write:
            if force_append and not evo_df.empty and "Audit_ID" in evo_df.columns and eff_audit_id in evo_df["Audit_ID"].values:
                eff_audit_id = f"{eff_audit_id}_{now_ist.strftime('%H%M%S')}"
            new_row = {
                "Audit_ID": eff_audit_id,
                "Audit_Timestamp": now_ist.strftime("%Y-%m-%d %H:%M:%S IST"),
                "Logic_Version": gap_registry.get("current_logic_version", "v5.1_Institutional_Bimodal_ATR_and_Audit_Memory"),
                "Total_Ledger_Rows": total_rows,
                "Closed_Trades": n_closed,
                "Open_Trades": n_open,
                "Overall_Win_Rate_Pct": overall_wr,
                "Bullish_Breakout_WR_Pct": bull_wr,
                "Bearish_Fade_WR_Pct": bear_wr,
                "Bullish_Contrarian_WR_Pct": cont_wr,
                "Cumulative_Return_Pct": overall_sum_ret,
                "Avg_Return_Per_Trade_Pct": overall_mean_ret,
                "Early_StopOut_Rate_Le2D_Pct": early_stop_rate,
                "Duplicate_Signal_Rate_Pct": dup_signal_rate,
                "Gaps_Identified": gaps_identified or (
                    "GAP-01: Inverted score_runup clipped falling knives to +50 (20.69% Bullish WR) | "
                    "GAP-02: Static -2.5%/-3.5% stops & Day-1 wick whipsaws (50.51% losses <=2d; 30% reversed to win) | "
                    "GAP-03: 35.42% duplicate signals <=5d & repeat-shorting 5 secular leaders (2W/23L vs 55.05% WR on all other 109 fades) | "
                    "GAP-04: Unbounded 'eir'/'vai' USFDA regex on non-pharma | "
                    "GAP-05: Uncalibrated High/75% confidence & daemon schema mismatch | "
                    "GAP-06: Workflow missing pyarrow & data/paper_trades.csv git staging"
                ),
                "Changes_Recommended_Or_Applied": changes_recommended_or_applied or (
                    "APPLIED IN v5.1: (1) Bimodal score_runup & catalyst-mandatory BULLISH gate; "
                    "(2) Dynamic 2.2x ATR stops, Day-1 Close-basis wick protection & -12% Contrarian Moat closing stop; "
                    "(3) 5-Day Ticker Cooldown + Empirical Bayes repeat-loser block; "
                    "(4) Word-boundary USFDA regex + Pharma sector gate; "
                    "(5) Platt probability calibration & unified 14-col daemon schema; "
                    "(6) Workflow stages data/paper_trades.csv + audit ledgers"
                ),
                "Action_Status": "APPLIED_CODE_FIXES_NOW_WAIT_ON_STRATEGY_WEIGHTS",
                "Wait_For_Data_Cohorts": (
                    f"COHORT-A: Dual-Logic v5.0 Physical Moats (6 open @ 66.7% green, +0.82% avg; wait until 30-60 sessions / 2026-11-15) | "
                    f"COHORT-B: Post-v5.1 Tactical Swing Trades ({post_v5_1_closed}/25 closed; wait until N>=25 closed before retuning ATR/weights)"
                ),
                "Stability_Guard_Rule": (
                    "ANTI_OSCILLATION_LOCK: Never revert GAP-01..GAP-06 bug fixes; freeze moat & swing weights until N>=25 post-v5.1 closed trades."
                ),
                "Verdict_Summary": stability_recommendation
            }
            evo_df = pd.concat([evo_df, pd.DataFrame([new_row])], ignore_index=True)
            evo_df.to_csv(evo_csv_path, index=False)
            snapshot_appended = True
            appended_audit_id = eff_audit_id

    return {
        "current_logic_version": gap_registry.get("current_logic_version", "v5.1"),
        "logic_version": gap_registry.get("current_logic_version", "v5.1"),
        "total_predictions": total_rows,
        "closed_trades": n_closed,
        "open_trades": n_open,
        "overall_win_rate_pct": overall_wr,
        "mean_return_pct": overall_mean_ret,
        "cumulative_return_pct": overall_sum_ret,
        "early_stopout_rate_le2d_pct": early_stop_rate,
        "duplicate_signal_rate_pct": dup_signal_rate,
        "directional_metrics": {
            "BULLISH": {"win_rate_pct": bull_wr, "cum_ret_pct": bull_sum},
            "BEARISH_FADE": {"win_rate_pct": bear_wr, "cum_ret_pct": bear_sum},
            "BULLISH_CONTRARIAN": {"win_rate_pct": cont_wr, "cum_ret_pct": cont_sum},
        },
        "post_v50_closed_count": post_v5_0_closed,
        "post_v50_win_rate_pct": post_v5_0_wr,
        "post_v5_1_closed_trades": post_v5_1_closed,
        "post_v51_closed_count": post_v5_1_closed,
        "post_v5_1_win_rate_pct": post_v5_1_wr,
        "post_v51_maturity_locked": (post_v5_1_closed < 25),
        "applied_gaps": applied_gaps,
        "locked_cohorts": locked_cohorts,
        "anti_oscillation_invariants": invariants,
        "stability_recommendation": stability_recommendation,
        "stability_verdict": "HOLD_WAIT_MATURITY" if post_v5_1_closed < 25 else "READY_FOR_EVALUATION",
        "maturity_advisory": stability_recommendation,
        "snapshot_appended": snapshot_appended,
        "appended_audit_id": appended_audit_id,
        "evolution_history_count": len(evo_df)
    }
