"""
ticker_utils.py - High-precision Institutional Ticker & Entity Matching for Indian Equities
Prevents false positives on short ticker abbreviations (LT, BEL, HAL, REC, etc.)
and resolves full corporate entity names to canonical NSE tickers.
"""

import re
from typing import List

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
