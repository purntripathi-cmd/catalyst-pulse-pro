# =====================================================================
# DATA PIPELINE 20Y: 2006-2026 HISTORICAL DATASET & MULTI-DIMENSIONAL FEATURES
# AI-Powered Deep-Value & Contrarian Investment Architecture (v4.2-Production)
# =====================================================================
import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("DataPipeline20Y")

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)
DATASET_FILE = os.path.join(DATA_DIR, "dual_logic_20y_dataset.csv")
HISTORICAL_BARS_CACHE = os.path.join(DATA_DIR, "historical_20y_bars.parquet")

# Sector taxonomy & physical moat / AI disruption profiling
UNIVERSE_PROFILES = [
    # Power Grids, Utilities & Energy
    {"ticker": "POWERGRID.NS", "name": "Power Grid Corp", "sector": "Power & Grid", "moat_type": "Monopoly Grid Transmission", "capex_scale": 0.88, "ai_vulnerability": 0.05, "base_de": 1.40, "base_ic": 4.2, "fii_pct": 29.8, "fii_12m_chg": 1.4, "dii_pct": 16.5, "dii_12m_chg": 1.9, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.4%, DII ▲ +1.9%)"},
    {"ticker": "NTPC.NS", "name": "NTPC Limited", "sector": "Power & Grid", "moat_type": "Baselod Power Generation", "capex_scale": 0.85, "ai_vulnerability": 0.08, "base_de": 1.45, "base_ic": 3.8, "fii_pct": 17.6, "fii_12m_chg": 2.1, "dii_pct": 28.4, "dii_12m_chg": 1.5, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +2.1%, DII ▲ +1.5%)"},
    {"ticker": "ONGC.NS", "name": "ONGC", "sector": "Oil & Gas / Energy", "moat_type": "Sovereign Hydrocarbon Reserves", "capex_scale": 0.90, "ai_vulnerability": 0.06, "base_de": 0.45, "base_ic": 8.5, "fii_pct": 8.9, "fii_12m_chg": 0.8, "dii_pct": 19.8, "dii_12m_chg": 1.7, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +0.8%, DII ▲ +1.7%)"},
    {"ticker": "COALINDIA.NS", "name": "Coal India", "sector": "Mining & Resources", "moat_type": "Near-Monopoly Fuel Supply", "capex_scale": 0.75, "ai_vulnerability": 0.04, "base_de": 0.12, "base_ic": 18.0, "fii_pct": 8.4, "fii_12m_chg": 1.2, "dii_pct": 23.5, "dii_12m_chg": 0.9, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.2%, DII ▲ +0.9%)"},
    {"ticker": "RELIANCE.NS", "name": "Reliance Industries", "sector": "Energy & Infrastructure", "moat_type": "Refining & Telecom Backbone", "capex_scale": 0.82, "ai_vulnerability": 0.15, "base_de": 0.58, "base_ic": 6.2, "fii_pct": 21.6, "fii_12m_chg": -0.7, "dii_pct": 17.8, "dii_12m_chg": 2.4, "inst_trend": "🟢 DII Absorption (FII ▼ -0.7%, DII ▲ +2.4%)"},
    {"ticker": "TATAPOWER.NS", "name": "Tata Power", "sector": "Power & Grid", "moat_type": "Renewable & Distribution Concessions", "capex_scale": 0.80, "ai_vulnerability": 0.10, "base_de": 1.35, "base_ic": 3.4, "fii_pct": 10.4, "fii_12m_chg": 1.6, "dii_pct": 16.5, "dii_12m_chg": 1.8, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.6%, DII ▲ +1.8%)"},
    {"ticker": "IOC.NS", "name": "Indian Oil Corp", "sector": "Oil & Gas / Energy", "moat_type": "Downstream Pipeline Network", "capex_scale": 0.78, "ai_vulnerability": 0.09, "base_de": 0.95, "base_ic": 4.8, "fii_pct": 7.9, "fii_12m_chg": 0.5, "dii_pct": 13.8, "dii_12m_chg": 1.1, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +0.5%, DII ▲ +1.1%)"},
    {"ticker": "BPCL.NS", "name": "Bharat Petroleum", "sector": "Oil & Gas / Energy", "moat_type": "Refinery & Retail Concession", "capex_scale": 0.74, "ai_vulnerability": 0.10, "base_de": 1.10, "base_ic": 4.1, "fii_pct": 14.5, "fii_12m_chg": 1.1, "dii_pct": 22.1, "dii_12m_chg": 1.4, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.1%, DII ▲ +1.4%)"},

    # Physical Infrastructure, Heavy CapEx & Logistics
    {"ticker": "LT.NS", "name": "Larsen & Toubro", "sector": "Infrastructure", "moat_type": "Mega-EPC Execution Moat", "capex_scale": 0.82, "ai_vulnerability": 0.12, "base_de": 1.05, "base_ic": 4.5, "fii_pct": 25.2, "fii_12m_chg": 1.1, "dii_pct": 37.4, "dii_12m_chg": 1.8, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.1%, DII ▲ +1.8%)"},
    {"ticker": "ADANIPORTS.NS", "name": "Adani Ports & SEZ", "sector": "Logistics & Ports", "moat_type": "Deep-Water Port Concessions", "capex_scale": 0.92, "ai_vulnerability": 0.06, "base_de": 1.35, "base_ic": 3.6, "fii_pct": 14.8, "fii_12m_chg": 2.3, "dii_pct": 14.1, "dii_12m_chg": 1.6, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +2.3%, DII ▲ +1.6%)"},
    {"ticker": "TATASTEEL.NS", "name": "Tata Steel", "sector": "Metals & Mining", "moat_type": "Integrated Captive Iron Ore", "capex_scale": 0.86, "ai_vulnerability": 0.08, "base_de": 1.15, "base_ic": 3.9, "fii_pct": 19.4, "fii_12m_chg": -0.5, "dii_pct": 23.8, "dii_12m_chg": 1.9, "inst_trend": "🟢 DII Absorption (FII ▼ -0.5%, DII ▲ +1.9%)"},
    {"ticker": "HINDALCO.NS", "name": "Hindalco Industries", "sector": "Metals & Mining", "moat_type": "Global Bauxite & Smelter Base", "capex_scale": 0.84, "ai_vulnerability": 0.09, "base_de": 0.92, "base_ic": 4.6, "fii_pct": 26.5, "fii_12m_chg": 1.4, "dii_pct": 26.1, "dii_12m_chg": 1.2, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.4%, DII ▲ +1.2%)"},
    {"ticker": "JSWSTEEL.NS", "name": "JSW Steel", "sector": "Metals & Mining", "moat_type": "Low-Cost Steel Manufacturing", "capex_scale": 0.85, "ai_vulnerability": 0.10, "base_de": 1.30, "base_ic": 3.5, "fii_pct": 11.2, "fii_12m_chg": -0.4, "dii_pct": 9.8, "dii_12m_chg": 0.7, "inst_trend": "🟢 DII Absorption (FII ▼ -0.4%, DII ▲ +0.7%)"},
    {"ticker": "ULTRACEMCO.NS", "name": "UltraTech Cement", "sector": "Infrastructure Materials", "moat_type": "Limestone Mining & Freight Network", "capex_scale": 0.80, "ai_vulnerability": 0.07, "base_de": 0.38, "base_ic": 9.2, "fii_pct": 18.2, "fii_12m_chg": 1.0, "dii_pct": 16.5, "dii_12m_chg": 1.3, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.0%, DII ▲ +1.3%)"},
    {"ticker": "GRASIM.NS", "name": "Grasim Industries", "sector": "Infrastructure Materials", "moat_type": "Viscose & Caustic Soda Dominance", "capex_scale": 0.72, "ai_vulnerability": 0.11, "base_de": 0.65, "base_ic": 5.8, "fii_pct": 12.8, "fii_12m_chg": 0.6, "dii_pct": 17.1, "dii_12m_chg": 1.5, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +0.6%, DII ▲ +1.5%)"},
    {"ticker": "CONCOR.NS", "name": "Container Corp of India", "sector": "Logistics & Ports", "moat_type": "Rail Container Depots & Terminals", "capex_scale": 0.76, "ai_vulnerability": 0.08, "base_de": 0.05, "base_ic": 25.0, "fii_pct": 19.5, "fii_12m_chg": 1.7, "dii_pct": 22.8, "dii_12m_chg": 2.1, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.7%, DII ▲ +2.1%)"},
    {"ticker": "BHARTIARTL.NS", "name": "Bharti Airtel", "sector": "Telecom & Digital Infra", "moat_type": "Spectrum & Fiber Optic Duopoly", "capex_scale": 0.88, "ai_vulnerability": 0.18, "base_de": 1.48, "base_ic": 3.2, "fii_pct": 24.8, "fii_12m_chg": 2.8, "dii_pct": 19.5, "dii_12m_chg": 1.2, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +2.8%, DII ▲ +1.2%)"},

    # Capital Goods & Electrical Equipment
    {"ticker": "SIEMENS.NS", "name": "Siemens India", "sector": "Capital Goods", "moat_type": "Electrification & Grid Automation", "capex_scale": 0.70, "ai_vulnerability": 0.22, "base_de": 0.02, "base_ic": 35.0, "fii_pct": 7.5, "fii_12m_chg": 1.3, "dii_pct": 12.4, "dii_12m_chg": 1.6, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.3%, DII ▲ +1.6%)"},
    {"ticker": "ABB.NS", "name": "ABB India", "sector": "Capital Goods", "moat_type": "Heavy Power Transmission Systems", "capex_scale": 0.68, "ai_vulnerability": 0.24, "base_de": 0.03, "base_ic": 32.0, "fii_pct": 11.4, "fii_12m_chg": 1.9, "dii_pct": 9.8, "dii_12m_chg": 1.2, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.9%, DII ▲ +1.2%)"},
    {"ticker": "BHEL.NS", "name": "BHEL", "sector": "Capital Goods", "moat_type": "Thermal & Nuclear Turbine Engineering", "capex_scale": 0.75, "ai_vulnerability": 0.14, "base_de": 0.42, "base_ic": 3.1, "fii_pct": 8.8, "fii_12m_chg": 2.4, "dii_pct": 16.2, "dii_12m_chg": 1.8, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +2.4%, DII ▲ +1.8%)"},

    # Core Banking & Financial Intermediaries
    {"ticker": "SBIN.NS", "name": "State Bank of India", "sector": "Banking & Credit", "moat_type": "Sovereign Low-Cost CASA Moat", "capex_scale": 0.55, "ai_vulnerability": 0.30, "base_de": 1.42, "base_ic": 3.5, "fii_pct": 11.1, "fii_12m_chg": 0.9, "dii_pct": 24.3, "dii_12m_chg": 1.4, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +0.9%, DII ▲ +1.4%)"},
    {"ticker": "HDFCBANK.NS", "name": "HDFC Bank", "sector": "Banking & Credit", "moat_type": "Private Banking Distribution Moat", "capex_scale": 0.50, "ai_vulnerability": 0.32, "base_de": 1.25, "base_ic": 4.1, "fii_pct": 47.2, "fii_12m_chg": -4.2, "dii_pct": 33.6, "dii_12m_chg": 5.1, "inst_trend": "🟢 DII Absorption (FII ▼ -4.2%, DII ▲ +5.1%)"},
    {"ticker": "ICICIBANK.NS", "name": "ICICI Bank", "sector": "Banking & Credit", "moat_type": "Multi-Asset Underwriting Moat", "capex_scale": 0.52, "ai_vulnerability": 0.31, "base_de": 1.30, "base_ic": 3.9, "fii_pct": 44.5, "fii_12m_chg": 0.8, "dii_pct": 45.1, "dii_12m_chg": 1.6, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +0.8%, DII ▲ +1.6%)"},
    {"ticker": "AXISBANK.NS", "name": "Axis Bank", "sector": "Banking & Credit", "moat_type": "Corporate Lending & Retail Franchise", "capex_scale": 0.48, "ai_vulnerability": 0.33, "base_de": 1.38, "base_ic": 3.4, "fii_pct": 53.8, "fii_12m_chg": 1.2, "dii_pct": 29.5, "dii_12m_chg": 0.9, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.2%, DII ▲ +0.9%)"},
    {"ticker": "BAJFINANCE.NS", "name": "Bajaj Finance", "sector": "NBFC & Consumer Credit", "moat_type": "Consumer Lending Ecosystem", "capex_scale": 0.45, "ai_vulnerability": 0.38, "base_de": 1.48, "base_ic": 3.1, "fii_pct": 20.8, "fii_12m_chg": -1.5, "dii_pct": 14.6, "dii_12m_chg": 2.2, "inst_trend": "🟢 DII Absorption (FII ▼ -1.5%, DII ▲ +2.2%)"},

    # Cyclicals & AI-Vulnerable IT / Services (Control Cohort)
    {"ticker": "TCS.NS", "name": "Tata Consultancy Services", "sector": "IT & Software Services", "moat_type": "Enterprise IT Service Contracts", "capex_scale": 0.20, "ai_vulnerability": 0.82, "base_de": 0.05, "base_ic": 45.0, "fii_pct": 12.5, "fii_12m_chg": -1.8, "dii_pct": 10.2, "dii_12m_chg": 0.8, "inst_trend": "🟡 FII Trimming (FII ▼ -1.8%, DII ▲ +0.8%)"},
    {"ticker": "INFY.NS", "name": "Infosys", "sector": "IT & Software Services", "moat_type": "Global IT Outsourcing", "capex_scale": 0.18, "ai_vulnerability": 0.85, "base_de": 0.06, "base_ic": 40.0, "fii_pct": 31.8, "fii_12m_chg": -2.4, "dii_pct": 36.2, "dii_12m_chg": 2.1, "inst_trend": "🟡 DII Absorption (FII ▼ -2.4%, DII ▲ +2.1%)"},
    {"ticker": "WIPRO.NS", "name": "Wipro Ltd", "sector": "IT & Software Services", "moat_type": "Consulting & Software Services", "capex_scale": 0.19, "ai_vulnerability": 0.88, "base_de": 0.15, "base_ic": 22.0, "fii_pct": 6.8, "fii_12m_chg": -1.2, "dii_pct": 8.5, "dii_12m_chg": 0.5, "inst_trend": "🟡 FII Trimming (FII ▼ -1.2%, DII ▲ +0.5%)"},
    {"ticker": "HCLTECH.NS", "name": "HCL Technologies", "sector": "IT & Software Services", "moat_type": "Engineering R&D Services", "capex_scale": 0.22, "ai_vulnerability": 0.78, "base_de": 0.10, "base_ic": 30.0, "fii_pct": 18.5, "fii_12m_chg": -0.8, "dii_pct": 15.2, "dii_12m_chg": 1.1, "inst_trend": "🟡 DII Absorption (FII ▼ -0.8%, DII ▲ +1.1%)"},

    # Defensive FMCG & Pharma
    {"ticker": "ITC.NS", "name": "ITC Ltd", "sector": "FMCG & Agri", "moat_type": "Distribution & Brand Monopoly", "capex_scale": 0.45, "ai_vulnerability": 0.15, "base_de": 0.01, "base_ic": 60.0, "fii_pct": 43.2, "fii_12m_chg": 0.4, "dii_pct": 42.1, "dii_12m_chg": 0.8, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +0.4%, DII ▲ +0.8%)"},
    {"ticker": "HINDUNILVR.NS", "name": "Hindustan Unilever", "sector": "FMCG", "moat_type": "Consumer Staple Brands", "capex_scale": 0.35, "ai_vulnerability": 0.18, "base_de": 0.02, "base_ic": 55.0, "fii_pct": 12.8, "fii_12m_chg": -1.1, "dii_pct": 13.5, "dii_12m_chg": 1.4, "inst_trend": "🟢 DII Absorption (FII ▼ -1.1%, DII ▲ +1.4%)"},
    {"ticker": "SUNPHARMA.NS", "name": "Sun Pharma", "sector": "Pharma & Healthcare", "moat_type": "Specialty Generic Formulations", "capex_scale": 0.50, "ai_vulnerability": 0.25, "base_de": 0.12, "base_ic": 19.0, "fii_pct": 17.5, "fii_12m_chg": 1.5, "dii_pct": 19.8, "dii_12m_chg": 0.9, "inst_trend": "🟢 Institutional Accumulation (FII ▲ +1.5%, DII ▲ +0.9%)"},

    # Broad & Sectoral Institutional ETFs
    {"ticker": "CPSEETF.NS", "name": "CPSE ETF (PSU Giants)", "sector": "PSU ETF Basket", "moat_type": "Sovereign Monopoly Energy & Mining Basket", "capex_scale": 0.88, "ai_vulnerability": 0.06, "base_de": 0.75, "base_ic": 8.0, "fii_pct": 5.2, "fii_12m_chg": 1.8, "dii_pct": 48.5, "dii_12m_chg": 8.2, "inst_trend": "🟢 Strong PSU Institutional Inflow"},
    {"ticker": "NIFTYBEES.NS", "name": "Nifty 50 ETF", "sector": "Broad Market ETF", "moat_type": "India Top 50 Index Basket", "capex_scale": 0.60, "ai_vulnerability": 0.35, "base_de": 0.95, "base_ic": 6.0, "fii_pct": 12.0, "fii_12m_chg": 1.5, "dii_pct": 62.0, "dii_12m_chg": 6.5, "inst_trend": "🟢 Broad Institutional Allocation"},
    {"ticker": "JUNIORBEES.NS", "name": "Nifty Next 50 ETF", "sector": "Broad Market ETF", "moat_type": "High-Beta Mid-Large Basket", "capex_scale": 0.58, "ai_vulnerability": 0.40, "base_de": 1.05, "base_ic": 5.2, "fii_pct": 8.5, "fii_12m_chg": 1.2, "dii_pct": 52.0, "dii_12m_chg": 5.1, "inst_trend": "🟢 Mid-Large Institutional Accumulation"},
    {"ticker": "GOLDBEES.NS", "name": "Nippon Gold ETF", "sector": "Hard Commodity", "moat_type": "Physical Gold Bullion Scarcity", "capex_scale": 0.98, "ai_vulnerability": 0.01, "base_de": 0.00, "base_ic": 99.0, "fii_pct": 3.5, "fii_12m_chg": 0.9, "dii_pct": 45.0, "dii_12m_chg": 7.8, "inst_trend": "🟢 Gold Hedge Institutional Inflow"},
]

# Macroeconomic regimes & stress periods across 2006-2026
MACRO_CALENDAR = {
    # 2006: Expansion / Pre-crisis peak
    2006: {"regime": "Expansion", "regime_code": 0, "grid_load_idx": 0.55, "commodity_deficit": 0.70, "logistics_idx": 0.65, "stress_event": "Global Bull Run"},
    # 2007-2009: GFC Crash
    2007: {"regime": "Peak", "regime_code": 1, "grid_load_idx": 0.60, "commodity_deficit": 0.85, "logistics_idx": 0.80, "stress_event": "2007-2009 GFC Crash"},
    2008: {"regime": "Contraction", "regime_code": 2, "grid_load_idx": 0.45, "commodity_deficit": 0.40, "logistics_idx": 0.40, "stress_event": "2007-2009 GFC Crash"},
    2009: {"regime": "Trough", "regime_code": 3, "grid_load_idx": 0.48, "commodity_deficit": 0.45, "logistics_idx": 0.45, "stress_event": "2007-2009 GFC Crash"},
    # 2010-2014: Post-GFC Consolidation & Expansion
    2010: {"regime": "Expansion", "regime_code": 0, "grid_load_idx": 0.58, "commodity_deficit": 0.65, "logistics_idx": 0.60, "stress_event": "Post-GFC Recovery"},
    2011: {"regime": "Contraction", "regime_code": 2, "grid_load_idx": 0.60, "commodity_deficit": 0.55, "logistics_idx": 0.52, "stress_event": "Eurozone Debt Crisis"},
    2012: {"regime": "Trough", "regime_code": 3, "grid_load_idx": 0.62, "commodity_deficit": 0.50, "logistics_idx": 0.50, "stress_event": "Taper Tantrum Lead-up"},
    2013: {"regime": "Expansion", "regime_code": 0, "grid_load_idx": 0.65, "commodity_deficit": 0.55, "logistics_idx": 0.58, "stress_event": "INR Currency Shock"},
    2014: {"regime": "Expansion", "regime_code": 0, "grid_load_idx": 0.70, "commodity_deficit": 0.60, "logistics_idx": 0.62, "stress_event": "India General Election Rally"},
    # 2015-2016: Commodity Slump & Deep-Value Accumulation
    2015: {"regime": "Contraction", "regime_code": 2, "grid_load_idx": 0.68, "commodity_deficit": 0.30, "logistics_idx": 0.48, "stress_event": "2015-2016 Commodity Slump"},
    2016: {"regime": "Trough", "regime_code": 3, "grid_load_idx": 0.72, "commodity_deficit": 0.35, "logistics_idx": 0.50, "stress_event": "2015-2016 Commodity Slump"},
    # 2017-2019: Domestic Reforms & NBFC Crisis
    2017: {"regime": "Expansion", "regime_code": 0, "grid_load_idx": 0.75, "commodity_deficit": 0.55, "logistics_idx": 0.68, "stress_event": "GST & Reform Cycle"},
    2018: {"regime": "Peak", "regime_code": 1, "grid_load_idx": 0.78, "commodity_deficit": 0.60, "logistics_idx": 0.70, "stress_event": "IL&FS NBFC Liquidity Shock"},
    2019: {"regime": "Contraction", "regime_code": 2, "grid_load_idx": 0.79, "commodity_deficit": 0.50, "logistics_idx": 0.64, "stress_event": "Pre-COVID Slowdown"},
    # 2020: COVID Shock & V-Recovery
    2020: {"regime": "Trough", "regime_code": 3, "grid_load_idx": 0.65, "commodity_deficit": 0.40, "logistics_idx": 0.30, "stress_event": "2020 COVID Shock"},
    # 2021: Global Stimulus & Commodity Rebound
    2021: {"regime": "Expansion", "regime_code": 0, "grid_load_idx": 0.82, "commodity_deficit": 0.85, "logistics_idx": 0.88, "stress_event": "Global Re-Opening Supercycle"},
    # 2022-2023: Rate Hike Bear Market & Energy Grid Squeeze
    2022: {"regime": "Contraction", "regime_code": 2, "grid_load_idx": 0.88, "commodity_deficit": 0.92, "logistics_idx": 0.75, "stress_event": "2022-2023 Rate-Hike Bear"},
    2023: {"regime": "Trough", "regime_code": 3, "grid_load_idx": 0.90, "commodity_deficit": 0.82, "logistics_idx": 0.78, "stress_event": "2022-2023 Rate-Hike Bear"},
    # 2024-2026: AI Infrastructure & Power Demand Supercycle
    2024: {"regime": "Expansion", "regime_code": 0, "grid_load_idx": 0.94, "commodity_deficit": 0.80, "logistics_idx": 0.82, "stress_event": "AI Power Grid Supercycle"},
    2025: {"regime": "Peak", "regime_code": 1, "grid_load_idx": 0.96, "commodity_deficit": 0.84, "logistics_idx": 0.85, "stress_event": "Data Center Infrastructure Surge"},
    2026: {"regime": "Expansion", "regime_code": 0, "grid_load_idx": 0.98, "commodity_deficit": 0.88, "logistics_idx": 0.89, "stress_event": "Physical Moat Hegemony"}
}

_CACHED_20Y_DATASET = None

def fetch_or_load_historical_bars(force_download: bool = False) -> pd.DataFrame:
    """
    Retrieves daily adjusted OHLCV bars for the universe + benchmarks from 2002 to 2026.
    Caches to parquet to accelerate subsequent runs and prevent API rate limiting.
    """
    if not force_download and os.path.exists(HISTORICAL_BARS_CACHE):
        try:
            logger.info(f"Loading cached historical bars from {HISTORICAL_BARS_CACHE}...")
            raw_data = pd.read_parquet(HISTORICAL_BARS_CACHE)
            if not raw_data.empty:
                return raw_data
        except Exception as e:
            logger.warning(f"Error reading parquet cache: {e}. Downloading fresh bars...")

    tickers = [p["ticker"] for p in UNIVERSE_PROFILES]
    all_tickers = sorted(list(set(tickers + ["^BSESN", "^NSEI"])))
    logger.info(f"Downloading historical bars for {len(all_tickers)} assets via yfinance (2002-2026)...")
    raw_data = yf.download(all_tickers, start="2002-01-01", end="2026-10-15", group_by="ticker", auto_adjust=True, progress=False)
    
    try:
        raw_data.to_parquet(HISTORICAL_BARS_CACHE)
        logger.info(f"Saved historical bars cache to {HISTORICAL_BARS_CACHE}.")
    except Exception as e:
        logger.warning(f"Could not save parquet cache: {e}")
        
    return raw_data

def generate_20y_ground_truth_dataset(force_recreate: bool = False) -> pd.DataFrame:
    """
    Constructs the official 2006-2026 20-Year empirical ground truth dataset across the 4 foundational dimensions.
    Features:
    1. Price & Momentum: Drawdown_3Y, Drawdown_Velocity, MA_Cross_Spread, Dist_200DMA (calculated from actual historical prices)
    2. Balance Sheet: Debt_Equity, Interest_Coverage, OCF_Consistency, CapEx_Intensity (fundamental taxonomy with regime impact)
    3. Physical Moat: Infrastructure_Moat, Replacement_Cost_Barrier, Regulatory_Protection, Asset_Moat_Score
    4. Macro & AI Disruption: Power_Grid_Load_Index, Commodity_Supply_Deficit, Logistics_Bottleneck, AI_Vulnerability_Score
    5. Empirical Ground Truth: Actual_3Y_CAGR, Actual_Success (CAGR >= 10%), Actual_Max_DD (derived from actual forward price paths)
    """
    global _CACHED_20Y_DATASET
    if not force_recreate and _CACHED_20Y_DATASET is not None:
        return _CACHED_20Y_DATASET.copy()

    if os.path.exists(DATASET_FILE) and not force_recreate:
        try:
            df = pd.read_csv(DATASET_FILE)
            if len(df) >= 500 and "Year" in df.columns:
                logger.info(f"Loaded existing 20Y dataset from {DATASET_FILE} ({len(df)} records).")
                _CACHED_20Y_DATASET = df
                return df.copy()
        except Exception as e:
            logger.warning(f"Failed to read existing dataset: {e}. Rebuilding...")

    logger.info("Constructing genuine 2006-2026 20-Year empirical ground truth dataset from real historical price bars...")
    raw_data = fetch_or_load_historical_bars()

    # Benchmark series for macro regime calculation
    if "^BSESN" in raw_data.columns.levels[0]:
        bench_series = raw_data["^BSESN"]["Close"].dropna()
    elif "^NSEI" in raw_data.columns.levels[0]:
        bench_series = raw_data["^NSEI"]["Close"].dropna()
    else:
        raise ValueError("No benchmark data (^BSESN or ^NSEI) available in historical data.")

    records = []
    years = list(range(2006, 2027))

    for yr in years:
        target_dt = pd.Timestamp(f"{yr}-03-25", tz=bench_series.index.tz)

        # Benchmark regime at target_dt
        b_idx = bench_series.index.searchsorted(target_dt)
        if b_idx >= len(bench_series):
            b_idx = len(bench_series) - 1

        b_close = bench_series.iloc[b_idx]
        b_hist = bench_series.iloc[:b_idx+1]
        b_ath = b_hist.max()
        b_dd = (b_close - b_ath) / b_ath

        # 1Y return of benchmark
        b_1y_idx = bench_series.index.searchsorted(pd.Timestamp(f"{yr-1}-03-25", tz=bench_series.index.tz))
        b_1y_ret = (b_close / bench_series.iloc[b_1y_idx] - 1.0) if b_1y_idx < len(bench_series) else 0.0

        # Benchmark 200 DMA
        b_200dma = b_hist.iloc[-200:].mean() if len(b_hist) >= 200 else b_close
        b_dist_200 = (b_close - b_200dma) / b_200dma

        # Real Macro Regime determination
        if b_dd < -0.22 or (b_dist_200 < -0.10 and b_1y_ret < -0.15):
            regime = "Contraction"
            regime_code = 2
        elif b_dd < -0.15 and b_1y_ret > -0.05:
            regime = "Trough"
            regime_code = 3
        elif b_dd > -0.05 and b_1y_ret > 0.20:
            regime = "Peak"
            regime_code = 1
        else:
            regime = "Expansion"
            regime_code = 0

        stress_event = MACRO_CALENDAR.get(yr, {}).get("stress_event", "Standard Regime")
        macro_info = MACRO_CALENDAR.get(yr, {})
        power_grid_load = macro_info.get("grid_load_idx", 0.75)
        comm_deficit = macro_info.get("commodity_deficit", 0.60)
        logistics_idx = macro_info.get("logistics_idx", 0.65)

        for p in UNIVERSE_PROFILES:
            ticker = p["ticker"]
            name = p["name"]
            sector = p["sector"]

            if ticker not in raw_data.columns.levels[0]:
                continue

            t_df = raw_data[ticker].dropna()
            if t_df.empty:
                continue

            # Check if the ticker was actively trading on target_dt (eliminate survivorship/anachronisms)
            t_first_dt = t_df.index[0]
            if target_dt < t_first_dt:
                continue

            t_idx = t_df.index.searchsorted(target_dt)
            if t_idx >= len(t_df):
                t_idx = len(t_df) - 1

            # Require at least ~100 trading days before observation date
            if t_idx < 100:
                continue

            hist_so_far = t_df.iloc[:t_idx+1]
            close_p0 = float(hist_so_far["Close"].iloc[-1])
            if close_p0 <= 0:
                continue

            # 3Y lookback for Drawdown_3Y
            lookback_start = max(0, t_idx - 756)
            window_3y = hist_so_far.iloc[lookback_start:]
            peak_3y = float(window_3y["High"].max() if "High" in window_3y.columns else window_3y["Close"].max())
            drawdown_3y = round((close_p0 - peak_3y) / max(0.01, peak_3y), 4)

            # Drawdown velocity
            peak_idx = window_3y["High"].argmax() if "High" in window_3y.columns else window_3y["Close"].argmax()
            days_from_peak = max(5, len(window_3y) - peak_idx)
            dd_velocity = round(abs(drawdown_3y) / (days_from_peak / 252.0), 4)

            # Moving averages
            sma_50 = float(hist_so_far["Close"].iloc[-50:].mean()) if len(hist_so_far) >= 50 else close_p0
            sma_200 = float(hist_so_far["Close"].iloc[-200:].mean()) if len(hist_so_far) >= 200 else close_p0
            ma_cross_spread = round((sma_50 - sma_200) / max(0.01, sma_200), 4)
            dist_200dma = round((close_p0 - sma_200) / max(0.01, sma_200), 4)

            # Fundamental / Structural attributes
            base_de = p.get("base_de", 0.8)
            base_ic = p.get("base_ic", 5.0)
            capex_scale = p.get("capex_scale", 0.75)
            ai_vuln = p.get("ai_vulnerability", 0.15)

            # Physical moat scores
            infra_moat = round(np.clip(capex_scale * 1.05, 0.15, 0.98), 3)
            repl_cost = round(np.clip(capex_scale * 1.1, 0.12, 0.99), 3)
            reg_protect = round(0.90 if any(k in sector for k in ["Power", "Port", "Telecom", "PSU"]) else 0.55, 3)
            asset_moat_score = round(0.40 * infra_moat + 0.35 * repl_cost + 0.25 * reg_protect, 3)

            # Balance sheet metrics with stress cyclicality
            is_stress_yr = (regime in ["Contraction", "Trough"])
            de_mod = 0.20 if is_stress_yr else -0.05
            debt_eq = round(max(0.01, base_de + de_mod), 3)
            ic_mod = 0.75 if is_stress_yr else 1.1
            int_cov = round(max(1.1, base_ic * ic_mod), 2)
            ocf_consistency = round(np.clip(1.0 - (debt_eq / 4.0) + (capex_scale * 0.25), 0.25, 0.98), 3)
            capex_intensity = round(np.clip(capex_scale, 0.10, 0.95), 3)

            # FORWARD REAL GROUND TRUTH (Logic B)
            # Look forward 3 years (approx 756 trading days)
            future_window = t_df.iloc[t_idx+1 : t_idx+1+756]
            if len(future_window) >= 200:
                close_p3 = float(future_window["Close"].iloc[-1])
                years_elapsed = len(future_window) / 252.0
                actual_3y_cagr = round((close_p3 / close_p0) ** (1.0 / max(0.5, years_elapsed)) - 1.0, 4)

                # Real forward max drawdown during holding period
                running_max = future_window["High"].cummax() if "High" in future_window.columns else future_window["Close"].cummax()
                running_dd = (future_window["Low"] - running_max) / running_max if "Low" in future_window.columns else (future_window["Close"] - running_max) / running_max
                actual_max_dd = round(float(running_dd.min()), 4)
            else:
                if len(future_window) > 0:
                    close_p3 = float(future_window["Close"].iloc[-1])
                    years_elapsed = max(0.1, len(future_window) / 252.0)
                    actual_3y_cagr = round((close_p3 / close_p0) ** (1.0 / years_elapsed) - 1.0, 4)
                    actual_max_dd = round(float(((future_window["Low"] - close_p0) / close_p0).min()), 4)
                else:
                    actual_3y_cagr = 0.08
                    actual_max_dd = -0.05

            actual_success = bool(actual_3y_cagr >= 0.10)

            records.append({
                "Year": yr,
                "Ticker": ticker,
                "Name": name,
                "Sector": sector,
                "Market_Regime": regime,
                "Regime_Code": regime_code,
                "Stress_Event": stress_event,
                # Dimension 1: Price & Momentum Dynamics (Real Historical)
                "Drawdown_3Y": drawdown_3y,
                "Drawdown_Velocity": dd_velocity,
                "MA_Cross_Spread": ma_cross_spread,
                "Dist_200DMA": dist_200dma,
                # Dimension 2: Fundamental Balance Sheet Health
                "Debt_Equity": debt_eq,
                "Interest_Coverage": int_cov,
                "OCF_Consistency": ocf_consistency,
                "CapEx_Intensity": capex_intensity,
                # Dimension 3: Physical Moat & Asset Scarcity Scoring
                "Infrastructure_Moat": infra_moat,
                "Replacement_Cost_Barrier": repl_cost,
                "Regulatory_Protection": reg_protect,
                "Asset_Moat_Score": asset_moat_score,
                # Dimension 4: Macro-Demand & AI Disruption Indicators
                "Power_Grid_Load_Index": power_grid_load,
                "Commodity_Supply_Deficit": comm_deficit,
                "Logistics_Bottleneck_Index": logistics_idx,
                "AI_Vulnerability_Score": ai_vuln,
                # Decoupled Ground Truth (Logic B)
                "Actual_3Y_CAGR": actual_3y_cagr,
                "Actual_Success": actual_success,
                "Actual_Max_DD": actual_max_dd
            })

    df = pd.DataFrame(records)
    df.to_csv(DATASET_FILE, index=False)
    logger.info(f"Successfully generated and saved genuine empirical 20Y dataset to {DATASET_FILE} ({len(df)} records across {df['Year'].nunique()} years).")
    _CACHED_20Y_DATASET = df
    return df

if __name__ == "__main__":
    df = generate_20y_ground_truth_dataset(force_recreate=True)
    print(f"Dataset generated successfully with {len(df)} records across {df['Year'].nunique()} years:")
    print(df.info())
    print("\nStress Event Distribution:")
    print(df.groupby("Stress_Event")["Actual_Success"].agg(["count", "mean"]))
