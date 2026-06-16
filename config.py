# fund_replication/config.py
# Universe configuration for fund replication analysis.

# ── Passive ETF universe ──────────────────────────────────────────────────────
# All tickers confirmed present in sharadar_fundprices (DuckDB).
PASSIVE_ETFS: dict[str, str] = {
    # ── Broad US equity ──────────────────────────────────────────────────────
    "SPY":  "S&P 500 (SPDR, 0.09%)",
    "QQQ":  "Nasdaq 100 (Invesco, 0.20%)",
    "VTI":  "US Total Market (Vanguard, 0.03%)",
    "IWB":  "Russell 1000 (iShares, 0.15%)",
    "IWM":  "Russell 2000 Small (iShares, 0.19%)",
    "MDY":  "S&P MidCap 400 (SPDR, 0.23%)",
    "IJH":  "S&P MidCap 400 (iShares, 0.05%)",
    "IJR":  "S&P SmallCap 600 (iShares, 0.06%)",
    # ── Style ────────────────────────────────────────────────────────────────
    "IVW":  "S&P 500 Growth (iShares)",
    "IVE":  "S&P 500 Value (iShares)",
    "IWF":  "Russell 1000 Growth (iShares)",
    "IWD":  "Russell 1000 Value (iShares)",
    "IWO":  "Russell 2000 Growth (iShares)",
    "IWN":  "Russell 2000 Value (iShares)",
    "IJJ":  "S&P MidCap 400 Value (iShares)",
    "VTV":  "US Large Value (Vanguard, 0.04%)",
    "VUG":  "US Large Growth (Vanguard, 0.04%)",
    # ── Factor ETFs ──────────────────────────────────────────────────────────
    "MTUM": "Momentum Factor (iShares MSCI, 0.15%)",
    "QUAL": "Quality Factor (iShares MSCI, 0.15%)",
    "USMV": "Min Volatility (iShares MSCI, 0.15%)",
    "VLUE": "Value Factor (iShares MSCI, 0.15%)",
    # ── International equity ─────────────────────────────────────────────────
    "EFA":  "Developed Intl EAFE (iShares, 0.32%)",
    "VEA":  "Developed Intl FTSE (Vanguard, 0.05%)",
    "ACWI": "All-World MSCI (iShares, 0.32%)",
    "EEM":  "Emerging Markets (iShares, 0.70%)",
    "IEMG": "EM Broad Core (iShares, 0.09%)",
    "VWO":  "EM (Vanguard, 0.08%)",
    "EWJ":  "Japan (iShares, 0.50%)",
    # ── Fixed income ─────────────────────────────────────────────────────────
    "TLT":  "Long Treasury 20yr (iShares, 0.15%)",
    "IEF":  "Interm Treasury 7-10yr (iShares, 0.15%)",
    "SHY":  "Short Treasury 1-3yr (iShares, 0.15%)",
    "GOVT": "US Treasury All Maturities (iShares)",
    "TIP":  "TIPS Inflation-Protected (iShares)",
    "LQD":  "Investment Grade Corp (iShares, 0.14%)",
    "HYG":  "High Yield Corp (iShares, 0.49%)",
    "EMB":  "EM USD Bonds (iShares, 0.39%)",
    "MUB":  "Municipal Bonds (iShares, 0.07%)",
    "BND":  "Total US Bond (Vanguard, 0.03%)",
    "BNDX": "Total Intl Bond hedged (Vanguard, 0.07%)",
    "VCSH": "Corp Bond Short-Term (Vanguard, 0.04%)",
    "CWB":  "Convertible Bonds (SPDR)",
    # ── Sectors ──────────────────────────────────────────────────────────────
    "XLK":  "Technology (SPDR)",
    "XLF":  "Financials (SPDR)",
    "XLE":  "Energy (SPDR)",
    "XLV":  "Health Care (SPDR)",
    "XLY":  "Consumer Discret (SPDR)",
    "XLP":  "Consumer Staples (SPDR)",
    "XLI":  "Industrials (SPDR)",
    "XLB":  "Materials (SPDR)",
    "XLU":  "Utilities (SPDR)",
    "XLRE": "Real Estate (SPDR)",
    "SMH":  "Semiconductors (VanEck)",
    "FDN":  "Internet (First Trust)",
    "KRE":  "Regional Banks (SPDR)",
    "XME":  "Metals & Mining (SPDR)",
    "XOP":  "Oil & Gas E&P (SPDR)",
    "XRT":  "Retail (SPDR)",
    "ITB":  "Homebuilders (iShares)",
    "XHB":  "Homebuilders (SPDR)",
    "JETS": "Airlines (US Global)",
    # ── Real assets / alternatives ────────────────────────────────────────────
    "VNQ":  "US Real Estate (Vanguard, 0.13%)",
    "GLD":  "Gold (SPDR, 0.40%)",
    "DBC":  "Broad Commodities (Invesco, 0.85%)",
    "AMLP": "MLP Energy Infrastructure (Alerian)",
    # ── Income / hybrid ──────────────────────────────────────────────────────
    "VYM":  "High Dividend Yield (Vanguard, 0.06%)",
    "PFF":  "US Preferred Stock (iShares, 0.46%)",
    # ── Dividend-focused ETFs (from fundstoevaluate.csv) ─────────────────────
    "DES":  "US SmallCap Dividend (WisdomTree, 0.38%)",
    "DLN":  "US LargeCap Dividend (WisdomTree, 0.28%)",
    "DON":  "US MidCap Dividend (WisdomTree, 0.38%)",
    "SDY":  "S&P Dividend Aristocrats (SPDR, 0.35%)",
    # ── Leveraged ETFs ────────────────────────────────────────────────────────
    # These exist in DuckDB and allow the vol-constraint optimizer to match
    # high-volatility active funds (ARK, concentrated growth) without leverage.
    # NOTE: daily-rebalanced 2×/3× products — weights represent style exposure,
    # not a recommended buy-and-hold allocation.
    "QLD":  "Nasdaq 100 2× (ProShares Ultra QQQ, 0.95%)",        # 2006 — longest leveraged-QQQ history
    "TECL": "Technology 3× (Direxion, 1.04%)",                    # 2008 — key for ARK/tech decomposition
    "SPXL": "S&P 500 3× (Direxion, 1.01%)",                       # 2008 — high-vol broad market
    "FAS":  "Financials 3× (Direxion, 1.06%)",                    # 2008 — financial-sector funds
    "TNA":  "Small Cap 3× (Direxion Russell 2000, 1.06%)",         # 2008 — small-cap active funds
    "TMF":  "Long Treasury 3× (Direxion 20yr, 1.09%)",            # 2009 — balanced/multi-asset funds
    "TQQQ": "Nasdaq 100 3× (ProShares UltraPro QQQ, 0.88%)",      # 2010 — primary ARK/growth proxy
    "SOXL": "Semiconductors 3× (Direxion, 0.76%)",                # 2010 — semi exposure in ARK/tech
    "URTY": "Russell 2000 3× (ProShares UltraPro, 0.95%)",        # 2010 — small-cap growth leverage
    "CURE": "Healthcare 3× (Direxion, 1.04%)",                    # 2011 — healthcare / ARKG proxy
}

# ── Active ETFs (in DuckDB, from inception ~2014) ────────────────────────────
# Source: sharadar_fundprices.  ER 0.75-0.76%.
ACTIVE_ETFS: dict[str, str] = {
    "ARKK": "ARK Innovation (0.75% ER, Morningstar Neutral)",
    "ARKW": "ARK Next Gen Internet (0.76% ER, Morningstar Neutral)",
    "ARKG": "ARK Genomic Revolution (0.75% ER, Morningstar Neutral)",
    "ARKQ": "ARK Autonomous Tech (0.75% ER, Morningstar Bronze)",
    "ARKF": "ARK Fintech Innovation (0.75% ER, inception 2019)",
}

# ── Active mutual funds (pulled via yfinance) ─────────────────────────────────
# Full list from fundstoevaluate.csv + earlier research funds.
# er/stars/category are optional — fill in as known. Missing fields default to
# er=None, stars=None, category="Unknown" in run.py.
ACTIVE_MUTUAL_FUNDS: dict[str, dict] = {
    # ── Fidelity ──────────────────────────────────────────────────────────────
    "FEQIX": {"name": "Fidelity Equity Income",                "er": 0.0057, "stars": 3, "category": "Large Value"},
    "FMAGX": {"name": "Fidelity Magellan",                     "er": 0.0076, "stars": 3, "category": "Large Growth"},
    "FFIDX": {"name": "Fidelity 500 Index",                    "er": 0.0015, "stars": 5, "category": "Large Blend (passive)"},
    "FAGIX": {"name": "Fidelity Capital & Income",             "er": 0.0067, "stars": 4, "category": "High Yield Bond"},
    "FAGOX": {"name": "Fidelity Advisor Growth Opps",          "er": None,   "stars": None, "category": "Large Growth"},
    "FSTEX": {"name": "Fidelity Stock Selector Small Cap",     "er": None,   "stars": None, "category": "Small Blend"},
    "FBGRX": {"name": "Fidelity Blue Chip Growth",             "er": 0.0048, "stars": 5, "category": "Large Growth"},
    "FCNTX": {"name": "Fidelity Contrafund",                   "er": 0.0074, "stars": 5, "category": "Large Growth"},
    "FDGRX": {"name": "Fidelity Growth Company",               "er": 0.0083, "stars": 5, "category": "Large Growth"},
    "FKGRX": {"name": "Franklin Growth",                       "er": None,   "stars": None, "category": "Large Growth"},
    "FLCSX": {"name": "Fidelity Large Cap Stock",              "er": None,   "stars": None, "category": "Large Blend"},
    "FOCPX": {"name": "Fidelity OTC Portfolio",                "er": 0.0083, "stars": 4, "category": "Large Growth"},
    # ── Vanguard ──────────────────────────────────────────────────────────────
    "VDIGX": {"name": "Vanguard Dividend Growth",              "er": 0.0027, "stars": 5, "category": "Large Growth"},
    "VEXAX": {"name": "Vanguard Extended Market Index",        "er": 0.0006, "stars": 5, "category": "Mid Blend (passive)"},
    "VSEQX": {"name": "Vanguard Strategic Equity",             "er": 0.0016, "stars": 4, "category": "Mid Blend"},
    "VTCLX": {"name": "Vanguard Tax-Managed Cap Apprec",       "er": 0.0009, "stars": 5, "category": "Large Blend (passive)"},
    "VEUSX": {"name": "Vanguard European Stock Index",         "er": 0.0010, "stars": 4, "category": "Europe Stock (passive)"},
    "VWELX": {"name": "Vanguard Wellington",                   "er": 0.0025, "stars": 5, "category": "Moderate Allocation"},
    "VWNFX": {"name": "Vanguard Windsor",                      "er": 0.0033, "stars": 3, "category": "Large Value"},
    # ── T. Rowe Price ─────────────────────────────────────────────────────────
    "PRDGX": {"name": "T. Rowe Price Dividend Growth",         "er": 0.0063, "stars": 5, "category": "Large Growth"},
    "PRWCX": {"name": "T. Rowe Price Capital Appreciation",    "er": 0.0067, "stars": 5, "category": "Moderate Allocation"},
    "TRBCX": {"name": "T. Rowe Price Blue Chip Growth",        "er": 0.0069, "stars": 5, "category": "Large Growth"},
    "TRVLX": {"name": "T. Rowe Price Value",                   "er": 0.0076, "stars": 4, "category": "Large Value"},
    "PRFDX": {"name": "T. Rowe Price Equity Income",           "er": 0.0064, "stars": 3, "category": "Large Value"},
    "PRGFX": {"name": "T. Rowe Price Growth Stock",            "er": 0.0067, "stars": 4, "category": "Large Growth"},
    # ── Oakmark / Harris ──────────────────────────────────────────────────────
    "OAKMX": {"name": "Oakmark Fund Investor",                 "er": 0.0089, "stars": 5, "category": "Large Blend"},
    "OAKLX": {"name": "Oakmark International Investor",        "er": 0.0096, "stars": 4, "category": "Foreign Large Blend"},
    # ── Artisan ───────────────────────────────────────────────────────────────
    "ARTKX": {"name": "Artisan International Value Inv",       "er": 0.0124, "stars": 5, "category": "Foreign Large Value"},
    "ARTLX": {"name": "Artisan Large Cap Growth Inv",          "er": 0.0110, "stars": None, "category": "Large Growth"},
    "ARTMX": {"name": "Artisan Mid Cap Inv",                   "er": 0.0115, "stars": 4, "category": "Mid Growth"},
    # ── American Funds / Capital Group ────────────────────────────────────────
    "AGRFX": {"name": "American Funds Growth Fund of Amer A",  "er": 0.0065, "stars": 4, "category": "Large Growth"},
    "ACGIX": {"name": "American Century Growth Inv",           "er": 0.0100, "stars": 3, "category": "Large Growth"},
    # ── Dodge & Cox ───────────────────────────────────────────────────────────
    "DODGX": {"name": "Dodge & Cox Stock",                     "er": 0.0052, "stars": 4, "category": "Large Value"},
    # ── Harbor ────────────────────────────────────────────────────────────────
    "HACAX": {"name": "Harbor Capital Appreciation Ret",       "er": 0.0101, "stars": 5, "category": "Large Growth"},
    "HGOAX": {"name": "Harbor Growth A",                       "er": None,   "stars": None, "category": "Large Growth"},
    # ── Baron ─────────────────────────────────────────────────────────────────
    "BGRFX": {"name": "Baron Growth Retail",                   "er": 0.0129, "stars": 4, "category": "Mid Growth"},
    # ── MFS ───────────────────────────────────────────────────────────────────
    "MACGX": {"name": "MFS Growth A",                          "er": 0.0077, "stars": 4, "category": "Large Growth"},
    "MDFGX": {"name": "MFS Global Growth A",                   "er": None,   "stars": None, "category": "World Large Stock"},
    "MDDVX": {"name": "MFS Blended Research US Equity A",      "er": None,   "stars": None, "category": "Large Blend"},
    # ── Janus Henderson ───────────────────────────────────────────────────────
    "JLGRX": {"name": "Janus Henderson US Equity Ret",         "er": 0.0094, "stars": 4, "category": "Large Growth"},
    "JUESX": {"name": "Janus Henderson Enterprise Ret",        "er": 0.0091, "stars": 4, "category": "Mid Growth"},
    # ── Primecap / Odyssey ────────────────────────────────────────────────────
    "POAGX": {"name": "Primecap Odyssey Aggressive Growth",    "er": 0.0063, "stars": 5, "category": "Mid Growth"},
    "POGRX": {"name": "Primecap Odyssey Growth",               "er": 0.0063, "stars": 5, "category": "Large Growth"},
    # ── Gabelli ───────────────────────────────────────────────────────────────
    "GABAX": {"name": "Gabelli Asset A",                       "er": 0.0136, "stars": 3, "category": "Large Blend"},
    "GABGX": {"name": "Gabelli Growth A",                      "er": None,   "stars": None, "category": "Large Growth"},
    # ── Neuberger Berman ──────────────────────────────────────────────────────
    "NBGNX": {"name": "Neuberger Berman Genesis Inv",          "er": 0.0085, "stars": 4, "category": "Small Growth"},
    # ── William Blair ─────────────────────────────────────────────────────────
    "WGROX": {"name": "William Blair Growth N",                "er": 0.0121, "stars": 4, "category": "Large Growth"},
    # ── Wells Fargo / Allspring ───────────────────────────────────────────────
    "WAAEX": {"name": "Allspring Growth A",                    "er": None,   "stars": None, "category": "Large Growth"},
    # ── AllianceBernstein ─────────────────────────────────────────────────────
    "ADGAX": {"name": "AB Large Cap Growth A",                 "er": 0.0111, "stars": 4, "category": "Large Growth"},
    # ── Lateef ────────────────────────────────────────────────────────────────
    "LSGRX": {"name": "Lateef Asset Mgmt Equity Retail",       "er": None,   "stars": None, "category": "Large Blend"},
    # ── Others ────────────────────────────────────────────────────────────────
    "SSGLX": {"name": "SEI Large Cap Growth A",                "er": None,   "stars": None, "category": "Large Growth"},
    "PRILX": {"name": "Principal Large Cap Growth I",          "er": None,   "stars": None, "category": "Large Growth"},
    "AUIAX": {"name": "AB US Strategic Research A",            "er": None,   "stars": None, "category": "Large Blend"},
    "BSCFX": {"name": "Baird Mid Cap Growth Inv",              "er": None,   "stars": None, "category": "Mid Growth"},
    "CDDRX": {"name": "Carillon Scout Mid Cap Ret",            "er": None,   "stars": None, "category": "Mid Growth"},
    "CIPMX": {"name": "Columbia Integrated Large Cap Value A", "er": None,   "stars": None, "category": "Large Value"},
    "CIPSX": {"name": "Columbia Integrated Small Cap Val A",   "er": None,   "stars": None, "category": "Small Value"},
    "LMTIX": {"name": "Lord Abbett Mid Cap Stock I",           "er": None,   "stars": None, "category": "Mid Blend"},
    "RAFGX": {"name": "Rowe AF Growth A",                      "er": None,   "stars": None, "category": "Large Growth"},
    "RFNGX": {"name": "Rowe Financial Services Growth",        "er": None,   "stars": None, "category": "Financial"},
    "RGAGX": {"name": "American Century Large Cap Gr A",       "er": None,   "stars": None, "category": "Large Growth"},
    "RICGX": {"name": "Royce International Premier Svc",       "er": None,   "stars": None, "category": "Foreign Small/Mid Growth"},
    "RMFGX": {"name": "RMF Growth A",                          "er": None,   "stars": None, "category": "Large Growth"},
    "RPEAX": {"name": "Royce Pennsylvania Mutual Svc",         "er": None,   "stars": None, "category": "Small Blend"},
    "WPASX": {"name": "WPG Partners Select Shares No Load",    "er": None,   "stars": None, "category": "Small Blend"},
    # ── Research funds (from earlier analysis) ────────────────────────────────
    "PRBLX": {"name": "Parnassus Core Equity Investor",        "er": 0.0081, "stars": 5, "category": "Large Blend"},
    "FAIRX": {"name": "Fairholme Fund",                        "er": 0.0100, "stars": 3, "category": "Large Value"},
}

# ── Analysis parameters ───────────────────────────────────────────────────────
DEFAULT_START       = "2005-01-01"   # earliest common start across mutual funds
DEFAULT_END         = "2025-12-31"
TRAIN_MONTHS        = 36             # rolling window training length
REBAL_MONTHS        = 12             # rolling rebalance frequency (annual; change back to 3 for quarterly)
OOS_START           = "2015-01-01"   # IS/OOS split date
MIN_OVERLAP_MONTHS  = 24             # minimum months to run regression
LASSO_N_ALPHAS      = 50             # number of alpha values to CV over

# Minimum replica volatility as a fraction of fund volatility.
# 0.0 = unconstrained (current default).
# 1.0 = force replica vol >= fund vol.
# 1.1 = add 10% buffer to account for fund concentration risk.
# Raises tracking error slightly; set to 0 to disable.
MIN_VOL_RATIO       = 1.0
# Soft cap on benchmark vol relative to fund vol in FI benchmark selection.
# Scores are multiplied by min(1, MAX_BM_VOL_RATIO / vol_ratio) above the cap.
MAX_BM_VOL_RATIO    = 1.5

# ── Grade / scoring parameters ────────────────────────────────────────────────
# Period weights for weighted-average Sharpe used in scoring (re-normalised if period unavailable)
GRADE_TIME_WEIGHTS: dict[str, float] = {
    "10y":       0.30,
    "full_hist": 0.20,   # entire history, static IS weights vs full-period benchmark
    "full":      0.10,   # OOS-only full period
    "5y":        0.20,
    "3y":        0.15,
    "1y":        0.05,
}
GRADE_BLEND_REP_WT: float = 0.50   # replica's share in the Sharpe blend (vs bm_adj)
GRADE_HIGH_DIFF:    float =  0.20  # Sharpe diff where score reaches 5.0
GRADE_LOW_DIFF:     float = -0.20  # Sharpe diff where score drops to 1.0

# R²-based "closet index" penalty: a high full-history R² between the fund and a
# single static (never-rebalanced) passive blend means the fund hasn't materially
# changed style/allocation over time — a buy-and-hold mix would have replicated it
# just as well, independent of whatever Sharpe diff it happens to show.
GRADE_R2_PENALTY_THRESHOLD: float = 0.90   # full-history R² below this -> no penalty
GRADE_R2_PENALTY_MAX:       float = 1.0    # max score points deducted as R² -> 1.0

# ── Category → benchmark mapping ──────────────────────────────────────────────
# Maps Morningstar category to the most appropriate passive benchmark ETF.
# When no explicit benchmark override is given, this takes precedence over the
# vol-based SPY/QQQ auto-select so mid-cap, small-cap, international, and fixed
# income funds are compared fairly against a style-matched benchmark.
#
# Large-cap funds (Large Blend/Growth/Value) are intentionally OMITTED — the
# vol-based auto-select (SPY if fund vol < QQQ vol, else QQQ) handles them well.
#
# For unconfigured tickers where yfinance returns no category, app.py uses
# _infer_fi_category() to detect fixed income from RBSA weights automatically.
CATEGORY_BM_MAP: dict[str, str] = {
    # Mid-cap
    "Mid Growth":                    "MDY",
    "Mid Blend":                     "MDY",
    "Mid-Cap Growth":                "MDY",
    "Mid Blend (passive)":           "MDY",
    # Small-cap
    "Small Growth":                  "IWM",
    "Small Blend":                   "IWM",
    "Small Value":                   "IWN",
    # International / global
    "Foreign Large Blend":           "EFA",
    "Foreign Large Growth":          "EFA",
    "Foreign Large Value":           "EFA",
    "Foreign Small/Mid Growth":      "EFA",
    "Europe Stock":                  "EFA",
    "Europe Stock (passive)":        "EFA",
    "World Large Stock":             "ACWI",
    # Fixed income — credit
    "High Yield Bond":               "HYG",
    "Corporate Bond":                "LQD",
    "Bank Loan":                     "HYG",
    "Convertibles":                  "CWB",
    "Preferred Stock":               "PFF",
    # Fixed income — core / multi-sector
    "Intermediate Core Bond":        "BND",
    "Intermediate Core-Plus Bond":   "LQD",
    "Intermediate-Term Bond":        "BND",
    "Multisector Bond":              "BND",
    "Nontraditional Bond":           "BND",
    "Total Return Bond":             "BND",
    # Fixed income — government / duration
    "Long Government":               "TLT",
    "Long-Term Bond":                "TLT",
    "Intermediate Government":       "IEF",
    "Short Government":              "SHY",
    "Short-Term Bond":               "SHY",
    "Ultrashort Bond":               "SHY",
    "Inflation-Protected Bond":      "TIP",
    # Fixed income — specialty
    "Muni National Interm":          "MUB",
    "Muni National Long":            "MUB",
    "Muni National Short":           "MUB",
    "Muni California Interm/Short":  "MUB",
    "Muni California Long":          "MUB",
    "Emerging Markets Bond":         "EMB",
    "World Bond":                    "BND",
    # Equity sector
    "Financial":                     "XLF",
}

# Fixed income ETFs present in the passive universe — used by _infer_fi_category()
# to auto-detect bond funds when Morningstar category is unavailable from yfinance.
FI_ETFS: frozenset = frozenset({
    "TLT", "IEF", "SHY", "GOVT", "TIP",   # government / duration
    "LQD", "HYG", "VCSH", "CWB",           # credit
    "BND", "BNDX",                          # broad bond
    "MUB", "EMB",                           # muni / EM
})
