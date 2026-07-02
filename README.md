# Beyond the Hype: Can Financials Predict Semiconductor Valuations?

Final individual project for the Ironhack Data Analytics bootcamp. The project asks a simple question with a not-so-simple answer: **do traditional financial fundamentals actually explain and predict how the market values semiconductor companies or has the AI boom decoupled price from financials?**

> To what extent can historical and financial fundamentals explain and predict next-quarter valuation multiples and market capitalization in publicly traded semiconductor firms?

- **EDA question:** Does the market valuation of semiconductor companies reflect their underlying financial performance?
- **ML question:** Can historical financial and market indicators forecast a company's next-quarter market cap?

## Repository structure

```
semiconductor-forecasting/
├── main.py
├── pyproject.toml
├── uv.lock
├── config.yaml               # paths to clean data files, read by every notebook
├── README.md
├── src/                       # helper modules
├── notebooks/
│   ├── data_wrangling_cleaning.ipynb
│   ├── basic_EDA.ipynb
│   ├── feature_engineering_ML_1.ipynb
│   └── ML_2.ipynb
└── data/
    ├── clean/                 # cleaned CSVs consumed/produced by the notebooks
    └── graphs/                # exported plots (.png) from EDA and ML notebooks
```

## Data

**Companies (10 tickers):** NVDA, AVGO, MU, AMD, INTC, TXN, MRVL, QCOM, ADI, NXPI. They were selected upon the basis of being the largest US-listed semiconductor firms per Yahoo Finance's industry screener.

**Sources:**
- **SEC EDGAR XBRL `companyfacts` API** — quarterly financial statement data (revenue, gross profit, EBIT, net income, R&D, operating cash flow, capex, cash, debt, equity, D&A) pulled directly by CIK per company.
- **`yfinance`** — quarterly closing price and shares outstanding, used to build market cap; also enterprise value.

**Coverage:** 175 company-quarters, 2022 Q1 through early 2026, 45 raw/derived columns before feature engineering.

**Cleaning highlights (`data_wrangling_cleaning.ipynb`):**
- Company-by-company XBRL tag reconciliation — different companies file under different GAAP concept tags (e.g. `Revenues` vs `RevenueFromContractWithCustomerExcludingAssessedTax`), handled with fallback logic per concept.
- Manual reconstruction of missing quarterly values where companies only filed figures annually or YTD (e.g. NVIDIA capex, D&A gaps), backed out from cumulative filings.
- NVIDIA's fiscal year (ends in January) realigned so its "quarters" line up with the other 9 calendar-year filers, plus manual adjustment for NVIDIA's 10:1 stock split in June 2024.
- TTM (trailing-twelve-month) rollups for all flow metrics, YoY growth rates, margins, ROE, and valuation multiples (P/E, EV/EBITDA, P/S) computed from the cleaned quarterly base.

## Notebooks

### 1. `data_wrangling_cleaning.ipynb`
Pulls raw SEC XBRL and yfinance data, reconciles tag inconsistencies, fills structural gaps, derives market cap / TTM / margin / growth / valuation-multiple columns, and writes the clean quarterly panel to `data/clean/`.

### 2. `basic_EDA.ipynb`
Exploratory analysis on the clean panel:
- Distribution and outlier checks on valuation multiples (heavily right-skewed).
- Market cap and stock price evolution per company (linear and log scale). NVDA grew ~147x over 4 years, dwarfing the rest of the sector.
- Financial fundamentals evolution and a margin heatmap across companies and quarters (NVDA's gross margin expanded 56%→74% while INTC's collapsed 57%→30%).
- Market value vs. financial value: log-log market cap vs. revenue TTM (Pearson r = 0.66), with post-2024 points sitting visibly above the trend line.
- Company comparison on margin efficiency and valuation multiples.
- **Hypothesis testing**: KS test + Q-Q plot for normality of market cap → not normal → log-transform and use Spearman/rank stats; Pearson correlation of financial metrics vs. (log) market cap, all significant; one-tailed two-sample t-test confirming AI-adjacent companies (NVDA, AMD, AVGO, MU) have significantly higher mean market cap than the rest since 2023 (t = 4.12, p = 0.0001).

### 3. `feature_engineering_ML_1.ipynb`
- Lag-1 features for all financial ratios/metrics (autocorrelation analysis via ACF confirmed lag-1 as the strongest predictor, no seasonal lag-4 effect), log transforms, cyclical sin/cos encoding of fiscal quarter, one-hot encoding of ticker, and a Yeo-Johnson power transform of the market cap target.
- Multicollinearity reduction via iterative VIF elimination (starting VIF as high as 762 for `ebitda_ttm_lag1`) down to a clean, low-VIF feature set: `free_cash_flow_ttm_lag1`, `net_margin_lag1`, `revenue_growth_yoy_lag1`, `ebitda_growth_yoy_lag1`, `net_income_growth_yoy_lag1`, `roe_lag1`, `rd_to_revenue_lag1`, `capex_to_revenue_lag1`, `debt_to_ebitda_lag1` (all VIF < 8).
- Chronological train/test split, scaling/power-transform pipeline, and baseline models trained on financials only: **Linear Regression, KNN, Random Forest, and XGBoost**, each with a tuned variant. Random Forest (tuned) was the best financials-only model (R² = 0.71); XGBoost underperformed RF, most likely due to the small sample size (80–100 training rows).
- Feature importance and error analysis: the largest prediction errors cluster in NVDA, AVGO, MU, AMD, ADI; the companies whose valuations are most driven by AI-related market sentiment rather than reported fundamentals.

### 4. `ML_2.ipynb`
Extends the modeling with lagged **market cap** itself as a predictor:
- ACF confirms strong autocorrelation in market cap at lag-1 (0.60) and lag-2 (0.33), motivating `market_cap_lag1` / `market_cap_lag2` as features.
- Three Random Forest variants, each tuned via `RandomizedSearchCV` with `TimeSeriesSplit` cross-validation:
  - **Market-only** (lagged market cap + ticker dummies)
  - **Financials + market** (financial features + lagged market cap + ticker dummies)
- Final model comparison across all models from both notebooks (see below).

## Model results

| Model | Features | Test R² | MAE |
|---|---|---|---|
| Linear Regression | Financials | 0.13 | $609B |
| KNN | Financials | 0.48 | $482B |
| Random Forest | Financials | 0.67 | $365B |
| Random Forest (tuned) | Financials | 0.71 | $353B |
| XGBoost | Financials | 0.35 | $490B |
| XGBoost (tuned) | Financials | 0.12 | $576B |
| Random Forest | Market lag | 0.72 | $293B |
| **Random Forest (tuned)** | **Market lag** | **0.82** | **$281B** |
| Random Forest | Financials + market | 0.74 | $288B |
| Random Forest (tuned) | Financials + market | 0.76 | $312B |

**Takeaway:** the best model overall uses only lagged market cap, not financial fundamentals. Financials correlate with valuation but add little independent predictive power once the market already knows the company's recent price. Fundamentals can explain *where a company's value comes from*, but the market's own momentum is the better *predictor*.

## Conclusions

- Financial fundamentals **are** significantly correlated with market cap (Pearson r up to 0.93 for EBITDA TTM), but the relationship is far from linear and far from complete.
- A tuned Random Forest using only **lagged market cap** (last two quarters) explains **82% of the variance** (R² = 0.82) in next-quarter market cap. That's better than any model built purely on financial fundamentals (best financials-only model: R² = 0.71).
- Adding financial fundamentals on top of lagged market cap barely helps (R² = 0.76) and overfits more, meaning most of the predictive signal in "financials" was really just a proxy for "the company was already big."
- Three companies, **NVDA, AVGO, MU**, are the hardest to predict from fundamentals alone, because their valuations from 2023 onward are driven by AI sentiment rather than reported financials. A two-sample t-test confirms AI-adjacent companies (NVDA, AMD, AVGO, MU) had significantly higher mean market caps than the rest of the sector from 2023 onward (t = 4.12, p < 0.001).

## Tools

Python, `pandas`, `numpy`, `yfinance`, SEC EDGAR XBRL API, `scikit-learn`, `xgboost`, `statsmodels`, `scipy`, `matplotlib`/`seaborn`, `uv` for environment management.
