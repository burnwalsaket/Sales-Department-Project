# Rossmann Store Sales Forecasting

Time-series sales forecasting for Rossmann drug store chain using Facebook Prophet, with an end-to-end exploratory data analysis (EDA) pipeline covering ~1.1 million daily sales records across 1,115 stores.

## Business Problem

Rossmann operates over 3,000 drug stores across 7 European countries. Store managers currently forecast daily sales up to six weeks in advance based on individual judgment, leading to inconsistent accuracy. This project builds a data-driven forecasting model that accounts for promotions, competition, school and state holidays, seasonality, and store-specific trends — enabling more reliable sales predictions to support inventory planning, staffing, and revenue forecasting.

## Dataset

Source: [Rossmann Store Sales — Kaggle](https://www.kaggle.com/c/rossmann-store-sales/data)

Two files are included in this repository:
- **train.csv** (1,017,209 rows) — daily observations across 1,115 stores (Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday, DayOfWeek, Date)
- **store.csv** (1,115 rows) — store-level metadata (StoreType, Assortment, CompetitionDistance, CompetitionOpenSince[Month/Year], Promo2, Promo2Since[Week/Year], PromoInterval)

## Tech Stack

- **Python** — NumPy, Pandas
- **Visualization** — Matplotlib, Seaborn
- **Forecasting** — Prophet (Facebook/Meta's time-series forecasting library)

## Workflow

1. **Data Import & Inspection** — loaded and profiled both training and store datasets (`.info()`, `.describe()`, null checks)
2. **Data Cleaning**
   - Removed records for closed stores (`Open == 0`) before modeling, since closed-store days contribute zero sales and would bias the model
   - Imputed missing `CompetitionDistance` values with the column mean
   - Filled missing `Promo2`-related fields (`Promo2SinceWeek`, `Promo2SinceYear`, `PromoInterval`) and competition-open date fields with 0, based on the pattern that these were missing specifically for stores not enrolled in that program
3. **Exploratory Data Analysis**
   - Missing-value heatmaps for both datasets
   - Distribution histograms for all numeric features
   - Correlation analysis (heatmap + ranked correlations against `Sales`)
   - Merged sales and store data on `Store` for combined analysis
   - Extracted `Year`, `Month`, `Day` from the date field for seasonality analysis
   - Sales and customer trends by month, day of month, and day of week
   - Sales/customer distribution by `StoreType`
   - Promo impact analysis via bar plots and box plots
4. **Modeling — Baseline** — Prophet model trained per store on historical `Date`/`Sales` pairs, forecasting up to 60 days ahead
5. **Modeling — Holiday-Aware** — extended the Prophet model with a custom holidays dataframe combining school holidays and state holidays (public, Easter, Christmas), forecasting up to 90 days ahead for a selected store

## Key Insights from EDA

- Stores are open roughly 80% of the time; excluding closed-store days raised average daily sales from ~5,773 to ~6,955 (with average customers rising from ~633 to ~762)
- `Customers` is the strongest positive driver of `Sales`, followed by `Promo`; `Promo2` (the continuing promotion) shows little to no correlation with sales
- Sales and customer counts peak around the Christmas period and around the start/end of the month, with a dip near the 24th
- Running a promotion is associated with a visible increase in both sales and customer counts

## How to Run

1. Clone the repository (`train.csv` and `store.csv` are included)
2. Install dependencies:
   ```bash
   pip install numpy pandas seaborn matplotlib prophet
   ```
3. Run the notebook cells sequentially — Prophet model fitting is computationally intensive and is run per individual store
4. To forecast a different store, change the `Store_ID` argument passed to `sales_predictions()`

## Project Structure

```
├── Sales_Department_Project.ipynb   # Full analysis and modeling notebook
├── train.csv                        # Daily sales data (1,017,209 rows)
├── store.csv                        # Store metadata (1,115 rows)
└── README.md
```

## Future Improvements

- Evaluate forecast accuracy quantitatively (e.g., MAE, RMSE, MAPE) against a held-out test period rather than relying on visual inspection of forecast plots
- Compare Prophet against tree-based approaches (e.g., Random Forest, XGBoost, LightGBM) using engineered lag and rolling-window features
- Extend holiday-aware modeling to all stores rather than a single selected store, and evaluate per-`StoreType` model performance
- Incorporate `CompetitionDistance` and `CompetitionOpenSince` as regressors, since Prophet supports additional regressors beyond holidays

## Author

Saket
