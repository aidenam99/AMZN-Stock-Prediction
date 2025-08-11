# AMZN Stock Price Prediction Model (2010 - 2020)
This project uses **multiple machine learning algorithms** to predict Amazon's closing stock price based on various technical indicators, lagged prices, and market index correlations.  
The workflow includes **data preprocessing, model training, and performance evaluation** across several algorithms:  

- **XGBoost**
- **Artificial Neural Networks (ANN)**
- **Random Forest**
- **Support Vector Regression (SVR)**

## Data
The dataset contains historical Amazon stock prices and calculated features:
- **Momentum & Volatility Indicators**: MA5, MA10, MA20, MA50, RSI, MACD, ATR
- **Lagged Prices & Bollinger Bands**: Previous close prices, Upper/Lower Bands, SD20
- **Market Index Correlations**: QQQ_Close, SnP_Close, DJIA_Close, EMA20

## Methodology
1. **Data Cleaning**: Remove missing values.
2. **Feature Grouping**:
  - Set 1: Momentum & Volatility indicators
  - Set 2: Lagged prices & Bollinger Bands
  - Set 3: Market index correlations & trends
3. **Train-Test Split**: 80% training, 20% testing.
4. **Model Training**:
  - XGBoost (regression)
  - ANN (5 hidden neurons)
  - Random Forest (5-fold cross-validation)
  - SVR (default parameters)
5. **Evaluation Metrics**: R², MSE, RMSE, MAE, MAPE

## Results
| Model                 | R²    | MSE        | RMSE   | MAE    | MAPE    |
| --------------------- | ----- | ---------- | ------ | ------ | ------- |
| XGBoost (Set 1)       | 1.000 | 236.86     | 15.39  | 8.19   | 0.96%   |
| XGBoost (Set 2)       | 0.998 | 864.59     | 29.40  | 15.88  | 1.81%   |
| XGBoost (Set 3)       | 0.999 | 385.51     | 19.63  | 10.82  | 1.30%   |
| ANN (Set 1)           | NA    | 510,829.15 | 714.72 | 584.30 | 116.39% |
| ANN (Set 2)           | NA    | 510,829.14 | 714.72 | 584.30 | 116.39% |
| ANN (Set 3)           | NA    | 510,829.14 | 714.72 | 584.30 | 116.39% |
| Random Forest (Set 1) | 0.999 | 311.48     | 17.65  | 9.14   | 1.05%   |
| Random Forest (Set 2) | 0.999 | 692.22     | 26.31  | 13.90  | 1.62%   |
| Random Forest (Set 3) | 0.999 | 339.26     | 18.42  | 9.37   | 1.13%   |
| SVR (Set 1)           | 0.998 | 1241.98    | 35.24  | 29.15  | 6.91%   |
| SVR (Set 2)           | 0.997 | 1480.74    | 38.48  | 29.67  | 6.46%   |
| SVR (Set 3)           | 0.997 | 1460.97    | 38.22  | 31.00  | 6.46%   |

## Key Insights
- **Tree-based models (XGBoost and Random Forest) were the most accurate**, predicting Amazon’s stock price with extremely high precision (over 99% R² and very low error rates).

- **Momentum and volatility indicators (Set 1) gave the best overall performance across models**, suggesting these features are highly predictive.

- **Neural Networks performed poorly in this setup**, likely due to scaling and parameter tuning issues, making them unsuitable without further optimization.

- **Support Vector Regression was less accurate than tree-based methods** but still achieved reasonable performance for forecasting.
