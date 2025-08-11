##========================STOCK PREDICITON OF AMZN Stocks========================
# Name    : Aiden Awal Maulana
# Dataset : AMZN.csv


###========================= Preparation =========================
# Optional: Set working directory
setwd("C:/") 
#set your working directory here (where the dataset can be found)

# Load required packages
packages <- c("readr", "dplyr", "Metrics", "xgboost", "caret", "nnet","caret", "e1071", "randomForest")
lapply(packages, library, character.only = TRUE)

# Load and clean data
data <- read.csv("AMZN.csv") # make sure the title of the data is "AMZN.csv"
data <- na.omit(data)

## Set target and predictor sets
target <- "Close.t."

#Technical Indicators (Momentum + Volatility-based)
predictors_1 <- c("MA5", "MA10", "MA20", "MA50", "RSI", "MACD", "ATR")

#Lagged Prices and Bollinger Bands (Momentum + Mean Reversion)
predictors_2 <- c("S_Close.t.1.", "S_Close.t.2.", "S_Close.t.3.", "S_Close.t.5.",
                  "Upper_Band", "Lower_Band", "SD20")

#Market Index Correlations (QQQ, S&P, DJIA + Trend)
predictors_3 <- c("QQQ_Close", "SnP_Close", "DJIA_Close", "MA10", "EMA20", "MACD")

# Prepare matrices
x1 <- model.matrix(as.formula(paste("~", paste(predictors_1, collapse = "+"))), data)[, -1]
x2 <- model.matrix(as.formula(paste("~", paste(predictors_2, collapse = "+"))), data)[, -1]
x3 <- model.matrix(as.formula(paste("~", paste(predictors_3, collapse = "+"))), data)[, -1]
y <- data[[target]]

# Split into train/test
set.seed(42)
train_idx <- sample(1:nrow(data), 0.8 * nrow(data))
y_train <- y[train_idx]
y_test  <- y[-train_idx]

x1_train <- x1[train_idx, ]; x1_test <- x1[-train_idx, ]
x2_train <- x2[train_idx, ]; x2_test <- x2[-train_idx, ]
x3_train <- x3[train_idx, ]; x3_test <- x3[-train_idx, ]

# Evaluation function
evaluate <- function(name, actuals, preds) {
  safe_actuals <- ifelse(actuals == 0, 0.01, actuals)
  data.frame(
    Model = name,
    R2 = round(cor(actuals, preds)^2, 3),
    MSE = round(mse(actuals, preds), 2),
    RMSE = round(rmse(actuals, preds), 2),
    MAE = round(mae(actuals, preds), 2),
    MAPE = round(mape(safe_actuals, preds) * 100, 2)
  )
}

###========================= XGBoost =========================
xgb1 <- xgboost(data = x1_train, label = y_train, nrounds = 100,
                objective = "reg:squarederror", verbose = 0)
xgb2 <- xgboost(data = x2_train, label = y_train, nrounds = 100,
                objective = "reg:squarederror", verbose = 0)
xgb3 <- xgboost(data = x3_train, label = y_train, nrounds = 100,
                objective = "reg:squarederror", verbose = 0)

pred_xgb1 <- predict(xgb1, x1_test)
pred_xgb2 <- predict(xgb2, x2_test)
pred_xgb3 <- predict(xgb3, x3_test)

eval_xgb1 <- evaluate("XGBoost (Set 1)", y_test, pred_xgb1)
eval_xgb2 <- evaluate("XGBoost (Set 2)", y_test, pred_xgb2)
eval_xgb3 <- evaluate("XGBoost (Set 3)", y_test, pred_xgb3)

###======================= Neural Network =======================
# Normalize target for nnet
y_train_scaled <- scale(y_train)
y_test_scaled <- scale(y_test, center = attr(y_train_scaled, "scaled:center"), 
                       scale = attr(y_train_scaled, "scaled:scale"))

# Fit ANN models (size = 5 hidden neurons)
ann1 <- nnet(x1_train, y_train_scaled, size = 5, linout = TRUE, maxit = 500, trace = FALSE)
ann2 <- nnet(x2_train, y_train_scaled, size = 5, linout = TRUE, maxit = 500, trace = FALSE)
ann3 <- nnet(x3_train, y_train_scaled, size = 5, linout = TRUE, maxit = 500, trace = FALSE)

# Predict and reverse scale
pred_ann1 <- predict(ann1, x1_test) * attr(y_train_scaled, "scaled:scale") + attr(y_train_scaled, "scaled:center")
pred_ann2 <- predict(ann2, x2_test) * attr(y_train_scaled, "scaled:scale") + attr(y_train_scaled, "scaled:center")
pred_ann3 <- predict(ann3, x3_test) * attr(y_train_scaled, "scaled:scale") + attr(y_train_scaled, "scaled:center")

eval_ann1 <- evaluate("ANN (Set 1)", y_test, pred_ann1)
eval_ann2 <- evaluate("ANN (Set 2)", y_test, pred_ann2)
eval_ann3 <- evaluate("ANN (Set 3)", y_test, pred_ann3)

#==================== RANDOM FOREST ====================
ctrl <- trainControl(method = "cv", number = 5)

rf1 <- train(x = x1_train, y = y_train, method = "rf", trControl = ctrl)
rf2 <- train(x = x2_train, y = y_train, method = "rf", trControl = ctrl)
rf3 <- train(x = x3_train, y = y_train, method = "rf", trControl = ctrl)

eval_rf1 <- evaluate("Random Forest (Set 1)", y_test, predict(rf1, x1_test))
eval_rf2 <- evaluate("Random Forest (Set 2)", y_test, predict(rf2, x2_test))
eval_rf3 <- evaluate("Random Forest (Set 3)", y_test, predict(rf3, x3_test))

#==================== SVR ====================
svr1 <- svm(x1_train, y_train)
svr2 <- svm(x2_train, y_train)
svr3 <- svm(x3_train, y_train)

eval_svr1 <- evaluate("SVR (Set 1)", y_test, predict(svr1, x1_test))
eval_svr2 <- evaluate("SVR (Set 2)", y_test, predict(svr2, x2_test))
eval_svr3 <- evaluate("SVR (Set 3)", y_test, predict(svr3, x3_test))

#==================== RESULTS ====================
all_results <- rbind(
  eval_xgb1, eval_xgb2, eval_xgb3,
  eval_ann1, eval_ann2, eval_ann3,
  eval_rf1, eval_rf2, eval_rf3,
  eval_svr1, eval_svr2, eval_svr3
)

print(all_results)