# 0.dati 
pkgs <- c("pROC", "glmnet")
for (p in pkgs) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(pROC)
library(glmnet)

set.seed(123)
dir.create("output", showWarnings = FALSE)

# 1. carico dati
df <- read.table(file.choose(), header = FALSE, stringsAsFactors = TRUE)

names(df) <- c("checking_status", "duration_months", "credit_history",
               "purpose", "credit_amount", "savings", "employment_since",
               "installment_rate", "personal_status_sex", "other_debtors",
               "residence_since", "property", "age", "other_installments",
               "housing", "existing_credits", "job", "num_dependents",
               "telephone", "foreign_worker", "class")

# Target: 1 = default (bad), 0 = non-default (good)
df$default <- ifelse(df$class == 2, 1, 0)
df$class <- NULL

cat("Observations:", nrow(df), "\n")
cat("Default rate:", round(mean(df$default), 3), "\n")

#pulizia dati
collapse_rare <- function(x, min_n = 30) {
  if (!is.factor(x)) return(x)
  tab <- table(x)
  rare <- names(tab)[tab < min_n]
  if (length(rare) > 0) {
    x <- as.character(x)
    x[x %in% rare] <- "Other"
    x <- factor(x)
  }
  x
}
df[] <- lapply(df, collapse_rare)

# train 70% split 30%
idx_bad  <- which(df$default == 1)
idx_good <- which(df$default == 0)
train_idx <- c(sample(idx_bad,  round(0.7 * length(idx_bad))),
               sample(idx_good, round(0.7 * length(idx_good))))
train <- df[train_idx, ]
test  <- df[-train_idx, ]

# 4. regressione logistica 
logit <- glm(default ~ ., data = train, family = binomial)
print(summary(logit))

pd_test <- predict(logit, newdata = test, type = "response")

#5. Discriminatory power
roc_obj <- roc(test$default, pd_test, quiet = TRUE)
auc_val <- as.numeric(auc(roc_obj))
gini    <- 2 * auc_val - 1
ks_val  <- as.numeric(ks.test(pd_test[test$default == 1],
                              pd_test[test$default == 0])$statistic)

cat("\n--- Logistic regression: test set ---\n")
cat("AUC :", round(auc_val, 3), "\n")
cat("Gini:", round(gini, 3), "\n")
cat("KS  :", round(ks_val, 3), "\n")

png("output/roc_curve.png", width = 700, height = 600)
plot(roc_obj, main = paste0("ROC curve - AUC = ", round(auc_val, 3)))
dev.off()

#6.matice confusione
conf_matrix <- function(p, y, cutoff) {
  pred <- factor(ifelse(p >= cutoff, 1, 0), levels = c(0, 1))
  actual <- factor(y, levels = c(0, 1))
  tab <- table(Predicted = pred, Actual = actual)
  cat("\nCut-off =", cutoff, "\n")
  print(tab)
  cat("Accuracy   :", round(sum(diag(tab)) / sum(tab), 3), "\n")
  cat("Sensitivity:", round(tab[2, 2] / sum(tab[, 2]), 3),
      "(defaulters correctly flagged)\n")
  cat("Specificity:", round(tab[1, 1] / sum(tab[, 1]), 3), "\n")
}
conf_matrix(pd_test, test$default, 0.50)
conf_matrix(pd_test, test$default, 0.30)

# 7. Calibrazione: ppd predtto vs default osservato
bins <- cut(pd_test, breaks = quantile(pd_test, probs = seq(0, 1, 0.2)),
            include.lowest = TRUE, labels = paste("Class", 1:5))
calib <- data.frame(
  class       = levels(bins),
  n           = as.vector(table(bins)),
  mean_pred   = as.vector(tapply(pd_test, bins, mean)),
  observed_dr = as.vector(tapply(test$default, bins, mean))
)
print(calib)

png("output/calibration.png", width = 700, height = 600)
plot(calib$mean_pred, calib$observed_dr, pch = 19, col = "blue",
     xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Mean predicted PD", ylab = "Observed default rate",
     main = "Calibration by rating class")
abline(0, 1, lty = 2)
dev.off()

#8.regressione logistica lasso 
X_all <- model.matrix(default ~ ., df)[, -1]
y_all <- df$default
X_train <- X_all[train_idx, ]
X_test  <- X_all[-train_idx, ]

cv_lasso <- cv.glmnet(X_train, y_all[train_idx], family = "binomial",
                      alpha = 1, type.measure = "auc")
pd_lasso <- as.numeric(predict(cv_lasso, newx = X_test,
                               s = "lambda.1se", type = "response"))
auc_lasso <- as.numeric(auc(roc(test$default, pd_lasso, quiet = TRUE)))

cat("\n--- Lasso: test set ---\n")
cat("AUC :", round(auc_lasso, 3), "\n")
cat("Gini:", round(2 * auc_lasso - 1, 3), "\n")
cat("Non-zero coefficients:",
    sum(coef(cv_lasso, s = "lambda.1se")[-1] != 0),
    "of", ncol(X_train), "\n")

# 9. Da PD a Expected Loss 
# EL = PD x LGD x EAD
# ASSUMPTIONS (illustrative only): LGD = 45%, EAD = credit_amount
LGD <- 0.45
ead <- test$credit_amount
el  <- pd_test * LGD * ead

cat("\n--- Expected Loss on test portfolio ---\n")
cat("Total EAD          :", round(sum(ead)), "\n")
cat("Expected Loss (EL) :", round(sum(el)), "\n")
cat("EL / EAD           :", round(sum(el) / sum(ead) * 100, 2), "%\n")

# 10. simulazione montecarlo con perdite portafoglio
# assumiamo il defaukt come indipendente 
n_sim <- 10000
set.seed(2024)
loss_sim <- numeric(n_sim)
for (i in 1:n_sim) {
  defaults <- rbinom(length(pd_test), size = 1, prob = pd_test)
  loss_sim[i] <- sum(defaults * LGD * ead)
}
var99 <- as.numeric(quantile(loss_sim, 0.99))
es99  <- mean(loss_sim[loss_sim >= var99])

cat("\n--- Monte Carlo (", n_sim, " simulations) ---\n", sep = "")
cat("Mean simulated loss          :", round(mean(loss_sim)), "\n")
cat("99% Value-at-Risk of loss    :", round(var99), "\n")
cat("99% Expected Shortfall       :", round(es99), "\n")
cat("Unexpected loss (VaR - mean) :", round(var99 - mean(loss_sim)), "\n")

png("output/monte_carlo_loss.png", width = 700, height = 600)
hist(loss_sim, breaks = 50, col = "lightgrey",
     main = "Simulated portfolio loss distribution",
     xlab = "Portfolio loss")
abline(v = mean(loss_sim), col = "blue", lwd = 2)
abline(v = var99, col = "red", lwd = 2)
legend("topright", legend = c("Mean (EL)", "99% VaR"),
       col = c("blue", "red"), lwd = 2)
dev.off()

#11.summary 
res <- data.frame(
  metric = c("AUC logit", "Gini logit", "KS logit", "AUC lasso", "Gini lasso"),
  value  = round(c(auc_val, gini, ks_val, auc_lasso, 2 * auc_lasso - 1), 3)
)
write.csv(res, "output/results_summary.csv", row.names = FALSE)
write.csv(calib, "output/calibration_table.csv", row.names = FALSE)
cat("\nDone. Plots and tables saved in the 'output' folder.\n")

