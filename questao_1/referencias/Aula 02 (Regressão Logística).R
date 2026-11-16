# Modelagem Preditiva - 2025-2 - Aula 2 (Regressão Logística)

# rm(list = ls())

library(tidyverse)
library(rsample)
library(patchwork) # install.packages("patchwork")
library(ISLR)      # install.packages("ISLR")
library(pROC)      # install.packages("pROC")

# ?Default

str(Default)

# skimr::skim(Default)

theme_set(theme_bw())

p1 <- ggplot(Default, aes(x = balance, y = income, color = default, shape = default)) +
    geom_point(alpha = 0.6) +
    labs(x = "Balance", y = "Income")

p2 <- ggplot(Default, aes(x = default, y = balance, fill = default)) +
    geom_boxplot(alpha = 0.75, show.legend = FALSE) +
    labs(x = "Default",y = "Balance")

p3 <- ggplot(Default, aes(x = default, y = income, fill = default)) +
    geom_boxplot(alpha = 0.75, show.legend = FALSE) +
    labs(x = "Default", y = "Income")

p1 + p2 + p3

###

set.seed(42)

split <- initial_split(Default, prop = 0.5)

trn <- training(split)
tst <- testing(split)

# lreg <- glm(default ~ balance + income,
lreg <- glm(default ~ .,
            data = trn,
            family = binomial) # regressão logística

summary(lreg)

# ?predict.glm

prob <- predict(lreg, newdata = tst, type = "response")

threshold <- 0.5

y_hat <- ifelse(prob >= threshold, "Yes", "No")

table(Predicted = y_hat, Observed = tst$default)

###

roc_rlog <- roc(tst$default, prob)

names(roc_rlog)

with(roc_rlog, thresholds[which.max(sensitivities + specificities)])

par(mfrow = c(1, 1))

plot(roc_rlog, col = "blue", grid = TRUE,
     xlab = "FPR (1 -  Specificity)",
     ylab = "TPR (Sensitivity)",
     print.thres.pattern = " %.3f",
     # print.thres = threshold,
     print.thres = "best",
     main = "ROC", legacy.axes = TRUE, asp = FALSE, las = 1)

auc(roc_rlog)
