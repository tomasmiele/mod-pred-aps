# Modelagem Preditiva - 2026-2 - Aula 3 (Árvores de Classificação)

# rm(list = ls())

library(tidyverse)
library(rsample)
library(tree)
library(pROC)

spam <- read.csv("spam.csv", stringsAsFactors = TRUE)

str(spam)

spam |> count(Class)

set.seed(42)

split <- initial_split(spam, prop = 0.7)

trn <- training(split)
tst <- testing(split)

# CART

ctree <- tree(Class ~ . , data = trn)

# ctree <- tree(Class ~ . , data = trn, control = tree.control(nobs = nrow(training), mincut = 1, minsize = 2, mindev = 0))

summary(ctree)

plot(ctree, type = "uniform")
text(ctree, cex = 0.75)

cv <- cv.tree(ctree, FUN = prune.misclass)

plot(cv$size, cv$dev, type = "b", lwd = 2, col = "dark green",
	    xlab = "Número de folhas",
	    ylab = "Número de erros de classificação",
	    main = "Validação cruzada em 10 lotes")

pruned <- prune.misclass(ctree, best = 6)

plot(pruned, type = "uniform")
text(pruned, cex = 0.95)

y_hat_pruned <- predict(pruned, newdata = tst, type = "class")

mean(y_hat_pruned != tst$Class)

table(Predicted = y_hat_pruned, Observed = tst$Class)

prob_pruned <- predict(pruned, newdata = tst, type = "vector")[, 2]

roc_pruned <- roc(tst$Class, prob_pruned)

plot(roc_pruned, col = "blue", grid = TRUE,
     xlab = "FPR (1 -  Specificity)",
     ylab = "TPR (Sensitivity)",
     # print.thres.pattern = " %.3f"
     # print.thres = "best"
     # print.thres = 0.25
     main = "ROC", legacy.axes = TRUE, asp = FALSE, las = 1)

auc(tst$Class, prob_pruned)

# Regressão Logística

rlog <- glm(Class ~ ., data = trn, family = binomial)

summary(rlog)

prob_rlog <- predict(rlog, newdata = tst, type = "response")

roc_rlog <- roc(tst$Class, prob_rlog)

plot(roc_rlog, col = "red", add = TRUE) # , print.thres.pattern = " %.3f", print.thres = "best")

auc(tst$Class, prob_rlog)

legend("bottomright", legend = c("Tree", "RLog"), col = c("blue", "red"), lwd = 2, cex = 0.95)
