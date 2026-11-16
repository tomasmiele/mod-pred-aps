# Modelagem Preditiva - 2026-2 - Aula 5 (Bagging)

# rm(list = ls())

library(tree)
library(ranger)
library(pROC)
library(class)

### Classification: Spam data

spam <- read.csv("spam.csv", stringsAsFactors = TRUE)

# str(spam)

table(spam$Class) / nrow(spam)

set.seed(42)

idx <- sample(1:nrow(spam), size = round(0.7 * nrow(spam)), replace = FALSE)

trn <- spam[idx, ]
tst <- spam[-idx, ]

# Classification Tree

ctree <- tree(Class ~ ., data = trn)
prob_ctree <- predict(ctree, tst)[, 2]
y_hat_ctree <- ifelse(prob_ctree >= 0.5, "Spam", "Ham")
mean(y_hat_ctree != tst$Class)
roc_ctree <- roc(tst$Class, prob_ctree)
auc(roc_ctree)

# Bagging of trees

bag <- ranger(Class ~ ., data = trn, mtry = 57, probability = TRUE)
prob_bag <- predict(bag, data = tst)$predictions[, 2]
y_hat_bag <- ifelse(prob_bag >= 0.5, "Spam", "Ham")
mean(y_hat_bag != tst$Class)
roc_bag <- roc(tst$Class, prob_bag)
auc(roc_bag)

plot(roc_ctree, col = "dark green", grid = TRUE,
     xlab = "FPR (1 -  Specificity)",
     ylab = "TPR (Sensitivity)",
     main = "ROC Spam", legacy.axes = TRUE, asp = FALSE, las = 1)

plot(roc_bag, col = "blue", add = TRUE)

legend("bottomright", legend = c("Tree", "Bagging"),
       col = c("dark green", "blue"), lwd = 2, cex = 0.95)

# 1-NN

X_trn <- trn[, 1:57]
X_tst <- tst[, 1:57]
y_hat_1_NN <- knn(X_trn, X_tst, trn$Class, k = 1)
mean(y_hat_1_NN != tst$Class)

# 1-NN after scaling the data

X <- scale(as.matrix(spam[, 1:57]), center = TRUE, scale = TRUE)
X_trn <- X[idx, 1:57]
X_tst <- X[-idx, 1:57]
y_hat_1_NN_scl <- knn(X_trn, X_tst, trn$Class, k = 1)
mean(y_hat_1_NN_scl != tst$Class)

# Bagging of 1-NN classifiers

B <- 100
Y_hat <- matrix(nrow = nrow(X_tst), ncol = B) 
pb <- txtProgressBar(min = 1, max = B, style = 3)
for (i in 1:B) {
    boot <- sample(1:nrow(X_trn), size = nrow(X_trn), replace = TRUE)
	   xs <- sample(1:57, size = 25, replace = FALSE) # random subspace
	   Y_hat[, i] <- as.character(knn(X_trn[boot, xs], X_tst[, xs], trn$Class[boot], k = 1))
	   setTxtProgressBar(pb, i)
}
close(pb)

get_mode <- function(y) names(sort(table(y), decreasing = TRUE))[1]

y_hat_bagging_1_NN <- apply(Y_hat, 1, get_mode) # majority vote

mean(y_hat_bagging_1_NN != tst$Class)
