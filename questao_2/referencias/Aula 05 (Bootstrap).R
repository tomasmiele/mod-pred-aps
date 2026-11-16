# Modelagem Preditiva - 2026-2 - Aula 5 (Bootstrap)

# rm(list = ls())

set.seed(42)

alpha <- 2
beta <- 3

f <- function(x) dgamma(x, shape = alpha, rate = beta)
plot(f, xlim = c(0, 3), lwd = 2, col = "blue")

n <- 500
x <- rgamma(n, shape = alpha, rate = beta)

plot(ecdf(x), main = "Função de distribuição empírica",
     xlab = "", ylab = "", cex = 0.5, xlim = c(0, 2.5))
Fd <- function(t) pgamma(t, shape = alpha, rate = beta)
curve(Fd, col = "red", lwd = 2, add = TRUE)

true_median <- qgamma(0.5, shape = alpha, rate = beta)

bootstrap_CI <- function(x, gamma = 0.95, B = 10^3) {
    xs <- matrix(sample(x, size = B*n, replace = TRUE), ncol = n)
    theta_hat <- apply(xs, 1, median)
    quantile(theta_hat, probs = c((1 - gamma) / 2, (1 + gamma) / 2))
}

bootstrap_CI(x, gamma = 0.9, B = 10^4)

# verificando a cobertura

N <- 10^3
X <- matrix(rgamma(N*n, shape = alpha, rate = beta), nrow = N)
CI <- t(apply(X, 1, function(.x) bootstrap_CI(.x, gamma = 0.9)))
mean(CI[, 1] <= true_median & true_median <= CI[, 2])
