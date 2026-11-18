rm(list = ls())

if (!requireNamespace("RNAmf", quietly = TRUE)) {
  install.packages("C:/Users/Santiago/OneDrive/Escritorio/Research/Orthogonal Project/Code/R/RNAmf_0.1.2.tar.gz", repos = NULL, type = "source")
}
library(RNAmf)
library(DiceKriging)
library(MuFiCokriging)
library(RcppCNPy)
library(glue)
library(lhs)

eps <- sqrt(.Machine$double.eps) # small nugget for numeric stability

### Hartmann functions (x in [0, 1]^d) ###
# f(x) = - sum_i alpha_i exp(- sum_j A_ij (x_j - P_ij)^2)
# Multi-fidelity hierarchy (Kandasamy et al., 2016, MF-GP-UCB):
#   alpha^(m) = alpha + (M - m) * delta,  delta = (0.01, -0.01, -0.1, 0.1),  m = 1 (lowest), ..., M (high)
hart_alpha <- c(1.0, 1.2, 3.0, 3.2)
hart_delta <- c(0.01, -0.01, -0.1, 0.1)

hart3_A <- matrix(c(3.0, 10, 30,
                    0.1, 10, 35,
                    3.0, 10, 30,
                    0.1, 10, 35), 4, 3, byrow = TRUE)
hart3_P <- 1e-4 * matrix(c(3689, 1170, 2673,
                           4699, 4387, 7470,
                           1091, 8732, 5547,
                            381, 5743, 8828), 4, 3, byrow = TRUE)

hart6_A <- matrix(c(10,   3,   17,   3.5, 1.7,  8,
                    0.05, 10,  17,   0.1, 8,   14,
                    3,    3.5, 1.7,  10,  17,   8,
                    17,   8,   0.05, 10,  0.1, 14), 4, 6, byrow = TRUE)
hart6_P <- 1e-4 * matrix(c(1312, 1696, 5569,  124, 8283, 5886,
                           2329, 4135, 8307, 3736, 1004, 9991,
                           2348, 1451, 3522, 2883, 3047, 6650,
                           4047, 8828, 8732, 5743, 1091,  381), 4, 6, byrow = TRUE)

hartmann <- function(xx, A, P, alpha) {
  inner <- rowSums(A * (matrix(xx, nrow(A), ncol(A), byrow = TRUE) - P)^2)
  -sum(alpha * exp(-inner))
}

# Fidelity m of M (m = M is the high fidelity)
hartmann_fid <- function(A, P, m, M) {
  alpha_m <- hart_alpha + (M - m) * hart_delta
  function(xx) hartmann(xx, A, P, alpha_m)
}

### Configurations ###
# Level order is lowest -> highest fidelity, as in X1, X2, X3 (all 3-level: RNAmf + Cokriging).
configs <- list(
  "hartmann3" = list(d = 3, n = c(60, 30, 15),
                     funs = lapply(1:3, function(m) hartmann_fid(hart3_A, hart3_P, m, 3))),
  "hartmann6" = list(d = 6, n = c(120, 60, 30),
                     funs = lapply(1:3, function(m) hartmann_fid(hart6_A, hart6_P, m, 3)))
)

# Optional command-line override: Rscript "hartmann rnamf-cokr.R" <n.rep> [config ...]
args <- commandArgs(trailingOnly = TRUE)
n.rep <- if (length(args) >= 1) as.integer(args[1]) else 100
run_configs <- if (length(args) >= 2) args[-1] else names(configs)

n.test <- 1000
data_root <- "C:/Users/Santiago/OneDrive/Escritorio/Research/Orthogonal Project/Code/Python/Data"

run_hartmann <- function(cfg_name) {
  cfg <- configs[[cfg_name]]
  n <- cfg$n; d <- cfg$d
  L <- length(n)
  stopifnot(L <= 3)  # RNAmf supports at most 3 fidelity levels

  dir_name <- file.path(data_root, glue("{cfg_name}/{paste(n, collapse = '_')}"))
  if (!dir.exists(dir_name)) dir.create(dir_name, recursive = TRUE)

  cat("Hartmann", d, "D function: n =", paste(n, collapse = ", "), ", rep =", n.rep, "\n")

  result.rmse <- matrix(NA, n.rep, 2)
  colnames(result.rmse) <- c("RNAmf", "Cokriging")

  ### test data ###
  set.seed(0)
  x <- maximinLHS(n.test, d)
  y <- apply(x, 1, cfg$funs[[L]])

  npySave(glue("{dir_name}/x_test.npy"), x)
  npySave(glue("{dir_name}/yhigh_test.npy"), y)

  # --- Containers for saving: one .npy file per quantity, all replicates stacked ---
  X_rows <- replicate(L, list(), simplify = FALSE)
  y_rows <- replicate(L, list(), simplify = FALSE)
  rna_mean <- rna_sigma2 <- rna_lower <- rna_upper <- matrix(NA_real_, n.rep, n.test)
  cokm_mean <- cokm_sigma2 <- cokm_lower <- cokm_upper <- matrix(NA_real_, n.rep, n.test)
  rna_time <- cokm_time <- rep(NA_real_, n.rep)

  save_stacked <- function() {
    for (l in seq_along(X_rows)) {
      npySave(file.path(dir_name, glue("X{l}.npy")), do.call(rbind, X_rows[[l]]))
      npySave(file.path(dir_name, glue("y{l}.npy")), do.call(rbind, y_rows[[l]]))
    }
    npySave(file.path(dir_name, "rna_mean.npy"), rna_mean)
    npySave(file.path(dir_name, "rna_sigma2.npy"), rna_sigma2)
    npySave(file.path(dir_name, "rna_lower.npy"), rna_lower)
    npySave(file.path(dir_name, "rna_upper.npy"), rna_upper)
    npySave(file.path(dir_name, "rna_time.npy"), rna_time)
    npySave(file.path(dir_name, "cokm_mean.npy"), cokm_mean)
    npySave(file.path(dir_name, "cokm_sigma2.npy"), cokm_sigma2)
    npySave(file.path(dir_name, "cokm_lower.npy"), cokm_lower)
    npySave(file.path(dir_name, "cokm_upper.npy"), cokm_upper)
    npySave(file.path(dir_name, "cokm_time.npy"), cokm_time)
  }

  for (i in 1:n.rep) {
    set.seed(i)

    NX <- NestedX(n, d)

    X1 <- NX[[1]]
    X2 <- NX[[2]]
    X3 <- NX[[3]]

    X_rows[[1]][[i]] <- cbind(i, X1)
    X_rows[[2]][[i]] <- cbind(i, X2)
    X_rows[[3]][[i]] <- cbind(i, X3)

    y1 <- apply(X1, 1, cfg$funs[[1]])
    y2 <- apply(X2, 1, cfg$funs[[2]])
    y3 <- apply(X3, 1, cfg$funs[[3]])

    y_rows[[1]][[i]] <- cbind(i, y1)
    y_rows[[2]][[i]] <- cbind(i, y2)
    y_rows[[3]][[i]] <- cbind(i, y3)

    ### RNAmf ###
    tic.RNAmf <- proc.time()[3]
    fit.RNAmf <- RNAmf_three_level(X1, y1, X2, y2, X3, y3, kernel = "sqex", constant = TRUE)
    pred.RNAmf <- predict(fit.RNAmf, x)
    predy <- pred.RNAmf$mu
    predsig2 <- pred.RNAmf$sig2
    toc.RNAmf <- proc.time()[3]

    # RNAmf 95% Credible Interval
    lower.RNAmf <- predy - 1.96 * sqrt(predsig2 * length(y3) / (length(y3) - 2))
    upper.RNAmf <- predy + 1.96 * sqrt(predsig2 * length(y3) / (length(y3) - 2))

    ### Cokriging ###
    tic.cokm <- proc.time()[3]
    NestDesign <- NestedDesignBuild(design = list(X1, X2, X3))
    fit.muficokm <- MuFicokm(formula = list(~1, ~1, ~1), MuFidesign = NestDesign, covtype = "gauss",
                             lower = eps,
                             response = list(y1, y2, y3), nlevel = 3)
    pred.muficokm <- predict(fit.muficokm, x, "UK")
    toc.cokm <- proc.time()[3]

    pred_cokm_mean <- pred.muficokm$mean
    pred_cokm_sig2 <- pred.muficokm$sig2

    # Cokriging 95% Confidence Interval
    lower.cokm <- pred_cokm_mean - 1.96 * sqrt(pred_cokm_sig2)
    upper.cokm <- pred_cokm_mean + 1.96 * sqrt(pred_cokm_sig2)

    # Save individual replication intervals and predictions for Python
    rna_mean[i, ] <- predy
    rna_sigma2[i, ] <- predsig2
    rna_lower[i, ] <- lower.RNAmf
    rna_upper[i, ] <- upper.RNAmf

    cokm_mean[i, ] <- pred_cokm_mean
    cokm_sigma2[i, ] <- pred_cokm_sig2
    cokm_lower[i, ] <- lower.cokm
    cokm_upper[i, ] <- upper.cokm

    # Track metrics
    result.rmse[i, 1] <- sqrt(mean((predy - y)^2))
    result.rmse[i, 2] <- sqrt(mean((pred_cokm_mean - y)^2))

    # Save individual replication times for Python
    rna_time[i] <- toc.RNAmf - tic.RNAmf
    cokm_time[i] <- toc.cokm - tic.cokm

    # Write the stacked files every 10 replicates (checkpoint) and at the end
    if (i %% 10 == 0 || i == n.rep) save_stacked()
  }

  print(colMeans(result.rmse))
  boxplot(result.rmse[, , drop = FALSE], main = glue("Hartmann {d}D {paste(n, collapse = '_')}"))
  invisible(result.rmse)
}

for (cfg_name in run_configs) run_hartmann(cfg_name)
