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
cokm_nugget <- 1e-6              # fixed cokriging nugget (absolute; outputs have sd ~ 4)

### Forrester functions (x in [0, 1]) ###
# High fidelity (Forrester, Sobester & Keane, 2007)
forrester <- function(xx) {
  x <- xx[1]
  (6*x - 2)^2 * sin(12*x - 4)
}

# 2-fidelity low fidelity (Forrester et al., 2007): A = 0.5, B = 10, C = -5
forresterl <- function(xx) {
  x <- xx[1]
  0.5*forrester(x) + 10*(x - 0.5) + 5
}

# 4-fidelity hierarchy (Mainini et al., 2022, "Analytical benchmark problems for
# multifidelity optimization methods"): f1 = high, f4 = lowest
forrester_f2 <- function(xx) {
  x <- xx[1]
  (5.5*x - 2.5)^2 * sin(12*x - 4)
}
forrester_f3 <- function(xx) {
  x <- xx[1]
  0.75*forrester(x) + 5*(x - 0.5) - 2
}
forrester_f4 <- function(xx) {
  x <- xx[1]
  0.5*forrester(x) + 10*(x - 0.5) - 5
}

### Nested designs for any number of levels ###
# RNAmf::NestedX only supports 2 or 3 levels. For L > 3, apply the same algorithm
# (independent maximin LHS per level, then RNAmf's subsetX from the top level down,
# which swaps the closest lower-level points for the higher-level ones).
NestedX_L <- function(n, d) {
  L <- length(n)
  if (L <= 3) return(NestedX(n, d))
  if (is.unsorted(rev(n), strictly = TRUE)) stop("The number of design at each level must be descending order \n")
  X <- lapply(n, function(ni) maximinLHS(ni, d))
  for (l in (L - 1):1) {
    X[[l]] <- RNAmf:::subsetX(matrix(X[[l]], ncol = d), matrix(X[[l + 1]], ncol = d))$X
  }
  X
}

### Configurations ###
# Level order is lowest -> highest fidelity, as in X1, ..., XL.
# RNAmf supports at most 3 fidelity levels: with L > 3 only recursive co-kriging is fitted
# and no rna_* files are written.
configs <- list(
  "2fid" = list(n = c(12, 6),          funs = list(forresterl, forrester)),
  "4fid" = list(n = c(20, 15, 10, 6),  funs = list(forrester_f4, forrester_f3, forrester_f2, forrester))
)

# Optional command-line override: Rscript "forrester rnamf-cokr.R" <n.rep> [config ...]
args <- commandArgs(trailingOnly = TRUE)
n.rep <- if (length(args) >= 1) as.integer(args[1]) else 100
run_configs <- if (length(args) >= 2) args[-1] else names(configs)

d <- 1
n.test <- 1000
data_root <- "C:/Users/Santiago/OneDrive/Escritorio/Research/Orthogonal Project/Code/Python/Data"

run_forrester <- function(cfg) {
  n <- cfg$n
  L <- length(n)
  use_rna <- L <= 3

  dir_name <- file.path(data_root, glue("forrester/{paste(n, collapse = '_')}"))
  if (!dir.exists(dir_name)) dir.create(dir_name, recursive = TRUE)

  cat("Forrester function: n =", paste(n, collapse = ", "), ", rep =", n.rep,
      ifelse(use_rna, "(RNAmf + Cokriging)", "(Cokriging only: RNAmf supports <= 3 levels)"), "\n")

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
    if (use_rna) {
      npySave(file.path(dir_name, "rna_mean.npy"), rna_mean)
      npySave(file.path(dir_name, "rna_sigma2.npy"), rna_sigma2)
      npySave(file.path(dir_name, "rna_lower.npy"), rna_lower)
      npySave(file.path(dir_name, "rna_upper.npy"), rna_upper)
      npySave(file.path(dir_name, "rna_time.npy"), rna_time)
    }
    npySave(file.path(dir_name, "cokm_mean.npy"), cokm_mean)
    npySave(file.path(dir_name, "cokm_sigma2.npy"), cokm_sigma2)
    npySave(file.path(dir_name, "cokm_lower.npy"), cokm_lower)
    npySave(file.path(dir_name, "cokm_upper.npy"), cokm_upper)
    npySave(file.path(dir_name, "cokm_time.npy"), cokm_time)
  }

  for (i in 1:n.rep) {
    set.seed(i)

    Xs <- NestedX_L(n, d)
    ys <- lapply(1:L, function(l) apply(Xs[[l]], 1, cfg$funs[[l]]))

    for (l in 1:L) {
      X_rows[[l]][[i]] <- cbind(i, Xs[[l]])
      y_rows[[l]][[i]] <- cbind(i, ys[[l]])
    }
    nL <- length(ys[[L]])

    ### RNAmf (only for L <= 3) ###
    if (use_rna) {
      tic.RNAmf <- proc.time()[3]
      if (L == 2) {
        fit.RNAmf <- RNAmf_two_level(Xs[[1]], ys[[1]], Xs[[2]], ys[[2]], kernel = "sqex", constant = TRUE)
      } else {
        fit.RNAmf <- RNAmf_three_level(Xs[[1]], ys[[1]], Xs[[2]], ys[[2]], Xs[[3]], ys[[3]],
                                       kernel = "sqex", constant = TRUE)
      }
      pred.RNAmf <- predict(fit.RNAmf, x)
      predy <- pred.RNAmf$mu
      predsig2 <- pred.RNAmf$sig2
      toc.RNAmf <- proc.time()[3]

      # RNAmf 95% Credible Interval
      lower.RNAmf <- predy - 1.96 * sqrt(predsig2 * nL / (nL - 2))
      upper.RNAmf <- predy + 1.96 * sqrt(predsig2 * nL / (nL - 2))

      rna_mean[i, ] <- predy
      rna_sigma2[i, ] <- predsig2
      rna_lower[i, ] <- lower.RNAmf
      rna_upper[i, ] <- upper.RNAmf
      rna_time[i] <- toc.RNAmf - tic.RNAmf
      result.rmse[i, 1] <- sqrt(mean((predy - y)^2))
    }

    ### Recursive Cokriging ###
    tic.cokm <- proc.time()[3]
    NestDesign <- NestedDesignBuild(design = Xs)
    # In 1D the gauss correlation matrix becomes numerically singular for long ranges
    # (km's initial ranges are drawn up to 2x the design range), so a tiny fixed nugget
    # (~1e-7 relative to var(y)) is added for Cholesky stability.
    fit.muficokm <- MuFicokm(formula = rep(list(~1), L), MuFidesign = NestDesign, covtype = "gauss",
                             lower = eps, nugget = cokm_nugget,
                             response = ys, nlevel = L)
    pred.muficokm <- predict(fit.muficokm, x, "UK")
    toc.cokm <- proc.time()[3]

    pred_cokm_mean <- pred.muficokm$mean
    pred_cokm_sig2 <- pred.muficokm$sig2

    # Cokriging 95% Confidence Interval
    cokm_mean[i, ] <- pred_cokm_mean
    cokm_sigma2[i, ] <- pred_cokm_sig2
    cokm_lower[i, ] <- pred_cokm_mean - 1.96 * sqrt(pred_cokm_sig2)
    cokm_upper[i, ] <- pred_cokm_mean + 1.96 * sqrt(pred_cokm_sig2)
    cokm_time[i] <- toc.cokm - tic.cokm
    result.rmse[i, 2] <- sqrt(mean((pred_cokm_mean - y)^2))

    # Write the stacked files every 10 replicates (checkpoint) and at the end
    if (i %% 10 == 0 || i == n.rep) save_stacked()
  }

  print(colMeans(result.rmse))
  boxplot(result.rmse[, , drop = FALSE], main = glue("Forrester {paste(n, collapse = '_')}"))
  invisible(result.rmse)
}

for (cfg_name in run_configs) run_forrester(configs[[cfg_name]])
