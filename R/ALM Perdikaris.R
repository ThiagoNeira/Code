rm(list = ls())

# Locate this script's folder (Code/R/) so that no path depends on the machine or the working
# directory. Works with Rscript, source() and RStudio; otherwise the working directory must be Code/R/.
r_dir <- local({
  f <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
  f <- gsub("~+~", " ", f, fixed = TRUE)   # Rscript encodes spaces in the file name as ~+~
  if (length(f) == 0) f <- tryCatch(sys.frame(1)$ofile, error = function(e) NULL)
  if (length(f) == 0 && requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable())
    f <- rstudioapi::getSourceEditorContext()$path
  d <- if (length(f) == 1 && nzchar(f)) dirname(normalizePath(f)) else getwd()
  if (!file.exists(file.path(d, "RNAmf_0.1.2.tar.gz")))
    stop("Cannot locate Code/R/. Run this script with Rscript or source(), or setwd() to Code/R/ first.")
  normalizePath(d, winslash = "/")
})
install.packages(file.path(r_dir, "RNAmf_0.1.2.tar.gz"), repos = NULL, type = "source")
library(RNAmf)
library(RcppCNPy)
library(glue)
library(lhs)

### Perdikaris Example ###
f1 <- function(x)
{
  sin(8*pi*x)
}

f2 <- function(x)
{
  (x-sqrt(2))*(sin(8*pi*x))^2
}

### training data ###
n1 <- 13; n2 <- 8; n.rep <- 10
d <- 1

### active learning settings ###
# Saved to alm_settings.npy; the Python comparison (alm_perdikaris.py) reads them from there.
cost <- c(1, 3)     # cost of one run at level 1 and level 2 (a nested level-2 run costs sum(cost))
budget <- 50        # total cost allowed for the acquisitions (the initial design is free)
n.screen <- 256     # space-filling screening points per iteration (only pick the L-BFGS-B starts)
n.starts <- 3       # L-BFGS-B starts per level
min.sep <- 0.05     # minimum distance between starts in [0,1]^d

# Create dynamic folder name based on n1 and n2
# Results are written straight into the Python data folder (no manual copying needed)
data_root <- file.path(dirname(r_dir), "Python", "Data")   # Code/Python/Data
dir_name <- file.path(data_root, glue("perdikaris/alm_{n1}_{n2}"))

# Create the directory if it doesn't already exist
if (!dir.exists(dir_name)) {
  dir.create(dir_name, recursive = TRUE)
}

### test data ###
n.test <- 1000

cat("Perdikaris function - ALM: n1 =", n1, ", n2 =", n2, ", rep =", n.rep, ", budget =", budget, "\n")

set.seed(0)
x <- seq(0,1,length.out=n.test)
y <- f2(matrix(x,ncol=1))

npySave(glue("{dir_name}/x_test.npy"), x)
npySave(glue("{dir_name}/yhigh_test.npy"), y)
npySave(glue("{dir_name}/alm_settings.npy"), c(cost, budget, n.screen, n.starts, min.sep))

### ALM acquisition ###
# Same search as Python's active_learning.alm_select: the RNAmf variance at each level is screened
# on a space-filling pool, L-BFGS-B runs from the n.starts best pool points that are at least
# min.sep apart, and the level with the largest  max variance / nested cost  is chosen among the
# levels whose nested cost still fits in the budget. The criterion is RNAmf's own (obj.ALM_level_l).
alm_select <- function(fit, cost, allowed) {
  pool <- randomLHS(n.screen, d)
  v.pool <- cbind(RNAmf:::pred.GP(fit$fit1, pool)$sig2,   # level-1 variance
                  predict(fit, pool)$sig2)                # level-2 variance
  obj <- list(RNAmf:::obj.ALM_level_1, RNAmf:::obj.ALM_level_2)
  crit <- rep(-Inf, 2)
  X.best <- matrix(NA_real_, 2, d)
  for (l in which(allowed)) {
    ord <- order(v.pool[, l], decreasing = TRUE)
    starts <- pool[ord[1], , drop = FALSE]
    for (j in ord[-1]) {
      if (nrow(starts) >= n.starts) break
      if (all(sqrt(colSums((t(starts) - pool[j, ])^2)) >= min.sep)) starts <- rbind(starts, pool[j, ])
    }
    best.x <- pool[ord[1], ]; best.v <- v.pool[ord[1], l]   # the screen is the fallback
    for (s in 1:nrow(starts)) {
      out <- tryCatch(optim(starts[s, ], obj[[l]], method = "L-BFGS-B", lower = 0, upper = 1,
                            control = list(maxit = 100), fit = fit),
                      error = function(e) NULL)
      if (!is.null(out) && is.finite(out$value) && -out$value > best.v) {
        best.x <- out$par; best.v <- -out$value
      }
    }
    X.best[l, ] <- best.x
    crit[l] <- best.v / cumsum(cost)[l]
  }
  level <- which.max(crit)
  list(level = level, Xnext = matrix(X.best[level, ], nrow = 1), crit = crit)
}

# TRUE if xnew is already a row of X (same tolerance as numpy.isclose in active_learning.py)
in.design <- function(X, xnew) {
  tol <- 1e-8 + 1e-5 * abs(drop(xnew))
  any(apply(abs(sweep(X, 2, drop(xnew))), 1, function(r) all(r <= tol)))
}

# --- Containers for saving: one .npy file per quantity, all replicates stacked ---
# X{l}.npy / y{l}.npy : initial design of every replicate, first column = replicate index i
# rna_alm_history.npy : one row per RNAmf fit: i, step, spent, N1, N2, level, x_1..x_d
#                       (level and x = acquisition chosen after that fit; NA on the last fit)
# rna_alm_mean.npy / rna_alm_sigma2.npy : test predictions, one row per history row
# rna_alm_time.npy : (n.rep,), entry i = replicate i
X_rows <- list(list(), list())
y_rows <- list(list(), list())
hist_rows <- list()
mean_rows <- list()
sig2_rows <- list()
rna_time <- rep(NA_real_, n.rep)
rmse_curves <- list()

save_stacked <- function() {
  for (l in seq_along(X_rows)) {
    npySave(file.path(dir_name, glue("X{l}.npy")), do.call(rbind, X_rows[[l]]))
    npySave(file.path(dir_name, glue("y{l}.npy")), do.call(rbind, y_rows[[l]]))
  }
  npySave(file.path(dir_name, "rna_alm_history.npy"), do.call(rbind, hist_rows))
  npySave(file.path(dir_name, "rna_alm_mean.npy"), do.call(rbind, mean_rows))
  npySave(file.path(dir_name, "rna_alm_sigma2.npy"), do.call(rbind, sig2_rows))
  npySave(file.path(dir_name, "rna_alm_time.npy"), rna_time)
}

for(i in 1:n.rep) {
  set.seed(i)
  cat("Perdikaris - ALM: ", i, "/", n.rep, "\n")

  NX <- NestedX(c(n1, n2), d)

  X1 = NX[[1]]
  X2 = NX[[2]]

  y1 <- f1(X1)
  y2 <- f2(X2)

  X_rows[[1]][[i]] <- cbind(i, X1)
  X_rows[[2]][[i]] <- cbind(i, X2)
  y_rows[[1]][[i]] <- cbind(i, y1)
  y_rows[[2]][[i]] <- cbind(i, y2)

  tic.ALM <- proc.time()[3]
  spent <- 0
  step <- 0
  rmse.path <- cost.path <- c()

  #################
  ### Add point ###
  #################
  repeat {
    ### fit an RNAmf ###
    fit.RNAmf <- RNAmf_two_level(X1, y1, X2, y2, kernel="sqex", constant=TRUE)
    ### predict ###
    pred.RNAmf <- predict(fit.RNAmf, x)
    predy <- drop(pred.RNAmf$mu)       # mu is an (n.test x 1) matrix; store one row per fit
    predsig2 <- drop(pred.RNAmf$sig2)

    hist.row <- c(i, step, spent, nrow(X1), nrow(X2), NA, rep(NA, d))
    mean_rows[[length(mean_rows) + 1]] <- predy
    sig2_rows[[length(sig2_rows) + 1]] <- predsig2

    ### RMSE (monitoring only; the comparison metrics are computed in Python) ###
    rmse.path <- c(rmse.path, sqrt(mean((predy-y)^2)))
    cost.path <- c(cost.path, spent)
    cat(sprintf("  step %2d | spent %5.1f | N %d, %d | RMSE %.4f\n", step, spent, nrow(X1), nrow(X2), rmse.path[step + 1]))

    ### next point: only levels whose nested cost still fits in the remaining budget ###
    allowed <- cumsum(cost) <= budget - spent
    if (!any(allowed)) {
      hist_rows[[length(hist_rows) + 1]] <- hist.row
      break
    }
    alm.RNAmf <- alm_select(fit.RNAmf, cost, allowed)
    Xnext <- alm.RNAmf$Xnext
    hist.row[6] <- alm.RNAmf$level
    hist.row[7:(6 + d)] <- Xnext
    hist_rows[[length(hist_rows) + 1]] <- hist.row

    ### nested acquisition: evaluate Xnext at levels 1..level (skip a level that already has it) ###
    added <- FALSE
    if (!in.design(X1, Xnext)) {
      X1 <- rbind(X1, Xnext)
      y1 <- c(y1, f1(Xnext))
      spent <- spent + cost[1]
      added <- TRUE
    }
    if (alm.RNAmf$level == 2 && !in.design(X2, Xnext)) {
      X2 <- rbind(X2, Xnext)
      y2 <- c(y2, f2(Xnext))
      spent <- spent + cost[2]
      added <- TRUE
    }
    if (!added) { # Xnext already in the design: refitting would repeat forever
      cat("  ALM stopped: selected x is already in the design at levels 1 ..", alm.RNAmf$level, "\n")
      break
    }
    step <- step + 1
  }

  rna_time[i] <- proc.time()[3] - tic.ALM
  rmse_curves[[i]] <- cbind(cost.path, rmse.path)

  # Write the stacked files every 10 replicates (checkpoint) and at the end
  if (i %% 10 == 0 || i == n.rep) save_stacked()

  plot(NULL, xlim = c(0, budget), ylim = range(unlist(lapply(rmse_curves, function(m) m[, 2]))),
       log = "y", xlab = "cost", ylab = "RMSE", main = "RNAmf ALM")
  for (m in rmse_curves) lines(m[, 1], m[, 2], type = "s")
}
