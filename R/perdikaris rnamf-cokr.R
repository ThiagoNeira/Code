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
library(DiceKriging)
library(MuFiCokriging)
library(RcppCNPy)
library(glue)
library(lhs)

eps <- sqrt(.Machine$double.eps) # small nugget for numeric stability

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
n1 <- 13; n2 <- 8; n.rep <- 100
d <- 1

# Create dynamic folder name based on n1 and n2
# Results are written straight into the Python data folder (no manual copying needed)
data_root <- file.path(dirname(r_dir), "Python", "Data")   # Code/Python/Data
dir_name <- file.path(data_root, glue("perdikaris/{n1}_{n2}"))

# Create the directory if it doesn't already exist
if (!dir.exists(dir_name)) {
  dir.create(dir_name, recursive = TRUE)
}

### test data ###
n.test <- 1000

result.perd.rmse <- matrix(NA, n.rep, 2)
colnames(result.perd.rmse) <- c("RNAmf", "Cokriging")

cat("Perdikaris function: n1 =", n1, ", n2 =", n2, ", rep =", n.rep, "\n")

### test data ###
set.seed(0)
x <- seq(0,1,length.out=n.test)
y <- f2(matrix(x,ncol=1))

npySave(glue("{dir_name}/x_test.npy"), x)
npySave(glue("{dir_name}/yhigh_test.npy"), y)

# --- Containers for saving: one .npy file per quantity, all replicates stacked ---
# X{l}.npy / y{l}.npy : rows of every replicate, first column = replicate index i
# {rna,cokm}_{mean,sigma2,lower,upper}.npy : (n.rep, n.test), row i = replicate i
# {rna,cokm}_time.npy : (n.rep,), entry i = replicate i
X_rows <- list(list(), list())
y_rows <- list(list(), list())
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

for(i in 1:n.rep) {
  set.seed(i)
  
  NX <- NestedX(c(n1, n2), d)
  
  X1 = NX[[1]]
  X2 = NX[[2]]
  
  X_rows[[1]][[i]] <- cbind(i, X1)
  X_rows[[2]][[i]] <- cbind(i, X2)
  
  y1 <- f1(X1)
  y2 <- f2(X2)
  
  y_rows[[1]][[i]] <- cbind(i, y1)
  y_rows[[2]][[i]] <- cbind(i, y2)
  
  ### fit an RNAmf ###
  tic.RNAmf <- proc.time()[3]
  fit.RNAmf <- RNAmf_two_level(X1, y1, X2, y2, kernel="sqex", constant=TRUE)
  ### predict ###
  pred.RNAmf <- predict(fit.RNAmf, x)
  predy <- pred.RNAmf$mu
  predsig2 <- pred.RNAmf$sig2
  toc.RNAmf <- proc.time()[3]
  
  # RNAmf 95% Credible Interval
  lower.RNAmf <- predy - 1.96 * sqrt(predsig2 * length(y2) / (length(y2) - 2))
  upper.RNAmf <- predy + 1.96 * sqrt(predsig2 * length(y2) / (length(y2) - 2))
  
  ### Cokriging ###
  tic.cokm <- proc.time()[3]
  # NestedX already returns a nested design with X2 = last nrow(X2) rows of X1, so build the
  # NestDesign directly. Re-running the substitution via NestedDesignBuild can break in 1-D
  # (e.g. seed 5 gives a 52-row level-1 design for 51 responses -> "invalid class km object").
  NestDesign <- NestedDesign(X1, nlevel = 2,
                             indices = list(seq(nrow(X1) - nrow(X2) + 1, nrow(X1))))
  # A tiny fixed nugget is required: without it the gauss correlation matrix of the 1-D
  # level-2 design is numerically singular ("leading minor ... not positive") on every seed.
  fit.muficokm <- MuFicokm(formula = list(~1,~1),
                           MuFidesign = NestDesign,
                           covtype="gauss",
                           nugget=1e-8,
                           lower=1e-4,
                           response = list(y1,y2),
                           nlevel = 2)
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
  result.perd.rmse[i,1] <- sqrt(mean((predy-y)^2)) 
  result.perd.rmse[i,2] <- sqrt(mean((pred.muficokm$mean-y)^2)) 
  
  # Calculate execution times
  time.RNAmf <- toc.RNAmf - tic.RNAmf
  time.cokm <- toc.cokm - tic.cokm
  
  # Save individual replication times for Python
  rna_time[i] <- time.RNAmf
  cokm_time[i] <- time.cokm
  
  # Write the stacked files every 10 replicates (checkpoint) and at the end
  if (i %% 10 == 0 || i == n.rep) save_stacked()
  
  boxplot(result.perd.rmse[,, drop=FALSE])
  
}