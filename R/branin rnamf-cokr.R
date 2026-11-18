rm(list = ls())

install.packages("C:/Users/Santiago/OneDrive/Escritorio/Research/Orthogonal Project/Code/R/RNAmf_0.1.2.tar.gz", repos = NULL, type = "source")
library(DiceKriging)
library(RNAmf)
library(MuFiCokriging)
library(RcppCNPy)
library(glue)
library(lhs)

eps <- sqrt(.Machine$double.eps) # small nugget for numeric stability

### branin function ###
branin <- function(xx){
  x1 <- xx[1]
  x2 <- xx[2]
  
  (-1.275*x1^2/pi^2+5*x1/pi+x2-6)^2 + (10-5/(4*pi))*cos(x1)+ 10
}

braninm <- function(xx)
{ 
  x1 <- xx[1]
  x2 <- xx[2]
  
  10*sqrt((-1.275*(x1+2)^2/pi^2+5*(x1+2)/pi+(x2+2)-6)^2 + (10-5/(4*pi))*cos((x1+2))+ 10) + 2*(x1-0.5) - 3*(3*x2-1) - 1
}

braninl <- function(xx)
{ x1 <- xx[1]
x2 <- xx[2]

10*sqrt((-1.275*(1.2*x1+0.4)^2/pi^2+5*(1.2*x1+0.4)/pi+(1.2*x2+0.4)-6)^2 + (10-5/(4*pi))*cos((1.2*x1+0.4))+ 10) + 2*(1.2*x1+1.9) - 3*(3*(1.2*x2+2.4)-1) - 1 - 3*x2 + 1
}

output.branin <- function(x){
  factor_range <- list("x1" = c(-5, 10), "x2" = c(0, 15))
  
  for(i in 1:length(factor_range)) x[i] <- factor_range[[i]][1] + x[i] * diff(factor_range[[i]])
  branin(x[1:2])
} 

output.braninl <- function(x){
  factor_range <- list("x1" = c(-5, 10), "x2" = c(0, 15))
  
  for(i in 1:length(factor_range)) x[i] <- factor_range[[i]][1] + x[i] * diff(factor_range[[i]])
  braninl(x[1:2])
} 

output.braninm <- function(x){
  factor_range <- list("x1" = c(-5, 10), "x2" = c(0, 15))
  
  for(i in 1:length(factor_range)) x[i] <- factor_range[[i]][1] + x[i] * diff(factor_range[[i]])
  braninm(x[1:2])
} 


### training data ###
n1 <- 20; n2 <- 15; n3 <- 10; rep <- 100
d <- 2

# Create dynamic folder name based on n1 and n2
# Results are written straight into the Python data folder (no manual copying needed)
data_root <- "C:/Users/Santiago/OneDrive/Escritorio/Research/Orthogonal Project/Code/Python/Data"
dir_name <- file.path(data_root, glue("branin/{n1}_{n2}_{n3}"))

# Create the directory if it doesn't already exist
if (!dir.exists(dir_name)) {
  dir.create(dir_name, recursive = TRUE)
}

### test data ###
n.test <- 1000

cat("Branin function: n1 =", n1, ", n2 =", n2, ", n3 =", n3, ", rep =", rep, "\n")


result.branin.rmse <- matrix(NA, rep, 2)
colnames(result.branin.rmse) <- c("RNAmf", "Cokriging")

### test data ###
set.seed(0)
x <- maximinLHS(n.test, d)
y <- apply(x,1,output.branin)

npySave(glue("{dir_name}/x_test.npy"), x)
npySave(glue("{dir_name}/yhigh_test.npy"), y)

# --- Containers for saving: one .npy file per quantity, all replicates stacked ---
# X{l}.npy / y{l}.npy : rows of every replicate, first column = replicate index i
# {rna,cokm}_{mean,sigma2,lower,upper}.npy : (rep, n.test), row i = replicate i
# {rna,cokm}_time.npy : (rep,), entry i = replicate i
X_rows <- list(list(), list(), list())
y_rows <- list(list(), list(), list())
rna_mean <- rna_sigma2 <- rna_lower <- rna_upper <- matrix(NA_real_, rep, n.test)
cokm_mean <- cokm_sigma2 <- cokm_lower <- cokm_upper <- matrix(NA_real_, rep, n.test)
rna_time <- cokm_time <- rep(NA_real_, rep)

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

for(i in 1:rep) {
  set.seed(i)
  
  NX <- NestedX(c(n1, n2, n3), d)
  
  X1 = NX[[1]]
  X2 = NX[[2]]
  X3 = NX[[3]]
  
  X_rows[[1]][[i]] <- cbind(i, X1)
  X_rows[[2]][[i]] <- cbind(i, X2)
  X_rows[[3]][[i]] <- cbind(i, X3)
  
  y1 <- apply(X1,1,output.braninl)
  y2 <- apply(X2,1,output.braninm)
  y3 <- apply(X3,1,output.branin)

  y_rows[[1]][[i]] <- cbind(i, y1)
  y_rows[[2]][[i]] <- cbind(i, y2)
  y_rows[[3]][[i]] <- cbind(i, y3)
  

  ### RNAmf ###
  tic.RNAmf <- proc.time()[3]
  fit.RNAmf <- RNAmf_three_level(X1, y1, X2, y2, X3, y3, kernel="sqex", constant=TRUE)
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
  fit.muficokm <- MuFicokm(formula = list(~1,~1,~1), MuFidesign = NestDesign, covtype="gauss",
                           lower=eps, upper=0.5,
                           # coef.trend = list(0,c(0,0),c(0,0)),
                           response = list(y1,y2,y3), nlevel = 3)
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
  result.branin.rmse[i,1] <- sqrt(mean((predy-y)^2)) 
  result.branin.rmse[i,2] <- sqrt(mean((pred.muficokm$mean-y)^2)) 
  
  # Calculate execution times
  time.RNAmf <- toc.RNAmf - tic.RNAmf
  time.cokm <- toc.cokm - tic.cokm
  
  # Save individual replication times for Python
  rna_time[i] <- time.RNAmf
  cokm_time[i] <- time.cokm

  # Write the stacked files every 10 replicates (checkpoint) and at the end
  if (i %% 10 == 0 || i == rep) save_stacked()

  boxplot(result.branin.rmse[,, drop=FALSE])
}

