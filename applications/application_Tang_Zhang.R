
file_loc<-'./application/'
ScoreA<-as.matrix(read.table(paste0(file_loc,'ScoreA.txt')) )
ScoreB<-as.matrix(read.table(paste0(file_loc,'ScoreB.txt')) )
ScoreC<-as.matrix(read.table(paste0(file_loc,'ScoreC.txt')) )
grp_id<-c(rep('A',nrow(ScoreA)),rep('B',nrow(ScoreB)),rep('C',nrow(ScoreC)))
all_data_wide<-rbind(ScoreA,ScoreB,ScoreC)
nall<-nrow(all_data_wide)
all_data_array<-array(,c(nall,18,3) )  
all_data_array[,,1]<-all_data_wide[,1:18]
all_data_array[,,2]<-all_data_wide[,19:36]
all_data_array[,,3]<-all_data_wide[,37:54]

Qmat<-as.matrix(read.table(paste0(file_loc,'Q.txt')) )

#fitting via Stan

mod_mixture_wI_loaded <- cmdstan_model(exe_file = "./code/model_dina_hmm_mixture_wI.exe")



###simulate data

y <- all_data_array
Q <- Qmat[,-6]
K <- ncol(Q)
J <- nrow(Q)
A <- (as.matrix(expand.grid(rep(list(c(0, 1)), K))))[,K:1]#alpha patterns in order of bijection
nall <- dim(y)[1]
obs <- !is.na(y) * 1

# Step 2: Prepare your data
two_pow_K<-2^K
eta <- matrix(0, nrow = two_pow_K, ncol = J)
for (k in 1:two_pow_K) {
  for (j in 1:J) {
    eta[k, j] <- as.integer(all(Q[j, ] <= A[k, ]))
  }
}


library(cmdstanr)
data_list_mixture <- list(
  n = dim(y)[1],
  J = dim(y)[2],
  T = dim(y)[3],
  two_pow_K = two_pow_K,
  K = K,
  y = y,
  obs = obs,
  eta = eta,
  Apatterns = A,
  fixed_idx =c(1,2),
  free_idx = c(3:5),
  grainsize=4
)


mcmc_mixture_wI <- mod_mixture_wI_loaded$sample(
  data = data_list_mixture,
  chains = 5,
  parallel_chains = 5,
  iter_warmup = 1000,
  iter_sampling = 1000,
  seed = 123
)
# #11797.0
# 
cmd_out<-mcmc_mixture_wI$summary(variables=c('slip','guess','p_00_1','p_11_1','p_00_2','p_11_2_free','ws','init_state'))
View(cmd_out)
# 
# # draws <- fit_indep_trans$draws(variables=c('slip','guess','p_00'))  # for posterior samples
# summary(draws)
# fit_indep_trans$cmdstan_diagnose() 
# 

#map estimation
fit_mixture_wI_map <- mod_mixture_wI_loaded$optimize(
  data = data_list_mixture,
  jacobian = F  # Use TRUE for MAP, FALSE for MLE
)

lp_wI<-fit_mixture_wI_map$summary(variables='lp__')
nparm_wI<-2*J+two_pow_K-1+4*K
AIC_indep_mixture<- -2*lp_wI[1,2]+2*nparm_wI
BIC_indep_mixture<- -2*lp_wI[1,2]+log(nall)*nparm_wI

# View a summary table of the parameter estimates
View(fit_mixture_wI_map$summary(variables=c('slip','guess','p_00_1','p_11_1','p_00_2','p_11_2_free','ws','init_state')))


#dina with general hmm
#load compiled saved model
# model_dina_hmm <- readRDS("./code/compiled_model_dina_hmm.rds")
mod_dina_hmm_loaded <- cmdstan_model(exe_file = "./code/dina_hmm_stan.exe")

data_list_hmm <- list(
  n = dim(y)[1],
  J = dim(y)[2],
  T = dim(y)[3],
  two_pow_K = two_pow_K,
  K = K,
  y = y,
  obs = obs,
  eta = eta
)

fit_hmm <- mod_dina_hmm_loaded$optimize(
  data = data_list_hmm,
  jacobian = F  # Use TRUE for MAP, FALSE for MLE
)

lp_dina_hmm<-fit_hmm$summary(variables='lp__')
nparm_dina_hmm<-2*J+two_pow_K-1+(two_pow_K-1)*two_pow_K
AIC_hmm <- -2*lp_dina_hmm[1,2]+2*nparm_dina_hmm
BIC_hmm <- -2*lp_dina_hmm[1,2]+log(nall)*nparm_dina_hmm

#dina with independent transitions
mod_dina_indep_trans_loaded <- cmdstan_model(exe_file = "./code/dina_hmm_indep_trans.exe")

data_list_indep_trans <- list(
  n = dim(y)[1],
  J = dim(y)[2],
  T = dim(y)[3],
  two_pow_K = two_pow_K,
  K = K,
  y = y,
  obs = obs,
  eta = eta,
  Apatterns = A
)

fit_indep_tran <- mod_dina_indep_trans_loaded$optimize(
  data = data_list_indep_trans,
  jacobian = F  # Use TRUE for MAP, FALSE for MLE
)

lp_dina_indep_tran<-fit_indep_tran$summary(variables='lp__')
nparm_dina_indep_tran<-2*J+two_pow_K-1+2*K
AIC_indep <- -2*lp_dina_indep_tran[1,2]+2*nparm_dina_indep_tran
BIC_indep <- -2*lp_dina_indep_tran[1,2]+log(nall)*nparm_dina_indep_tran

#dina with mixture of independent transitions and I
mod_dina_indep_trans_pI_loaded <- cmdstan_model(exe_file = "./code/dina_hmm_indep_trans_pI.exe")

fit_indep_tran_pI <- mod_dina_indep_trans_pI_loaded$optimize(
  data = data_list_indep_trans,
  jacobian = F  # Use TRUE for MAP, FALSE for MLE
)

lp_dina_indep_tran_pI<-fit_indep_tran_pI$summary(variables='lp__')
nparm_dina_indep_tran_pI<-2*J+two_pow_K-1+2*K+1
AIC_indep_pI<- -2*lp_dina_indep_tran_pI[1,2]+2*nparm_dina_indep_tran_pI
BIC_indep_pI<- -2*lp_dina_indep_tran_pI[1,2]+log(nall)*nparm_dina_indep_tran_pI

View(fit_indep_tran_pI$summary(variables=c('slip','guess','p_00','p_11','ws','init_state')))




#dina with mixture of independent transitions + monotonicity and I
mod_dina_indep_trans_pI_monotone_loaded <- cmdstan_model(exe_file = "./code/dina_hmm_indep_trans_pI_monotone.exe")

fit_indep_tran_pI_monotone <- mod_dina_indep_trans_pI_monotone_loaded$optimize(
  data = data_list_indep_trans,
  jacobian = F  # Use TRUE for MAP, FALSE for MLE
)

lp_dina_indep_tran_pI_monotone<-fit_indep_tran_pI_monotone$summary(variables='lp__')
nparm_dina_indep_tran_pI_monotone<-2*J+two_pow_K-1+K+1
AIC_indep_pI_monotone<- -2*lp_dina_indep_tran_pI_monotone[1,2]+2*nparm_dina_indep_tran_pI_monotone
BIC_indep_pI_monotone<- -2*lp_dina_indep_tran_pI_monotone[1,2]+log(nall)*nparm_dina_indep_tran_pI_monotone



#dina with monotone A

dina_hmm_fixed0s_stan<-"
data {
  int<lower=1> n;     // number of persons
  int<lower=1> J;     // number of items
  int<lower=1> T;     // number of time points
  int<lower=1> two_pow_K;     // number of latent classes
  int<lower=1> K;     // number of attributes
  array[n, J, T] int<lower=0, upper=1> y;
  array[n, J, T] int<lower=0, upper=1> obs;
  array[two_pow_K, J] int<lower=0, upper=1> eta;
}


parameters {
  simplex[two_pow_K] init_state;
  vector<lower=0, upper=1>[J] slip;    // slipping parameters
  vector<lower=0, upper=1>[J] u;       // scaling factor for guess
  %s
  }

transformed parameters {
  vector<lower=0, upper=1>[J] guess;
  matrix[two_pow_K, J] theta;
  
  for (j in 1:J)
    guess[j] = (1 - slip[j]) * u[j];  // ensures monotonicity
  
  for (k in 1:two_pow_K) {
    for (j in 1:J) {
      theta[k, j] = eta[k, j] * (1 - slip[j]) + (1 - eta[k, j]) * guess[j];
    }
  }
    
  matrix[two_pow_K, two_pow_K] trans_mat = rep_matrix(0, two_pow_K, two_pow_K);
  %s
  
}

model {
  // Priors

  slip ~ beta(1, 2);   // flat prior for s and g on montone space
  u ~ beta(1, 1);
  
  // Likelihood
  for (i in 1:n) {
    matrix[two_pow_K, T] log_lik;
  
    for (t in 1:T) {
      for (k in 1:two_pow_K) {
        real lp = 0;
        for (j in 1:J) {
          if (obs[i, j, t] == 1)
            lp += bernoulli_lpmf(y[i, j, t] | theta[k, j]);
        }
        log_lik[k, t] = lp;
      }
    }
  
    // Correct: pass one person's init state vector
    target += hmm_marginal(log_lik, trans_mat, init_state);
  }
}
"
library(hmcdm)
TP <- TPmat(K)

library(stringr)
# 1. Define your mask (1 = transition allowed, 0 = structural zero)
mask <- TP[-two_pow_K,]

# 2. Generate the Parameter Strings
param_lines <- c()
trans_lines <- c()

for (i in 1:(two_pow_K-1)) {
  # Find which columns are allowed for this row
  allowed_cols <- which(mask[i, ] == 1)
  n_allowed <- length(allowed_cols)
  
  # Create parameter: simplex[n] row_1;
  param_lines <- c(param_lines, sprintf("  simplex[%d] row_%d;", n_allowed, i))
  
  # Create mapping logic: trans_mat[1, c(1,2)] = row_1;
  # Note: Stan doesn't support c(1,2) syntax, so we loop or use slices
  for (j in 1:n_allowed) {
    trans_lines <- c(trans_lines, 
                     sprintf("  trans_mat[%d, %d] = row_%d[%d];", i, allowed_cols[j], i, j))
  }
}
#include line of code so last element is 1
trans_lines<-c(trans_lines, 
               sprintf("  trans_mat[%d, %d] = 1;", two_pow_K,two_pow_K))

# 3. Construct the Full Stan Code
final_stan_code <- sprintf(dina_hmm_fixed0s_stan, 
                           paste(param_lines, collapse = "\n"), 
                           paste(trans_lines, collapse = "\n"))

# 4. Save to a temporary file
writeLines(final_stan_code, con = "dina_hmm_fixed0s.stan")

# mod_dina_hmm_fixed0s <- cmdstan_model("dina_hmm_fixed0s.stan",dir = "./code/")

# library(cmdstanr)

stan_data <- list(
  n = dim(y)[1],
  J = dim(y)[2],
  T = dim(y)[3],
  two_pow_K = nrow(A),
  K = ncol(Q),
  y = y,
  obs = obs,#missing data indicator
  eta=eta
)

N<-dim(y)[1]

fit_hmm_fixed0s <- mod_dina_hmm_fixed0s$optimize(
  data = stan_data,
  jacobian = F  # Use TRUE for MAP, FALSE for MLE
)

lp_dina_hmm_fixed0s<-fit_hmm_fixed0s$summary(variables='lp__')
nparm_dina_hmm_fixed0s<-2*J+two_pow_K-1+3^K-two_pow_K
AIC_hmm_fixed0s <- -2*lp_dina_hmm_fixed0s[1,2]+2*nparm_dina_hmm_fixed0s
BIC_hmm_fixed0s <- -2*lp_dina_hmm_fixed0s[1,2]+log(N)*nparm_dina_hmm_fixed0s

#########################################################
#monotone-one skill at-a-time
sq_dist_attributes<-as.matrix(dist(A,diag=T,upper=T)^2)
one_diff<-matrix(1*(sq_dist_attributes<2),two_pow_K,two_pow_K)
TP2<-TP*one_diff

# 1. Define your mask (1 = transition allowed, 0 = structural zero)
mask <- TP2[-two_pow_K,]

# 2. Generate the Parameter Strings
param_lines <- c()
trans_lines <- c()

for (i in 1:(two_pow_K-1)) {
  # Find which columns are allowed for this row
  allowed_cols <- which(mask[i, ] == 1)
  n_allowed <- length(allowed_cols)
  
  # Create parameter: simplex[n] row_1;
  param_lines <- c(param_lines, sprintf("  simplex[%d] row_%d;", n_allowed, i))
  
  # Create mapping logic: trans_mat[1, c(1,2)] = row_1;
  # Note: Stan doesn't support c(1,2) syntax, so we loop or use slices
  for (j in 1:n_allowed) {
    trans_lines <- c(trans_lines, 
                     sprintf("  trans_mat[%d, %d] = row_%d[%d];", i, allowed_cols[j], i, j))
  }
}
#include line of code so last element is 1
trans_lines<-c(trans_lines, 
               sprintf("  trans_mat[%d, %d] = 1;", two_pow_K,two_pow_K))

# 3. Construct the Full Stan Code
final_stan_code <- sprintf(dina_hmm_fixed0s_stan, 
                           paste(param_lines, collapse = "\n"), 
                           paste(trans_lines, collapse = "\n"))

# 4. Save to a temporary file
writeLines(final_stan_code, con = "dina_hmm_fixed0s.stan")

# mod_dina_hmm_fixed0s2 <- cmdstan_model("dina_hmm_fixed0s.stan",dir = "./code/")

fit_hmm_fixed0s2 <- mod_dina_hmm_fixed0s2$optimize(
  data = stan_data,
  jacobian = F  # Use TRUE for MAP, FALSE for MLE
)

lp_dina_hmm_fixed0s2<-fit_hmm_fixed0s2$summary(variables='lp__')
nparm_dina_hmm_fixed0s2<-2*J+two_pow_K-1+K*2^(K-1)#sum(TP2)-two_pow_K
AIC_hmm_fixed0s2 <- -2*lp_dina_hmm_fixed0s2[1,2]+2*nparm_dina_hmm_fixed0s2
BIC_hmm_fixed0s2 <- -2*lp_dina_hmm_fixed0s2[1,2]+log(N)*nparm_dina_hmm_fixed0s2





##create a table
# aics<-c(AIC_indep,AIC_indep_pI,AIC_indep_mixture,AIC_hmm)
# bics<-c(BIC_indep,BIC_indep_pI,BIC_indep_mixture,BIC_hmm)
# nparms<-c(nparm_dina_indep_tran,nparm_dina_indep_tran_pI,nparm_wI,nparm_dina_hmm)
# lps<- c(lp_dina_indep_tran[1,2]$estimate,lp_dina_indep_tran_pI[1,2]$estimate,lp_wI[1,2]$estimate,lp_dina_hmm[1,2]$estimate)
# tab_summary<- cbind(nparms,nparms-2*J-(two_pow_K-1),-2*lps,aics,bics)
aics<-c(AIC_indep_pI_monotone,AIC_indep,AIC_indep_pI,AIC_indep_mixture,AIC_hmm_fixed0s2,AIC_hmm_fixed0s,AIC_hmm)
bics<-c(BIC_indep_pI_monotone,BIC_indep,BIC_indep_pI,BIC_indep_mixture,BIC_hmm_fixed0s2,BIC_hmm_fixed0s,BIC_hmm)
nparms<-c(nparm_dina_indep_tran_pI_monotone,nparm_dina_indep_tran,nparm_dina_indep_tran_pI,nparm_wI,nparm_dina_hmm_fixed0s2,nparm_dina_hmm_fixed0s,nparm_dina_hmm)
lps<- c(lp_dina_indep_tran_pI_monotone$estimate,lp_dina_indep_tran[1,2]$estimate,lp_dina_indep_tran_pI[1,2]$estimate,lp_wI[1,2]$estimate,
        lp_dina_hmm_fixed0s2$estimate,lp_dina_hmm_fixed0s$estimate,lp_dina_hmm[1,2]$estimate)
tab_summary<- cbind(nparms,nparms-2*J-(two_pow_K-1),-2*lps,aics,bics)


# save(fit_mixture_wI_map,fit_hmm,fit_indep_tran,fit_indep_tran_pI,tab_summary,
#      fit_indep_tran_pI_monotone,fit_hmm_fixed0s2,fit_hmm_fixed0s,file='./application/tang_zhang_results030826.Rdata')
# load(file='./application/tang_zhang_results030826.Rdata')

library(xtable)
rownames(tab_summary)<-c('$\\mathbf A_+(\\bm\\tau,\\omega)$','Independent','$\\mathbf A(\\bm\\tau,\\omega)$',
                         '$\\mathbf A(\\bm\\tau_1,\\bm\\tau_2,\\omega)$','$\\mathbf A_{+,1}$','$\\mathbf A_+$','$\\mathbf A$')
colnames(tab_summary)<-c('Total','Transition','-2LL','AIC','BIC')

print(xtable(tab_summary,
             digits=rep(0,6)),hline.after=c(-1,0,nrow(tab_summary)),
      sanitize.text.function = function(x) x,include.rownames = T)

which.min(aics)
which.min(bics)


est_table<-cbind(fit_indep_tran_pI_monotone$summary(variables=c('p_00') ),1)[,-c(1)]
rownames(est_table)<-c('Rational Numbers','Related Concepts of Rational Numbers','Axis','Addition and Subtract','Multiplication and Division')
colnames(est_table)<-c('$p_{00}$','$p_{11}$')

print(xtable(est_table,digits=c(rep(2,2),0)),hline.after=c(-1,0,nrow(est_table)),
      sanitize.text.function = function(x) x,include.rownames = T)
fit_indep_tran_pI_monotone$summary(variables=c('ws') )

