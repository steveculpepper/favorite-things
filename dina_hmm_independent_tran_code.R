
# temp_file_dina_hmm_indep_trans <- tempfile(fileext = ".stan")

dina_hmm_indep_trans_stan<-"
functions {
  int num_to_int(real x) {
    return x > 0.5 ? 1 : 0;
  }
  // Recursive Kronecker product for a list of matrices
  matrix kronecker_list_general(array[] matrix T, int K,int two_pow_K,matrix Apatterns) {
    int S = two_pow_K; // Total number of states
    matrix[S, S] Gamma;
    
    for (i in 1:S) {//row
      for (j in 1:S) {//column
        real val = 1.0;
        //int row_idx = i - 1; // 0-indexed for bit logic
        //int col_idx = j - 1;
        
        for (k in 1:K) {
          // Determine if trait k is 0 or 1 for this state index
          // We look at the bits from most significant to least significant
          //int shift = K - k;
          int r_bit = num_to_int(Apatterns[i,k]); 
          int c_bit = num_to_int(Apatterns[j,k]);
          
          // Multiply by the corresponding entry in the k-th 2x2 matrix
          // Adding 1 because Stan is 1-indexed
          val *= T[k][r_bit + 1, c_bit + 1];
        }
        Gamma[i, j] = val;
      }
    }
    return Gamma;
  }
}

data {
  int<lower=1> n;     // number of persons
  int<lower=1> J;     // number of items
  int<lower=1> T;     // number of time points
  int<lower=1> two_pow_K;     // number of latent classes
  int<lower=1> K;     // number of attributes
  array[n, J, T] int<lower=0, upper=1> y;
  array[n, J, T] int<lower=0, upper=1> obs;
  array[two_pow_K, J] int<lower=0, upper=1> eta;
  matrix[two_pow_K,K] Apatterns;
}


parameters {
  simplex[two_pow_K] init_state;
  array[two_pow_K] simplex[two_pow_K] trans_mat_array;
  vector<lower=0, upper=1>[J] slip;    // slipping parameters
  vector<lower=0, upper=1>[J] u;       // scaling factor for guess
  vector<lower=0, upper=1>[K] p_00; //prob of staying 0
  vector<lower=0, upper=1>[K] p_11; //prob of staying 1
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
  
  array[K] matrix[2, 2] T_list;
  matrix[two_pow_K,two_pow_K] trans_mat;
  
  for (k in 1:K) {
    T_list[k, 1, 1] = p_00[k];
    T_list[k, 1, 2] = 1 - p_00[k];
    T_list[k, 2, 1] = 1 - p_11[k];
    T_list[k, 2, 2] = p_11[k];
  }
  
  trans_mat = kronecker_list_general(T_list, K,two_pow_K,Apatterns);
}

model {
  // Priors
  //not specifying prior for init_state => uniform on simplex

  slip ~ beta(1, 2);   // or more informative if you prefer
  u ~ beta(1, 1);
  p_00 ~ beta(1,1);
  p_11 ~ beta(1,1);
  
  
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

writeLines(dina_hmm_indep_trans_stan, con = "dina_hmm_indep_trans.stan")

# mod_dina_indep_trans <- cmdstan_model("dina_hmm_indep_trans.stan",dir = "C:/Users/steve/Dropbox/Apps/ShareLaTeX/stochastic matrix mixture/code/")
# mod_dina_indep_trans_loaded <- cmdstan_model(exe_file = "C:/Users/steve/Dropbox/Apps/ShareLaTeX/stochastic matrix mixture/code/dina_hmm_indep_trans.exe")



# writeLines(dina_hmm_indep_trans_stan, con = temp_file_dina_hmm_indep_trans)

# generated quantities { - consider this for classification


##Step 0: compile model
# library(rstan)

#compile, do not run
# model_dina_hmm <- stan_model(temp_file_dina_hmm)

#save model
# saveRDS(model_dina_hmm, file = "C:/Users/steve/Dropbox/Apps/ShareLaTeX/stochastic matrix mixture/code/compiled_model_dina_hmm.rds")

#load compiled saved model
# model_dina_hmm <- readRDS("C:/Users/steve/Dropbox/Apps/ShareLaTeX/stochastic matrix mixture/code/compiled_model_dina_hmm.rds")


###simulate data
# install.packages('hmcdm')
library(hmcdm)
# ??Q_matrix

latin_square <- matrix(c(
  1, 2, 3, 4, 5,
  2, 3, 4, 5, 1,
  3, 4, 5, 1, 2,
  4, 5, 1, 2, 3,
  5, 1, 2, 3, 4
), nrow = 5, byrow = TRUE)

n_per_group <- 400
num_groups <- 5

N <- n_per_group * num_groups 
J <- nrow(Q_matrix)
# N <- nrow(Design_array)
Design_array2<-hmcdm:::design_array(Test_order=latin_square,Test_versions=rep(1:5,each=n_per_group),Jt=J/5)

K <- ncol(Q_matrix)
L <- dim(Design_array2)[3]


#' Create a 2^K x 2^K transition matrix from a list of 2x2 matrices
create_kronecker_gamma <- function(T_list) {
  K <- length(T_list)
  
  # Initialize with the first matrix
  Gamma <- T_list[[1]]
  
  # Recursively apply kronecker product for the remaining K-1 matrices
  if (K > 1) {
    for (k in 2:K) {
      Gamma <- kronecker(Gamma, T_list[[k]])
    }
  }
  
  return(Gamma)
}


# --- Example Usage ---
# Define probabilities for 4 independent traits
p00 <- c(0.9, 0.8, 0.7, 0.95) # Prob stay in state 0
p11 <- c(0.7, 0.6, 0.8, 0.85) # Prob stay in state 1

# Create the list of 2x2 matrices
my_T_list <- lapply(1:4, function(k) {
  matrix(c(p00[k], 1 - p00[k], 
           1 - p11[k], p11[k]), 
         nrow = 2, byrow = TRUE)
})

# Generate the 16x16 matrix
Omega_true <- create_kronecker_gamma(my_T_list)


# Verify stochasticity (rows should sum to 1)
rowSums(Omega_true)

# TP <- TPmat(K)
# Omega_true <- rOmega(TP)


class_0 <- sample(1:2^K, N, replace = T)
Alphas_0 <- matrix(0,N,K)
for(i in 1:N){
  Alphas_0[i,] <- inv_bijectionvector(K,(class_0[i]-1))
}
# Alphas <- sim_alphas(model="FOHM", Design_array = Design_array2,Omega = Omega_true, N=N, L=L,alpha0=Alphas_0)
Alphas <- hmcdm:::simulate_alphas_FOHM(Omega = Omega_true,N=N,L=L,alpha0=Alphas_0)


itempars_true <- matrix(runif(J*2,.2,.2), ncol=2)
# nomissing_Design<-array(1,dim=dim(Design_array))
Y_sim <- sim_hmcdm(model="DINA",Alphas,Q_matrix,Design_array=Design_array2,
                   itempars=itempars_true)


###step 1: process missing data
# Assume y is n x J x T with NA for missing
y <- Y_sim
Q <- Q_matrix
A <- (as.matrix(expand.grid(rep(list(c(0, 1)), K))))[,K:1]#alpha patterns in order of bijection


obs <- !is.na(y) * 1
y[is.na(y)] <- 0  # dummy fill-in (will be ignored)


# # Step 1: Compile the Stan model
# library(rstan)
# 
# model <- stan_model(file = "dina_hmm.stan")  # or use a character string via model_code =

# Step 2: Prepare your data
two_pow_K<-2^K
eta <- matrix(0, nrow = two_pow_K, ncol = J)
for (k in 1:two_pow_K) {
  for (j in 1:J) {
    eta[k, j] <- as.integer(all(Q[j, ] <= A[k, ]))
  }
}

# stan_data <- list(
#   n = dim(y)[1],
#   J = dim(y)[2],
#   T = dim(y)[3],
#   K = ncol(Q),
#   two_pow_K = nrow(A),
#   y = y,
#   obs = obs,#missing data indicator
#   # Q = Q,
#   # A = A,
#   eta=eta
# )
# 
# 
# # Step 3: Fit the model
# fit <- sampling(model_dina_hmm,
#                 data = stan_data,
#                 chains = 1,
#                 iter = 2000,
#                 warmup = 1000,
#                 cores = 1,
#                 pars=c("slip","guess","init_state","trans_mat"),
#                 control = list(adapt_delta = 0.9))  # may need to increase adapt_delta
# 
# # Step 4: View results
# # print(fit, pars = c("slip", "guess", "trans_mat", "init_state"))
# summary(fit)$summary[,c('mean','Rhat',"n_eff")]
# 
# ##

#Fitting with cmdstanr
# install.packages("cmdstanr", repos = c("https://mc-stan.org/r-packages/", getOption("repos")))
# cmdstanr::check_cmdstan_toolchain(fix = TRUE)
# cmdstanr::install_cmdstan()
library(cmdstanr)
data_list <- list(
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

writeLines(dina_hmm_indep_trans_stan, con = "model_dina_hmm_indep_trans.stan")

mod <- cmdstan_model("model_dina_hmm_indep_trans.stan")

fit_indep_trans <- mod$sample(
  data = data_list,
  chains = 5,
  parallel_chains = 5,
  iter_warmup = 1000,
  iter_sampling = 1000,
  seed = 123
)
#12:54:00

cmd_out<-fit_indep_trans$summary(variables=c('slip','guess','p_00','p_11','init_state'))
View(cmd_out)

# draws <- fit_indep_trans$draws(variables=c('slip','guess','p_00'))  # for posterior samples
summary(draws)
fit_indep_trans$cmdstan_diagnose() 


#map estimation
fit_indep_trans_map <- mod$optimize(
  data = data_list,
  jacobian = F  # Use TRUE for MAP, FALSE for MLE
)

#
# View a summary table of the parameter estimates
View(fit_indep_trans_map$summary(variables=c('slip','guess','p_00','p_11','init_state')))

      