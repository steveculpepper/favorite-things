
.libPaths("/u/sculpepp/R_libs/")

library("Rcpp")
library("RcppArmadillo")
library("posterior")
library("rlang")
library("rstan")
library("StanHeaders")
library("cmdstanr")
sourceCpp('./code/binary_rlcm_simcode.cpp')

args <- commandArgs(trailingOnly = TRUE)
# args<-c(1,1,1,3,1,232)

# Ns<-c(250,500,1000)
# Kvals<-c(3,5,7)
# rhos<-c(0,.35,.7)

sim.rep<-as.numeric(args[2])
# n_cond<-Ns[as.numeric(args[3])]
# K_cond <-Kvals[as.numeric(args[4])]
simulation_index<-as.numeric(args[6])


fname<-paste0('sim_data_',simulation_index,'.Rdata')
load(paste0('./simdata/',fname))

##stan model. Compiled once, saved exe file and load every time

temp_file_gdm <- tempfile(fileext = ".stan")

gdm_stan_vectorized<-"
data {
  int<lower=1> N;           // number of people
  int<lower=1> J;           // number of items
  int<lower=1> C;           // number of latent classes (2^K)
  array[N, J] int<lower=0, upper=1> Y; // responses
  array[J] int<lower=1> G;             // number of *distinct groups* per item
  array[J, C] int<lower=1> group;      // group mapping for each item/class.

  // Essential for accessing item_group_prob_flat
  array[J] int<lower=1> start_idx; // Starting index in item_group_prob_flat for each item
}

parameters {
  simplex[C] pi;                       // latent class proportions
  // item_group_prob_flat is a flattened vector of phijc's
  vector<lower=0, upper=1>[sum(G)] item_group_prob_flat; // Flattened probabilities for each item's groups
}

transformed parameters {
  matrix[J, C] theta;

  for (j in 1:J) {
    // Because your preprocessing `as.integer(factor(patterns))` ensures:
    // 1. The group ID for the least proficient pattern (often associated with class c=1) is 1.
    // 2. The group ID for the most proficient pattern (often associated with class c=C) is G[j].

    // Access the parameter for group 1 (always at the start of the item's block)
    real phi_val_group_1_for_item_j = item_group_prob_flat[start_idx[j]];

    // Access the parameter for group G[j] (always at the end of the item's block)
    real phi_val_group_C_for_item_j = item_group_prob_flat[start_idx[j] + G[j] - 1];

    real term_common = phi_val_group_1_for_item_j * phi_val_group_C_for_item_j;
    
    vector[G[j]] item_eqv_class_probs;
    item_eqv_class_probs[1] = term_common;
    item_eqv_class_probs[G[j]] = phi_val_group_C_for_item_j;
    if(G[j]>2){
      for(cc in 2:(G[j]-1)){
        real phi_val_current_group = item_group_prob_flat[start_idx[j] + cc - 1];
        item_eqv_class_probs[cc] = (1 - phi_val_current_group)*term_common + phi_val_current_group*phi_val_group_C_for_item_j;
      }
    }

    for (c in 1:C) {
      int current_group_id = group[j, c]; // This maps class 'c' to its specific group ID (1 to G[j])
      theta[j, c] = item_eqv_class_probs[current_group_id];
    }
  }
}

model {
  // Priors for item_group_prob_flat
  // `current_phi_idx_prior` correctly iterates through the flattened vector.
  // `g_idx_in_item` correctly represents the relative group index (1 to G[j])
  // within the current item's set of parameters.
  int current_phi_idx_prior = 1;
  for (j in 1:J) {
    for (g_idx_in_item in 1:G[j]) { // Loop over the actual number of groups for item j
      if (g_idx_in_item == 1) { // Apply beta prior for the first group of item j
        item_group_prob_flat[current_phi_idx_prior] ~ beta(1, G[j] - 1);
      } else if (g_idx_in_item == G[j]) { // Apply beta prior for the last group of item j
        item_group_prob_flat[current_phi_idx_prior] ~ beta(G[j], 1);
      } else { // Apply beta(1,1) for intermediate groups of item j
        item_group_prob_flat[current_phi_idx_prior] ~ beta(1,1);
      }
      current_phi_idx_prior += 1; // Move to the next element in the flattened vector
    }
  }
  
  pi ~ dirichlet(rep_vector(1.0, C));

  // Likelihood (unchanged from previous optimization)
  for (n in 1:N) {
    vector[C] log_lik = log(pi);
    for (c in 1:C) {
      log_lik[c] += bernoulli_lpmf(Y[n] | theta[, c]);
    }
    target += log_sum_exp(log_lik);
  }
}
"

writeLines(gdm_stan_vectorized, con = temp_file_gdm)


# A: 2^K x K attribute profiles
# Q: J x K
Amat<-CL_gen_invbijection_table(K,2,2^K)#possible attribute patterns

group <- matrix(NA, nrow = J, ncol = ncol(Amat))

Q<-Qt


full_etamat<-ETAmat(K,J,2,Qt)
dina_etamat<-full_etamat
dina_etamat[which(dina_etamat<1)]<-0

group<-dina_etamat+1

# for (j in 1:J) {
#   # required <- which(Qt[j, ] == 1)
#   required <- which(Q[j, ] == 1)
#   patterns <- apply(t(Amat)[, required, drop = FALSE], 1, paste0, collapse = "")
#   group[j, ] <- as.integer(factor(patterns))
# }

G <- apply(group, 1, max)  # number of unique groups per item
start_idx<-cumsum(c(1,G))[-(length(G)+1)]


Y<-Ysim
stan_data <- list(
  N = N,
  J = J,
  C = 2^K,
  Y = Y,
  G = G,
  group = group, # Pre-calculated as discussed
  start_idx = start_idx # Make sure this is passed!
)



writeLines(gdm_stan_vectorized, con = "model_gdm.stan")

# mod_gdm <- cmdstan_model("model_gdm.stan",dir = "./reparm/")
mod_gdm_loaded <- cmdstan_model(exe_file = "./reparm/model_gdm")

# t0<-proc.time()[3]
fit2 <- mod_gdm_loaded$sample(
  data = stan_data,
  chains = 5,
  parallel_chains = 5,
  iter_warmup = 1000,
  iter_sampling = 1000
)
# t1<-proc.time()[3]
timing_info <- fit2$time()
timing_info_fit2<- timing_info$total
# timing_info_fit2 <- t1-t0 #fit2$time()

cmd_out<-fit2$summary(variables=c('theta'))
cmd_out<-cmd_out[c(1:J,nrow(cmd_out)-J:1+1),]
conv_info<-cmd_out[,c('rhat','ess_bulk','ess_tail')]
conv_info<-cbind(conv_info,conv_info[,2:3]/timing_info_fit2)

theta_mean<-matrix(cmd_out$mean,J,2^K,byrow=F)
theta_diff<-cmd_out[,'mean']-rep(c(.2,.8),each=J)

cmd_out_pi<-fit2$summary(variables=c('pi'))
pi_diff<-cmd_out_pi$mean-pis1


#save output
output_fname<-paste0('output_transform_',simulation_index,'.Rdata')
save(timing_info_fit2,cmd_out,conv_info,theta_diff,cmd_out_pi,pi_diff,file=paste0('./dina_stan/output_transform/',output_fname) )

