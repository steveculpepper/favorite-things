dina_hmm_mixture_wI_stan<-"
functions {
  // --- Optimized Kronecker ---
    int num_to_int(real x) { return x > 0.5 ? 1 : 0; }
  
  matrix kronecker_list_general(array[] matrix T, int K, int S, matrix Apatterns) {
    matrix[S, S] Gamma;
    for (i in 1:S) {
      for (j in 1:S) {
        real val = 1.0;
        for (k in 1:K) {
          int r_bit = num_to_int(Apatterns[i,k]); 
          int c_bit = num_to_int(Apatterns[j,k]);
          val *= T[k][r_bit + 1, c_bit + 1];
        }
        Gamma[i, j] = val;
      }
    }
    return Gamma;
  }
  
  // --- The Partial Sum Function for Parallelization ---
    real partial_log_lik_lpmf(array[] int seq, int start, int end,
                              array[,,] int y, array[,,] int obs,
                              matrix log_prob_1, matrix log_prob_0,
                              matrix trans_mat, vector init_state,
                              int J, int T, int S) {
      real p_log_lik = 0;
      
      // Loop over the slice of subjects (from start to end)
      for (i in start:end) {
        matrix[S, T] log_lik_matrix;
        for (t in 1:T) {
          for (k in 1:S) {
            real lp = 0;
            for (j in 1:J) {
              if (obs[i, j, t] == 1) {
                lp += (y[i, j, t] == 1) ? log_prob_1[k, j] : log_prob_0[k, j];
              }
            }
            log_lik_matrix[k, t] = lp;
          }
        }
        p_log_lik += hmm_marginal(log_lik_matrix, trans_mat, init_state);
      }
      return p_log_lik;
    }
}

data {
  int<lower=1> n;
  int<lower=1> J;
  int<lower=1> T;
  int<lower=1> two_pow_K;
  int<lower=1> K;
  array[n, J, T] int<lower=0, upper=1> y;
  array[n, J, T] int<lower=0, upper=1> obs;
  array[two_pow_K, J] int<lower=0, upper=1> eta;
  matrix[two_pow_K, K] Apatterns;
  array[2] int<lower=1, upper=K> fixed_idx;
  array[K-2] int<lower=1, upper=K> free_idx;
  int<lower=1> grainsize; // Control parallel chunk size (try n/num_cores)
}

transformed data {
  matrix[two_pow_K, two_pow_K] Imat = diag_matrix(rep_vector(1.0, two_pow_K));
  array[n] int seq = rep_array(1, n); // Dummy sequence for reduce_sum
}

parameters {
  simplex[two_pow_K] init_state;
  vector<lower=0, upper=1>[J] slip;
  vector<lower=0, upper=1>[J] u;
  vector<lower=0, upper=1>[K] p_00_1;
  vector<lower=0, upper=1>[K] p_11_1;
  vector<lower=0, upper=1>[K] p_00_2;
  vector<lower=0, upper=1>[K-2] p_11_2_free;
  simplex[3] ws;
}

transformed parameters {
  matrix[two_pow_K, J] theta;
  vector<lower=0, upper=1>[J] guess;

  matrix[two_pow_K, J] log_prob_1;
  matrix[two_pow_K, J] log_prob_0;
  
  // 1. Vectorized Item Probabilities
  for (j in 1:J) {
    guess[j] = (1 - slip[j]) * u[j];
    for (k in 1:two_pow_K) {
      theta[k, j] = eta[k, j] * (1 - slip[j]) + (1 - eta[k, j]) * guess[j];
      log_prob_1[k, j] = log(theta[k, j]);
      log_prob_0[k, j] = log1m(theta[k, j]);
    }
  }
  
  // 2. Build Transitions (unchanged logic, but structured)
  vector[K] p_11_2;
  p_11_2[fixed_idx] = rep_vector(1.0, 2);
  p_11_2[free_idx] = p_11_2_free;
  
  array[K] matrix[2, 2] T_list_1;
  array[K] matrix[2, 2] T_list_2;
  for (k in 1:K) {
    T_list_1[k] = [[p_00_1[k], 1 - p_00_1[k]], [1 - p_11_1[k], p_11_1[k]]];
    T_list_2[k] = [[p_00_2[k], 1 - p_00_2[k]], [1 - p_11_2[k], p_11_2[k]]];
  }
  
  matrix[two_pow_K, two_pow_K] trans_mat_1 = kronecker_list_general(T_list_1, K, two_pow_K, Apatterns);
  matrix[two_pow_K, two_pow_K] trans_mat_2 = kronecker_list_general(T_list_2, K, two_pow_K, Apatterns);
  
  matrix[two_pow_K, two_pow_K] trans_mat = ws[1]*trans_mat_1 + ws[2]*trans_mat_2 + ws[3]*Imat;
}

model {
  // Priors
  slip ~ beta(1, 2);
  u ~ beta(1, 1);
  p_00_1 ~ beta(1, 1); p_11_1 ~ beta(1, 1);
  p_00_2 ~ beta(1, 1); p_11_2_free ~ beta(1, 1);
  
  // Parallelized Likelihood
  target += reduce_sum(partial_log_lik_lpmf, seq, grainsize,
                       y, obs, log_prob_1, log_prob_0,
                       trans_mat, init_state, J, T, two_pow_K);
}
"
