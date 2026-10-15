
# temp_file_dina_hmm_indep_trans <- tempfile(fileext = ".stan")

dina_hmm_indep_trans_pI_stan<-"
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

transformed data {
  matrix[two_pow_K, two_pow_K] Imat = diag_matrix(rep_vector(1.0, two_pow_K));
}


parameters {
  simplex[two_pow_K] init_state;
  array[two_pow_K] simplex[two_pow_K] trans_mat_array;
  vector<lower=0, upper=1>[J] slip;    // slipping parameters
  vector<lower=0, upper=1>[J] u;       // scaling factor for guess
  vector<lower=0, upper=1>[K] p_00; //prob of staying 0
  vector<lower=0, upper=1>[K] p_11; //prob of staying 1
  simplex[2] ws;

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
  
  trans_mat = ws[1]*kronecker_list_general(T_list, K,two_pow_K,Apatterns)+ws[2]*Imat;
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

writeLines(dina_hmm_indep_trans_pI_stan, con = "dina_hmm_indep_trans_pI.stan")

# mod_dina_indep_trans_pI <- cmdstan_model("dina_hmm_indep_trans_pI.stan",dir = "C:/Users/steve/Dropbox/Apps/ShareLaTeX/stochastic matrix mixture/code/")
# mod_dina_indep_trans_pI_loaded <- cmdstan_model(exe_file = "C:/Users/steve/Dropbox/Apps/ShareLaTeX/stochastic matrix mixture/code/mod_dina_indep_trans_pI.exe")



# writeLines(dina_hmm_indep_trans_stan, con = temp_file_dina_hmm_indep_trans)

