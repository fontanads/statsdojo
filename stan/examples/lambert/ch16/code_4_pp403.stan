data {
    int<lower=1> nStudy;        // number of studies
    int<lower=1> N;        // number of observations
    array[nStudy] int<lower=0, upper=N> X;  // number of successes per study
}

parameters {
    ordered[2] alpha;
}

transformed parameters {
    array[2] real<lower=0> theta;
    matrix[nStudy, 2] lp;

    // transform alpha to theta
    for(i in 1:2) 
    {
        // from real domain to probability
        theta[i] = inv_logit(alpha[i]);  // inv logit is the same as a the sigmoid function
    }

    // likelihood
    for(n in 1:nStudy) 
    {
        for(s in 1:2) 
        {
            lp[n, s] = log(0.5) + binomial_logit_lpmf(X[n] | N, alpha[s]);
        }
    }

}

model {
    for(n in 1:nStudy) 
    {
        target += log_sum_exp(lp[n]);
    }
}

generated quantities {
    array[nStudy] real pstate;
    for(n in 1:nStudy) 
    {
        pstate[n] = exp(lp[n, 1]) / (exp(lp[n, 1]) + exp(lp[n, 2]));
    }
}