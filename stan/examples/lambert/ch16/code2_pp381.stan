// functions need to be declared before its use in the model block
functions {
    real covariateMean(real aX, real aBeta)
    {
        return (aBeta * log(aX));
    }
}
// functions can also be used to allow sampling from any distribution whose log density can be written in Stan code


data {
    int<lower=1> N;   // number of observations
    array[N] real Y;  // heights for N people
    array[N] real X;  // covariate for N people
}

parameters {
    real beta;            // coefficient for covariate
    real<lower=0> sigma;  // std of height distribution
}

model {
    for (i in 1:N)
    {
        Y[i] ~ normal(covariateMean(X[i], beta), sigma);  // likelihood
    }

    beta ~ normal(0, 1);  // prior for beta
    sigma ~ gamma(1, 1);  // prior for sigma
}