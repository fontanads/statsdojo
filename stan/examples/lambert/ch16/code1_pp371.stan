data {
    int<lower=1> N;        // number of observations
    array[N] real Y;  // heights for 10 people
}

parameters {
    real mu;  // mean height in population
    real<lower=0> sigma;  // std of height distribution
}

model {
    for (i in 1:N)
    {
        Y[i] ~ normal(mu, sigma);  // likelihood
    }

    mu ~ normal(1.5, .1);  // prior for mu
    sigma ~ gamma(1, 1);  // prior for sigma
}