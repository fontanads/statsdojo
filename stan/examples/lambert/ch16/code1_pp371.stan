// Section 16.5.1 The Main Blocks of a Stan Program
data {
    int<lower=1> N;        // number of observations
    array[N] real Y;  // heights for N people
}

transformed data {
    array[N] real lSqDeviation;
    for (i in 1:N)
    {
        lSqDeviation[i] = (Y[i] - mean(Y)) ^ 2;
    }
}

parameters {
    real mu;  // mean height in population
    // real<lower=0> sigma;  // std of height distribution
    real<lower=0> sigmaSq;  // variance of height distribution
}

transformed parameters {
    real<lower=0> sigma;    // std of height distribution
    sigma = sqrt(sigmaSq);  // we use the square root of the variance to get the standard deviation
}

model {
    for (i in 1:N)
    {
        Y[i] ~ normal(mu, sigma);  // likelihood
    }

    mu ~ normal(1.5, .1);  // prior for mu
    // sigma ~ gamma(1, 1);  // prior for sigma
    sigmaSq ~ gamma(5, 1);  // prior for sigmaSq
}

// added in Section 16.5.4 The Other Blocks - Generated Quantities
generated quantities {
   // this block can be used to generate quantities of interest that are not part of the model
   // for example, posterior predictive checks
   // or to generate samples from parameters that interest us at a given level of a hierarchical model (Chapter 17)

   // declaring variables we intend to use in this block
   int aMax_indicator;
   int aMin_indicator;

    // Alternative 1:
    // if done this way, we'll keep the values of N elements of simulated data vector
//    vector[N] lSimData;  

   // we then use the posterior samples of mu and sigma 
   // to generate posterior predictive samples of the same length as the observed data

   // Generate posterior predictive samples
    // for (i in 1:N)
    // {
    //      lSimData[i] = normal_rng(mu, sigma);
         
    //      // the rng suffix is to generate  a single pseudo-independent draw from a normal distribution
    //      // with mean mu and standard deviation sigma

    //      // this is different from the likelihood block where we use the ~ operator
    //      // because there we are incrementing the overall log probability by an amount given by the log likelihood of a data point Y
    //      // for a normal distribution with mean mu and standard deviation sigma
    // }

    // Alternative 2:
    // using a local block to create a local scope for the simulated data
    {
        // we declare the variable inside the block
        // this data won't be kept in the output of the model
        vector[N] lSimData;  
        for (i in 1:N)
        {
            lSimData[i] = normal_rng(mu, sigma);
        }

    // we then determine weather the maximum and minimum of the simulated data are more extreme than the observed data
    // for that we create a binary indicator for each case
    // these will represent the "Bayesian p-values"

    // Compare with real data
    // this data is available in the output because the variables have been declared in the outer scope
    aMax_indicator = max(lSimData) > max(Y);
    aMin_indicator = min(lSimData) < min(Y);

    }
}