data {
    real mu;
    real kappa;
}

model {

}

generated quantities {
   int Y;
   Y = neg_binomial_2_rng(mu, kappa);
}