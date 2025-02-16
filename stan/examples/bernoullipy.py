import os
import numpy as np

from cmdstanpy import CmdStanModel

# set the path to the CmdStan installation
from cmdstanpy import cmdstan_path, set_cmdstan_path
set_cmdstan_path(os.path.join("C:", "Users", "fonta", "cmdstan"))
cmdstan_path()

# for reproducibility
__NP_SEED__ = 23456789
np.random.seed(__NP_SEED__)

# input params
N_data_num_samples = 1000
p_success_bernoulli = 0.3

MCMC_chains = 8
MCMC_warm_up_iters = 1000
MCMC_sampling_iters = 1000

PATH_FINDER_num_paths = 10

MODEL_filename = 'bernoulli.stan'

# DATASET FOR THE MODEL FITTING

# it can be provided from a json file, e.g.
# data = os.path.join(os.path.dirname(__file__), 'bernoulli.data.json')
# fit = model.sample(data=data)

# but it can be provided python objects directly, which i find more convenient to learn and debug
y = np.random.binomial(n=1, p=p_success_bernoulli, size=N_data_num_samples)
data = {'N': N_data_num_samples, 'y': y}

# Instantiate a STAN model
stan_model_parent_dir = os.path.dirname(__file__)
model = CmdStanModel(stan_file=os.path.join(stan_model_parent_dir, MODEL_filename))
print(model)
print(model.exe_info())

fit = model.sample(
    data=data,
    chains=MCMC_chains,
    iter_warmup=MCMC_warm_up_iters,
    iter_sampling=MCMC_sampling_iters
)

# INSPECTING MODEL FIT
print(fit.summary())  # print summary of the inference
print(fit.diagnose())  # print diagnostic information

# different ways to access the draws of the posterior distribution
print(fit.stan_variable('theta'))  # get the posterior draws of theta
print(fit.draws_pd('theta')[:3])  # get the first 3 draws of theta as a pandas DataFrame
print(fit.draws_xr('theta'))  # get the posterior draws of theta as an xarray.Dataset
# sometimes we want to use higher dimensional arrays (ndim > 2), or arrays for which the order of dimensions 
# (e.g., columns vs rows) shouldn’t really matter. 
# For example, the images of a movie can be natively represented as an array with four dimensions: time, row, column and color.

print("Shape of the draws:")
for k, v in fit.stan_variables().items():  # print the shape of the posterior draws
    print(f'{k}\t{v.shape}')

print("Shape of the method variables:")
for k, v in fit.method_variables().items():  # print the shape of the method variables
    print(f'{k}\t{v.shape}')

print(f'numpy.ndarray of draws: {fit.draws().shape}')  # print the shape of the draws
# expect this to be (num iter sampling, chains, theta_dims + num vars in method)

print(fit.draws_pd())  # print the shape of the draws as a pandas DataFrame

print(fit.metric_type)  # print the metric type used in the adaptation
print(fit.metric)  # print the metric used in the adaptation
print(fit.step_size)  # print the step size used in the adaptation
print(fit.metadata.cmdstan_config['model'])  # print the model name
print(fit.metadata.cmdstan_config['seed'])  # print the seed used in the sampling

# MLE FITTING INSTEAD OF POSTERIOR SAMPLING
# run CmdStan's otpimize method, returns object `CmdStanMLE`
# mle = model.optimize(data=data_file)
mle = model.optimize(data=data)
print(mle.column_names)
print(mle.optimized_params_dict)
print(mle.optimized_params_pd)


# PATHFINDER EXAMPLE
# run CmdStan's pathfinder method, returns object `CmdStanPathfinder`
pathfinder = model.pathfinder(data=data, draws=10, num_paths=PATH_FINDER_num_paths)
print(pathfinder)
print(pathfinder.metadata)
print(pathfinder.stan_variable("theta").shape)
print(pathfinder.column_names)
print(pathfinder.draws().shape)
print(pathfinder.draws())
# theta
print(pathfinder.stan_variable("theta"))
# mean theta
print(pathfinder.stan_variable("theta").mean(axis=0))
print("done")