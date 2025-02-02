from typing import List
import numpy as np


def exp_cdf_parametrized(x: float, p0: float, lambda_: float) -> float:
    """
    Computes the cumulative return probability using a modified exponential CDF.
    
    Args:
        x (float): The time at which to evaluate the return probability.
        p0 (float): The probability of return at time x = 0.
        lambda_ (float): The rate parameter controlling the decay.

    Returns:
        float: The cumulative return probability at time x.
    """
    # The characteristic time scale of the process is given by: tau = 1 / lambda_
    # tau (time constant) represents the time required for probability decay to approximately 63.2% of its maximum change

    return p0 + (1 - p0) * (1 - np.exp(-lambda_ * x))


def next_day_return_probability(inactive_days: int, prob_next_day: float, tau_window_days: float):
    # most returns happen within tau_window days
    lambda_ = 1 / tau_window_days
    if inactive_days == 0:
        return prob_next_day
    else:
        cdf_x = exp_cdf_parametrized(inactive_days, prob_next_day, lambda_)
        cdf_previous_x = exp_cdf_parametrized(inactive_days - 1, prob_next_day, lambda_)
        return cdf_x - cdf_previous_x
    

def next_week_return_probability(inactive_weeks: int, prob_next_week: float, tau_window_weeks: float):
    # most returns happen within tau_window weeks
    lambda_ = 1 / tau_window_weeks
    if inactive_weeks == 0:
        return prob_next_week
    else:
        cdf_x = exp_cdf_parametrized(inactive_weeks, prob_next_week, lambda_)
        cdf_previous_x = exp_cdf_parametrized(inactive_weeks - 1, prob_next_week, lambda_)
        return cdf_x - cdf_previous_x

def return_probability(inactive_days: int, prob_next_day: float, prob_next_week: float, tau_window_days: float, tau_window_weeks: float):
    if inactive_days > 0 and inactive_days % 7 == 0:
        inactive_weeks = inactive_days // 7
        return next_week_return_probability(inactive_weeks, prob_next_week, tau_window_weeks)
    else:
        return next_day_return_probability(inactive_days, prob_next_day, tau_window_days)


# Example usage:
def main(
    inactive_days: List[int], 
    prob_next_day: float = 0.1,
    prob_next_week: float = 0.5,
    tau_window_days: float = 14,
    tau_window_weeks: float = 4,
):
    
    next_day_probs = [next_day_return_probability(x, prob_next_day, tau_window_days) for x in inactive_days]
    next_week_probs = [next_week_return_probability(x // 7, prob_next_week, tau_window_weeks) for x in inactive_days]
    return_probabilities = [return_probability(x, prob_next_day, prob_next_week, tau_window_days, tau_window_weeks) for x in inactive_days]
   
    return return_probabilities, next_day_probs, next_week_probs


if __name__ == "__main__":
    
    import matplotlib.pyplot as plt
    inactive_days = range(0, 30)

    return_probabilities, next_day_probs, next_week_probs = main(
        inactive_days, 
        prob_next_day=0.1, 
        prob_next_week=0.5, 
        tau_window_days=14,
        tau_window_weeks=4
    )

    plt.stem(inactive_days, next_day_probs, label='Next day', markerfmt='ro', linefmt='r--', basefmt='r-')
    plt.stem(inactive_days, next_week_probs, label='Next week', markerfmt='bo', linefmt='b--', basefmt='b-')
    plt.stem(inactive_days, return_probabilities, label='Combined', markerfmt='go', linefmt='g--', basefmt='g-')
    plt.xlabel('Days inactive')
    plt.ylabel('Return probability')
    plt.legend()
    plt.yscale('log')
    plt.show()