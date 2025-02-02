import numpy as np
import pandas as pd


def expand_user_ids(df_user_min_day: pd.DataFrame, num_days: int):
    """Expand user ids over time to create a user-day matrix.
    Users IDs are flagged daily since their first day of activity.
    This can be used as an auxiliar table to join and compute daily cumulative statistics.
    """
    # Create user-day boolean matrix
    is_active = np.zeros((len(df_user_min_day), num_days), dtype=bool)
    # Set True values for first day and onwards
    for i, first_day in enumerate(df_user_min_day):
        is_active[i, first_day:] = True
    # Extract user ids and days where active
    user_ids, days = np.where(is_active)
    # Convert arrays into DataFrame
    expanded_user_ids_over_time = pd.DataFrame({"user_id": df_user_min_day.index[user_ids], "day": days})
    return expanded_user_ids_over_time


def redistribute_non_assigned_users(num_users, num_arms):
    """If weights result in non-integer number of users, redistribute the remaining users."""
    if num_users > 0:
        array_with_ones = np.ones((num_users, 1), dtype=int)
        arms_matrix = np.zeros((num_arms, num_users), dtype=int)
        arms_matrix[:num_users] = array_with_ones
        return arms_matrix.sum(axis=1)
    else:
        return np.zeros(num_arms, dtype=int)


# Uplift function: Compute uplifted value based on control value and uplift percentage
def compute_uplift(control_val, uplift_pctg, positive_bounded=True):
    """Compute uplifted value based on control value and uplift percentage."""
    arm_value = control_val * (1 + uplift_pctg)
    if positive_bounded:
        return max(0.0, arm_value)
    else:
        return arm_value