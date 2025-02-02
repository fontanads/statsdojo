import numpy as np
import pandas as pd

from return_probability_model import return_probability
from utils import compute_uplift, expand_user_ids, redistribute_non_assigned_users
from user_class import User
from typing import List

# --------------------------------------------------------------------------- #
# ARBITRARY INPUT PARAMETERS: TRUE VALUES FOR THE MODEL
# --------------------------------------------------------------------------- #

# Global Parameters
__SEED__ = 12345  # Seed for reproducibility

POPULATION_SIZE = 1_000_000  # Total user base
DAILY_NEW_USERS_INJECTION_RATE = 0.001  # Fraction of population entering daily
TEST_DAYS = 30  # Number of days in the test

# Control group parameters
TRUE_CONVERSION_RATE = 0.1  # True probability of conversion in control
TRUE_REVENUE_PER_CUSTOMER = 100  # Mean revenue per converted customer
TRUE_PROB_RETURN_NEXT_DAY = 0.1
TRUE_PROB_RETURN_NEXT_WEEK = 0.2
# these are the same for all arms
TRUE_REVENUE_STD = 10  # Standard deviation of revenue
TRUE_TAU_WINDOW_DAYS = 14
TRUE_TAU_WINDOW_WEEKS = 4

K_ARMS = 4  # Number of arms in the A/B test
# per arm parameters: default example is control, A/A, Positive Uplift, Negative Uplift
SPLIT_WEIGHTS = np.array([0.25, 0.25, 0.25, 0.25])  # Must sum to 1
CONVERSION_RATE_UPLIFTS = np.array([0.0, 0.0, 0.1, -0.1])  # Relative increase in conversion or revenue per arm
REVENUE_PER_CUSTOMER_UPLIFTS = np.array([0.0, 0.0, 0.05, -0.05])  # Relative increase in conversion or revenue per arm
PROB_REUTRN_NEXT_DAY_UPLIFTS = np.array([0.0, 0.0, 0.2, -0.2])
PROB_RETURN_NEXT_WEEK_UPLIFTS = np.array([0.0, 0.0, 0.2, -0.2])

assert len(CONVERSION_RATE_UPLIFTS) == K_ARMS
assert len(REVENUE_PER_CUSTOMER_UPLIFTS) == K_ARMS
assert len(SPLIT_WEIGHTS) == K_ARMS
assert sum(SPLIT_WEIGHTS) == 1

# --------------------------------------------------------------------------- #
# PARAMETERS TO BE ESTIMATED IN EACH ARM
# --------------------------------------------------------------------------- #

vec_compute_uplift = np.vectorize(compute_uplift)
# adjusted values for treatment arms
p_k = vec_compute_uplift(TRUE_CONVERSION_RATE, np.array(CONVERSION_RATE_UPLIFTS))
r_k = vec_compute_uplift(TRUE_REVENUE_PER_CUSTOMER, np.array(REVENUE_PER_CUSTOMER_UPLIFTS))
rr_day_k = vec_compute_uplift(TRUE_PROB_RETURN_NEXT_DAY, np.array(PROB_REUTRN_NEXT_DAY_UPLIFTS))
rr_week_k = vec_compute_uplift(TRUE_PROB_RETURN_NEXT_WEEK, np.array(PROB_RETURN_NEXT_WEEK_UPLIFTS))
# these are the same for all arms
r_std_k = TRUE_REVENUE_STD * np.ones(K_ARMS)
tau_day_k = TRUE_TAU_WINDOW_DAYS * np.ones(K_ARMS)
tau_week_k = TRUE_TAU_WINDOW_WEEKS * np.ones(K_ARMS)

# --------------------------------------------------------------------------- #
# SIMULATION
# --------------------------------------------------------------------------- #

# Initialize simulation state
users: List[User] = []
current_user_id: int = 0

def simulate_day(day):
    np.random.seed(__SEED__ + day)
    global current_user_id
    
    new_users: List[User] = []
    # compute injection rate
    DAILY_NEW_ACTIVE_USERS = int(POPULATION_SIZE * DAILY_NEW_USERS_INJECTION_RATE)
    
    # allocation must be ensure that all of the new users injected get their assignment
    arm_allocation = (np.array(SPLIT_WEIGHTS) * DAILY_NEW_ACTIVE_USERS).astype(int)
    non_assigned_users = DAILY_NEW_ACTIVE_USERS - sum(arm_allocation)
    arm_allocation += redistribute_non_assigned_users(non_assigned_users, K_ARMS)

    # Inject new users
    for _ in range(DAILY_NEW_ACTIVE_USERS):
        new_user = User(current_user_id, day)
        new_users.append(new_user)
        current_user_id += 1

    # Assign new users to test arms
    np.random.shuffle(new_users)  # Shuffle users before allocation
    start = 0
    for k in range(K_ARMS):
        end = start + arm_allocation[k]
        for user in new_users[start:end]:
            user.assign_arm(k)
        start = end

    users.extend(new_users)

    daily_active_users = 0

    # Simulate conversions
    for user in users:

        if day > user.entry_day:
            # sample activity status of existing users
            user_return_prob_today = return_probability(user.days_inactive, rr_day_k[user.arm], rr_week_k[user.arm], tau_day_k[user.arm], tau_week_k[user.arm])
            is_active = np.random.rand() < user_return_prob_today
        else:
            # new users are always active
            is_active = True

        daily_active_users += is_active
        # updated days of inactivity and activity status
        user.update_activity(day, is_active)

        if user.daily_metrics.activity_status:
            user.attempt_conversion(p_k[user.arm], r_k[user.arm], r_std_k[user.arm])
        
        user.updated_user_history_metrics()

    print(f"User penetration @ day {day}: {len(users) / POPULATION_SIZE: .2%} {len(users):,d} /{POPULATION_SIZE:,d}")
    print(f"Penetration of DAU @ day {day}: {daily_active_users / POPULATION_SIZE: .2%} {daily_active_users:,d} / {POPULATION_SIZE:,d}")
    print(f"Daily active users: {daily_active_users}")
    return users

data_records = []
for day in range(TEST_DAYS):
    users = simulate_day(day)
    
    # Collect daily data
    for user in users:
        data_records.append(
                {
                    'day': day,
                    'user_id': user.user_id,
                    'arm': user.arm,
                    'is_active': user.daily_metrics.activity_status,
                    'converted': user.daily_metrics.converted,
                    'revenue': user.daily_metrics.revenue,
                    'transactions': user.daily_metrics.transactions,
                    # user history cumulative metrics 
                    'active_days': user.active_days,
                    'inactive_days': user.inactive_days,
                    'cumulative_converted': user.converted,
                    'cumulative_revenue': user.revenue,
                    'cumulative_transactions': user.transactions
                }
        )

# --------------------------------------------------------------------------- #
# POST-SIMULATION ANALYSIS
# --------------------------------------------------------------------------- #

# Create DataFrame
df = pd.DataFrame(data_records)
# Compute additional metrics
df['active_user_id'] = df['user_id'].where(df['is_active'], None)
df['revenue_per_transaction'] = df['cumulative_revenue'] / df['cumulative_transactions']
df['revenue_per_active_day'] = df['cumulative_revenue'] / df['active_days']
df['transactions_per_active_day'] = df['cumulative_transactions'] / df['active_days']
df['revenue_per_transaction_per_active_day'] = df['cumulative_revenue'] / df['cumulative_transactions'] / df['active_days']


def compute_daily_stats(df):
    # Compute daily means and std for arms
    daily_stats = df.groupby(["day", "arm"]).agg(
        unique_users=("user_id", "nunique"),
        active_users=("active_user_id", "nunique"),
        mean_conversion=("converted", "mean"),
        std_conversion=("converted", "std"),
        mean_cum_conversion=("cumulative_converted", "mean"),
        std_cum_conversion=("cumulative_converted", "std"),
        mean_dau=("is_active", "mean"),
        std_dau=("is_active", "std"),
        mean_transactions=("transactions", "mean"),
        std_transactions=("transactions", "std"),
        mean_cum_transactions=("cumulative_transactions", "mean"),
        std_cum_transactions=("cumulative_transactions", "std"),
        mean_transactions_per_active_day=("transactions_per_active_day", "mean"),
        std_transactions_per_active_day=("transactions_per_active_day", "std"),
        mean_revenue=("revenue", "mean"),
        std_revenue=("revenue", "std"),
        mean_cum_revenue=("cumulative_revenue", "mean"),
        std_cum_revenue=("cumulative_revenue", "std"),
        mean_revenue_per_transaction=("revenue_per_transaction", "mean"),
        std_revenue_per_transaction=("revenue_per_transaction", "std"),
        mean_revenue_per_transaction_per_active_day=("revenue_per_transaction_per_active_day", "mean"),
        std_revenue_per_transaction_per_active_day=("revenue_per_transaction_per_active_day", "std"),
        mean_revenue_per_active_day=("revenue_per_active_day", "mean"),
        std_revenue_per_active_day=("revenue_per_active_day", "std"),
    ).reset_index()
    return daily_stats

# Compute daily stats
daily_stats = compute_daily_stats(df)
daily_stats['inactivity_rate'] = 1 - daily_stats['active_users'] / daily_stats['unique_users']

customers_daily_stats = compute_daily_stats(df[df['converted'] == True])
customers_daily_stats['inactivity_rate'] = 1 - customers_daily_stats['active_users'] / customers_daily_stats['unique_users']

# Print results
print('-' * 50)
print("Simulation complete.")
print(df.head())
print('\n' * 2)

print('-' * 50)
print("Totals")
print(f"Total users: {len(users)}")
print(f"Total Revenue: {df['revenue'].sum():.2f}")
print(f"Total Transactions: {df['transactions'].sum():.2f}")
print(f"Total Conversions: {df['converted'].sum():.2f}")
print('\n' * 2)

# Stats per arm
def get_stats_per_arm(daily_stats) -> pd.DataFrame:
    stats_per_arm = daily_stats.set_index('day').loc[TEST_DAYS-1].groupby(['arm']).agg(
            num_users_assigned=('unique_users', 'sum'),
            num_active_users=('active_users', 'sum'),
            conversion=('mean_conversion', 'mean'),
            cumulative_conversion=('mean_cum_conversion', 'mean'),
            dau=('mean_dau', 'mean'),
            transactions=('mean_transactions', 'mean'),
            cum_transactions=('mean_cum_transactions', 'mean'),
            transactions_per_active_day=('mean_transactions_per_active_day', 'mean'),
            revenue=('mean_revenue', 'mean'),
            cum_revenue=('mean_cum_revenue', 'mean'),
            revenue_per_active_day=('mean_revenue_per_active_day', 'mean'),
            revenue_per_transaction=('mean_revenue_per_transaction', 'mean'),
            revenue_per_transaction_per_active_day=('mean_revenue_per_transaction_per_active_day', 'mean'),
        ).T
    return stats_per_arm

# Stats per arm
full_pop_stats_per_arm = get_stats_per_arm(daily_stats)
customers_stats_per_arm = get_stats_per_arm(customers_daily_stats)

print('-' * 50)
print("User stats per arm:")
print(full_pop_stats_per_arm)
print('\n' * 2)

print('-' * 50)
print("Customer stats per arm :")
print(customers_stats_per_arm)
print('\n' * 2)

# Save dataset
df.to_csv("ab_test_simulation_data.csv", index=False)

# Save daily stats
daily_stats.to_csv("ab_test_simulation_daily_stats.csv", index=False)


print("Dataset saved.")