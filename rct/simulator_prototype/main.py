import numpy as np
import pandas as pd
from scipy.stats import halfnorm

# Global Parameters
__SEED__ = 12345  # Seed for reproducibility

POPULATION_SIZE = 100_000  # Total user base
DAILY_INJECTION_RATE = 0.01  # Fraction of population entering daily
TEST_DAYS = 10  # Number of days in the test
K_ARMS = 2  # Number of arms in the A/B test
SPLIT_WEIGHTS = [0.5, 0.5]  # Must sum to 1

assert len(SPLIT_WEIGHTS) == K_ARMS
assert sum(SPLIT_WEIGHTS) == 1

# Control group parameters
TRUE_CONVERSION_RATE = 0.10  # True probability of conversion in control
TRUE_REVENUE_PER_CUSTOMER = 50  # Mean revenue per converted customer
TRUE_REVENUE_STD = 20  # Standard deviation of revenue

# Uplift per arm
CONVERSION_RATE_UPLIFTS = [0.0, 0.1]  # Relative increase in conversion or revenue per arm
REVENUE_PER_CUSTOMER_UPLIFTS = [0.0, 0.1]  # Relative increase in conversion or revenue per arm

# Return rate function: Probability of return as a function of days since last visit
def return_probability(days_since_last):
    return max(0.05, np.exp(-0.3 * days_since_last))  # Example: Exponential decay

def compute_uplift(control_val, uplift_pctg, positive_bounded=True):
    arm_value = control_val * (1 + uplift_pctg)
    if positive_bounded:
        return min(0.0, arm_value)
    else:
        return arm_value


class User:
    def __init__(self, user_id, entry_day):
        self.user_id = user_id
        self.entry_day = entry_day
        self.last_seen = entry_day
        self.converted = False
        self.revenue = 0.0
        self.arm: int | None = None
    
    def assign_arm(self, arm: int):
        self.arm = arm
    
    def attempt_conversion(self, p, r_mean, r_std):
        self.converted = np.random.rand() < p
        if self.converted:
            self.revenue = halfnorm.rvs(scale=r_std) + r_mean

# Compute derived parameters
DAILY_ACTIVE_USERS = int(POPULATION_SIZE * DAILY_INJECTION_RATE)
ARM_ALLOCATION = [int(DAILY_ACTIVE_USERS * weight) for weight in SPLIT_WEIGHTS]

# Compute adjusted values for treatment arms
p_k = [compute_uplift(TRUE_CONVERSION_RATE, uplift) for uplift in CONVERSION_RATE_UPLIFTS]
r_k = [compute_uplift(TRUE_REVENUE_PER_CUSTOMER, uplift) for uplift in REVENUE_PER_CUSTOMER_UPLIFTS]
r_std_k = [TRUE_REVENUE_STD for _ in range(K_ARMS)]


# Initialize simulation state
users = []
current_user_id = 0

def simulate_day(day):

    np.random.seed(__SEED__ + day)

    global current_user_id
    
    # Inject new users
    for _ in range(DAILY_ACTIVE_USERS):
        user = User(current_user_id, day)
        users.append(user)
        current_user_id += 1
    
    # Assign users to test arms
    np.random.shuffle(users)  # Shuffle users before allocation
    start = 0
    for k in range(K_ARMS):
        end = start + ARM_ALLOCATION[k]
        for user in users[start:end]:
            user.assign_arm(k)
        start = end
    
    # Simulate conversions
    for user in users:
        if user.arm is not None:
            user.attempt_conversion(p_k[user.arm], r_k[user.arm], r_std_k[user.arm])
    
    # Return users based on probability
    returning_users = []
    for user in users:
        if day > user.entry_day:  # If user is not new today
            if np.random.rand() < return_probability(day - user.last_seen):
                user.last_seen = day
                returning_users.append(user)
    
    return returning_users

# Run Simulation
data_records = []
for day in range(TEST_DAYS):
    returning_users = simulate_day(day)
    
    # Collect daily data
    for user in users:
        data_records.append([day, user.user_id, user.arm, user.converted, user.revenue])

df = pd.DataFrame(data_records, columns=["day", "user_id", "arm", "converted", "revenue"])

print("Simulation complete.")
print(df.head())
print(f"Total users: {len(users)}")
print(f"Total conversions: {df['converted'].sum()}")
print(f"Total revenue: {df['revenue'].sum()}")
print(f"Conversion rate: {df['converted'].mean()}")
print(f"Revenue per user: {df['revenue'].mean()}")
print(f"Revenue per converted user: {df[df['converted']]['revenue'].mean()}")
print(f"Revenue per user (control): {df[df['arm'] == 0]['revenue'].mean()}")
print(f"Revenue per user (treatment): {df[df['arm'] == 1]['revenue'].mean()}")
print(f"Revenue per converted user (control): {df[(df['converted']) & (df['arm'] == 0)]['revenue'].mean()}")
print(f"Revenue per converted user (treatment): {df[(df['converted']) & (df['arm'] == 1)]['revenue'].mean()}")
print(f"Conversion rate (control): {df[df['arm'] == 0]['converted'].mean()}")
print(f"Conversion rate (treatment): {df[df['arm'] == 1]['converted'].mean()}")
print(f"Conversion rate uplift: {df[df['arm'] == 1]['converted'].mean() - df[df['arm'] == 0]['converted'].mean()}")
print(f"Revenue per user uplift: {df[df['arm'] == 1]['revenue'].mean() - df[df['arm'] == 0]['revenue'].mean()}")
print(f"Revenue per converted user uplift: {df[(df['converted']) & (df['arm'] == 1)]['revenue'].mean() - df[(df['converted']) & (df['arm'] == 0)]['revenue'].mean()}")
print(f"Total revenue uplift: {df[df['arm'] == 1]['revenue'].sum() - df[df['arm'] == 0]['revenue'].sum()}")
print(f"Total revenue uplift (normalized): {df[df['arm'] == 1]['revenue'].sum() / df[df['arm'] == 0]['revenue'].sum() - 1}")
print("Saving dataset as 'ab_test_simulation_data.csv'...")
df.to_csv("ab_test_simulation_data.csv", index=False)
print("Dataset saved.")

