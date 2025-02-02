

import numpy as np
from scipy.stats import lognorm


class UserDailyMetrics:
    def __init__(self, day):
        self.day: int = day
        self.activity_status: bool = True
        self.converted: bool = False
        self.transactions: int = 0
        self.revenue: float = 0.0


class User:
    def __init__(self, user_id, entry_day):
        # persistent variables
        self.user_id: int = user_id
        self.entry_day: int = entry_day
        self.arm: int | None = None
        self.days_inactive: int = 0
        self.active_days: int = 0
        self.inactive_days: int = 0
        self.converted: bool = False
        self.transactions: int = 0
        self.revenue: float = 0.0
        # daily metrics
        self.daily_metrics = UserDailyMetrics(entry_day)

    def update_activity(self, day: int, activity_status: bool):
        if activity_status:
            self.days_inactive = 0
            self.active_days += 1
            self.daily_metrics.activity_status = True
        else:
            self.days_inactive += 1
            self.inactive_days += 1
            self.daily_metrics.activity_status = False

    def assign_arm(self, arm: int):
        self.arm = arm

    def attempt_conversion(self, p, r_mean, r_std):
        self.daily_metrics.converted = np.random.rand() < p
        if self.daily_metrics.converted:
            self.daily_metrics.transactions = 1
            # Compute lognormal parameters
            sigma = np.sqrt(np.log(1 + (r_std / r_mean) ** 2))  # Log-space std dev
            mu = np.log(r_mean) - 0.5 * sigma**2  # Log-space mean

            # Sample from lognormal
            self.daily_metrics.revenue = lognorm.rvs(s=sigma, scale=np.exp(mu))
        else:
            self.revenue = 0.0
            
    def updated_user_history_metrics(self):
        self.revenue += self.daily_metrics.revenue
        self.transactions += 1
        self.converted = self.converted or self.daily_metrics.converted