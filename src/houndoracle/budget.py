"""Reserve before requests. Local per-query caps; no claim of cloud-wide hard limits."""

import math
import threading
import time


class BudgetExceeded(Exception):
    pass


class Budget:
    def __init__(self, max_requests, max_seconds, max_cost_usd):
        self.max_requests, self.max_cost = max_requests, max_cost_usd
        self.deadline = time.monotonic() + max_seconds
        self.requests, self.cost = 0, 0.0
        self.lock = threading.Lock()

    def reserve(self, cost):
        with self.lock:
            self.check_time()
            if not math.isfinite(cost) or cost < 0:
                raise ValueError("A finite nonnegative provider upper bound is required")
            if self.requests >= self.max_requests or self.cost + cost > self.max_cost + 1e-12:
                raise BudgetExceeded("BUDGET_EXHAUSTED")
            self.requests += 1
            self.cost += cost

    def check_time(self):
        if time.monotonic() >= self.deadline:
            raise BudgetExceeded("TIME_EXHAUSTED")

    def timeout(self):
        self.check_time()
        return max(0.001, self.deadline - time.monotonic())
