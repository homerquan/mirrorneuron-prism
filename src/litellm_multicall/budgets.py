class BudgetTracker:
    def __init__(self, max_model_calls: int = 5):
        self.max_model_calls = max_model_calls
        self.used = 0

    def reserve(self):
        if self.used >= self.max_model_calls:
            raise RuntimeError("Budget exhausted")
        self.used += 1

