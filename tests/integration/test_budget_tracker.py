import pytest
from litellm_multicall.budgets import BudgetTracker

def test_budget_tracker_reserves():
    bt = BudgetTracker(max_model_calls=2)
    bt.reserve()
    bt.reserve()
    with pytest.raises(RuntimeError):
        bt.reserve()
