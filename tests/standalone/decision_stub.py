"""Deterministic model injection for socket tests; no checkpoint download."""


class RulesAgent:
    def system_one(self, state, questions, **kwargs):
        return {
            "answers": {
                "strategy": {
                    "choice": state["rules_strategy"],
                    "answer_confidence": 1.0,
                }
            },
            "usage": {"truncated": False},
        }
