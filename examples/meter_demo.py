from litellm_multicall.meter import RequestMeter, get_price_estimate

meter = RequestMeter(logical_request_id="req_123")
meter.record("local-small", prompt_tokens=120, completion_tokens=80)
meter.record("local-small", prompt_tokens=130, completion_tokens=90)
meter.record("local-strong", prompt_tokens=200, completion_tokens=150)

agg = meter.aggregate()
print(agg)
estimates = get_price_estimate(meter)
# Pricing is not wired yet (Step 01 baseline): estimates are unknown (None),
# never silent zeros. Format accordingly.
formatted = {k: (f"${v:.6f}" if v is not None else "unknown") for k, v in estimates.items()}
print(formatted)
