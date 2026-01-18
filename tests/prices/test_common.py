import pytest
from kavalai.prices.common import ModelPricing, TokenPricing


def test_calculate_cost_flat():
    pricing = ModelPricing(
        model_name="test-flat",
        input=TokenPricing(price_per_1m=1.0),
        output=TokenPricing(price_per_1m=2.0),
    )
    # 100k input, 50k output
    cost = pricing.calculate_cost(100_000, 50_000)
    # (100k * 1 / 1M) + (50k * 2 / 1M) = 0.1 + 0.1 = 0.2
    assert pytest.approx(cost) == 0.2


def test_calculate_cost_cached():
    pricing = ModelPricing(
        model_name="test-cached",
        input=TokenPricing(price_per_1m=1.0),
        cached_input=TokenPricing(price_per_1m=0.5),
        output=TokenPricing(price_per_1m=2.0),
    )
    # 100k prompt (40k cached), 50k output
    cost = pricing.calculate_cost(100_000, 50_000, cached_tokens=40_000)
    # (60k * 1 / 1M) + (40k * 0.5 / 1M) + (50k * 2 / 1M) = 0.06 + 0.02 + 0.1 = 0.18
    assert pytest.approx(cost) == 0.18


def test_calculate_cost_tiered():
    pricing = ModelPricing(
        model_name="test-tiered",
        input=TokenPricing(tiered={"<=200k": 1.0, ">200k": 2.0}),
        output=TokenPricing(tiered={"<=200k": 5.0, ">200k": 10.0}),
    )

    # Below threshold
    cost_low = pricing.calculate_cost(100_000, 50_000)  # Total 150k
    # (100k * 1 / 1M) + (50k * 5 / 1M) = 0.1 + 0.25 = 0.35
    assert pytest.approx(cost_low) == 0.35

    # Above threshold
    cost_high = pricing.calculate_cost(200_000, 100_000)  # Total 300k
    # (200k * 2 / 1M) + (100k * 10 / 1M) = 0.4 + 1.0 = 1.4
    assert pytest.approx(cost_high) == 1.4
