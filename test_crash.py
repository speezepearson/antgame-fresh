#!/usr/bin/env python3
"""Test script to verify the noise crash is fixed."""

import sys
from mechanics import GameState, FoodConfig

def test_food_generation(iterations: int = 100) -> None:
    """Test food generation multiple times to catch intermittent crashes."""
    print(f"Testing food generation {iterations} times...")
    for i in range(iterations):
        # Test with random seed (None)
        try:
            state = GameState(food_config=FoodConfig())
            food_count = len(state.food)
            if (i + 1) % 10 == 0:
                print(f"  Iteration {i + 1}: OK ({food_count} food locations)")
        except Exception as e:
            print(f"  Iteration {i + 1}: FAILED with {type(e).__name__}: {e}")
            sys.exit(1)

    print(f"✓ All {iterations} iterations completed successfully!")

    # Test with specific seeds
    print("\nTesting with specific seed values...")
    test_seeds = [0, 255, 529236, 513444, 989147, 1000000]
    for seed in test_seeds:
        try:
            state = GameState(food_config=FoodConfig(seed=seed))
            food_count = len(state.food)
            print(f"  Seed {seed}: OK ({food_count} food locations)")
        except Exception as e:
            print(f"  Seed {seed}: FAILED with {type(e).__name__}: {e}")
            sys.exit(1)

    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_food_generation()
