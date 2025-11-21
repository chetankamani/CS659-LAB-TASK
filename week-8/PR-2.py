#!/usr/bin/env python3
"""
PROBLEM 3: Gbike Bicycle Rental - MODIFIED Version with Policy Iteration

This code solves the MODIFIED Gbike bicycle rental problem with:
1. FREE shuttle: First bike from location 1 → location 2 is FREE
2. Parking cost: INR 4 if more than 10 bikes at a location overnight

Author: Lab Assignment 8
Course: Reinforcement Learning
"""

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import time

class GbikeRentalModifiedMDP:
    """
    Modified Gbike Bicycle Rental Problem - MDP Formulation

    MODIFICATIONS from base problem:
    ---------------------------------
    1. FREE SHUTTLE: 
       - An employee rides bus from loc1 to loc2 each night
       - She shuttles ONE bike for FREE
       - Movement cost for loc1→loc2: 0 for 1st bike, INR 2 for each additional
       - Movement cost for loc2→loc1: Still INR 2 per bike (no free shuttle)

    2. PARKING COST:
       - Limited parking space at each location
       - If > 10 bikes at a location overnight (after moving): INR 4 cost
       - This cost is INDEPENDENT of how many bikes (11 or 20, still INR 4)
       - Applied to EACH location separately

    MDP Components:
    ---------------
    State Space (S): 
        - (n1, n2) where n1, n2 ∈ [0, 20]
        - n1 = bikes at location 1 at end of day
        - n2 = bikes at location 2 at end of day
        - Total states: 21 × 21 = 441

    Action Space (A):
        - a ∈ [-5, 5] (net bikes moved from loc1 to loc2)
        - Positive: move from loc1 to loc2
        - Negative: move from loc2 to loc1

    Transition Function P(s'|s,a):
        - Same as base problem
        - Poisson-distributed rental requests: λ1=3 (loc1), λ2=4 (loc2)
        - Poisson-distributed returns: λ1=3 (loc1), λ2=2 (loc2)

    Reward Function R(s,a) - MODIFIED:
        - Rental revenue: 10 × (rentals at loc1 + rentals at loc2)
        - Movement cost: 
            * If a > 0: 2 × (a-1)  [first bike free]
            * If a ≤ 0: 2 × |a|    [no free shuttle in reverse]
        - Parking cost:
            * -4 if bikes_loc1_after_move > 10
            * -4 if bikes_loc2_after_move > 10

    Discount Factor (γ):
        - γ = 0.9 (continuing task)
    """

    def __init__(self):
        """Initialize MDP parameters"""
        # Environment parameters
        self.MAX_BIKES = 20          # Maximum bikes at each location
        self.MAX_MOVE = 5            # Maximum bikes that can be moved
        self.RENTAL_REWARD = 10      # INR earned per rental
        self.MOVE_COST = 2           # INR cost per bike moved
        self.PARKING_COST = 4        # INR cost for using second parking lot
        self.PARKING_THRESHOLD = 10  # Threshold for parking cost
        self.GAMMA = 0.9             # Discount factor

        # Poisson distribution parameters
        self.RENTAL_REQUEST_LOC1 = 3  # Expected rental requests at location 1
        self.RENTAL_REQUEST_LOC2 = 4  # Expected rental requests at location 2
        self.RETURN_LOC1 = 3          # Expected returns at location 1
        self.RETURN_LOC2 = 2          # Expected returns at location 2

        # Computational efficiency parameter
        self.POISSON_UPPER_BOUND = 11

        # Precompute Poisson probabilities
        print("Precomputing Poisson probabilities...")
        self.poisson_cache = self._precompute_poisson()
        print("  Done! Cached probabilities for λ ∈ {2, 3, 4}")

    def _precompute_poisson(self):
        """Precompute Poisson probabilities for efficiency."""
        cache = {}
        lambda_values = [
            self.RENTAL_REQUEST_LOC1, 
            self.RENTAL_REQUEST_LOC2,
            self.RETURN_LOC1, 
            self.RETURN_LOC2
        ]

        for lam in lambda_values:
            for n in range(self.POISSON_UPPER_BOUND):
                cache[(n, lam)] = poisson.pmf(n, lam)

        return cache

    def get_poisson_prob(self, n, lam):
        """Get Poisson probability P(X = n) where X ~ Poisson(λ)."""
        return self.poisson_cache.get((n, lam), 0.0)

    def expected_return(self, state, action, V):
        """
        Calculate expected return for taking action in state - MODIFIED VERSION.

        MODIFICATIONS:
        1. Free shuttle: First bike from loc1→loc2 is free
        2. Parking cost: INR 4 if > 10 bikes at a location

        Args:
            state: tuple (bikes_loc1, bikes_loc2)
            action: int, bikes to move from loc1 to loc2 (can be negative)
            V: numpy array [21×21], current value function

        Returns:
            float: expected return
        """
        expected_value = 0.0
        bikes_loc1, bikes_loc2 = state

        # Check if action is valid
        if action > bikes_loc1 or -action > bikes_loc2:
            return -np.inf  # Invalid action

        # ──────────────────────────────────────────────────────────────────
        # MODIFICATION 1: Free shuttle for first bike from loc1 to loc2
        # ──────────────────────────────────────────────────────────────────
        if action > 0:
            # Moving from loc1 to loc2: First bike is FREE!
            # Cost = 2 × (action - 1)
            # Examples: action=1 → cost=0, action=2 → cost=2, action=5 → cost=8
            cost_of_moving = self.MOVE_COST * max(0, action - 1)
        else:
            # Moving from loc2 to loc1 (or not moving): Standard cost
            # Cost = 2 × |action|
            cost_of_moving = self.MOVE_COST * abs(action)

        expected_value -= cost_of_moving

        # State after moving bikes overnight
        bikes_loc1_after_move = min(bikes_loc1 - action, self.MAX_BIKES)
        bikes_loc2_after_move = min(bikes_loc2 + action, self.MAX_BIKES)

        # Ensure non-negative bikes
        bikes_loc1_after_move = max(0, bikes_loc1_after_move)
        bikes_loc2_after_move = max(0, bikes_loc2_after_move)

        # ──────────────────────────────────────────────────────────────────
        # MODIFICATION 2: Parking costs (assessed overnight, before rentals)
        # ──────────────────────────────────────────────────────────────────
        # If more than 10 bikes at location 1, pay INR 4 for extra parking
        if bikes_loc1_after_move > self.PARKING_THRESHOLD:
            expected_value -= self.PARKING_COST

        # If more than 10 bikes at location 2, pay INR 4 for extra parking
        if bikes_loc2_after_move > self.PARKING_THRESHOLD:
            expected_value -= self.PARKING_COST

        # ──────────────────────────────────────────────────────────────────
        # Rest of the dynamics are UNCHANGED from base problem
        # ──────────────────────────────────────────────────────────────────

        # Iterate over all possible rental requests (Poisson distributed)
        for rental_request_loc1 in range(self.POISSON_UPPER_BOUND):
            for rental_request_loc2 in range(self.POISSON_UPPER_BOUND):

                # Probability of this rental request combination
                prob_rental_requests = (
                    self.get_poisson_prob(rental_request_loc1, self.RENTAL_REQUEST_LOC1) *
                    self.get_poisson_prob(rental_request_loc2, self.RENTAL_REQUEST_LOC2)
                )

                # Skip negligible probabilities for efficiency
                if prob_rental_requests < 1e-10:
                    continue

                # Actual rentals = min(available bikes, requested bikes)
                actual_rentals_loc1 = min(bikes_loc1_after_move, rental_request_loc1)
                actual_rentals_loc2 = min(bikes_loc2_after_move, rental_request_loc2)

                # Reward from rentals (UNCHANGED)
                rental_reward = (actual_rentals_loc1 + actual_rentals_loc2) * self.RENTAL_REWARD

                # Bikes remaining after rentals
                bikes_after_rental_loc1 = bikes_loc1_after_move - actual_rentals_loc1
                bikes_after_rental_loc2 = bikes_loc2_after_move - actual_rentals_loc2

                # Iterate over all possible returns (Poisson distributed)
                for returns_loc1 in range(self.POISSON_UPPER_BOUND):
                    for returns_loc2 in range(self.POISSON_UPPER_BOUND):

                        # Probability of this return combination
                        prob_returns = (
                            self.get_poisson_prob(returns_loc1, self.RETURN_LOC1) *
                            self.get_poisson_prob(returns_loc2, self.RETURN_LOC2)
                        )

                        # Skip negligible probabilities
                        if prob_returns < 1e-10:
                            continue

                        # Combined probability of this trajectory
                        prob = prob_rental_requests * prob_returns

                        # Final state at end of day (after returns, capped at MAX_BIKES)
                        final_bikes_loc1 = min(bikes_after_rental_loc1 + returns_loc1, 
                                              self.MAX_BIKES)
                        final_bikes_loc2 = min(bikes_after_rental_loc2 + returns_loc2, 
                                              self.MAX_BIKES)

                        # Bellman equation: immediate reward + discounted future value
                        expected_value += prob * (
                            rental_reward + 
                            self.GAMMA * V[final_bikes_loc1, final_bikes_loc2]
                        )

        return expected_value

    def policy_evaluation(self, policy, V, theta=0.1):
        """Policy Evaluation: Iteratively evaluate value function for given policy."""
        iterations = 0

        while True:
            delta = 0

            for bikes_loc1 in range(self.MAX_BIKES + 1):
                for bikes_loc2 in range(self.MAX_BIKES + 1):
                    v_old = V[bikes_loc1, bikes_loc2]
                    action = policy[bikes_loc1, bikes_loc2]
                    V[bikes_loc1, bikes_loc2] = self.expected_return(
                        (bikes_loc1, bikes_loc2), action, V
                    )
                    delta = max(delta, abs(v_old - V[bikes_loc1, bikes_loc2]))

            iterations += 1
            if delta < theta:
                break

        return V, iterations

    def policy_improvement(self, V, policy):
        """Policy Improvement: Update policy to be greedy with respect to V."""
        policy_stable = True

        for bikes_loc1 in range(self.MAX_BIKES + 1):
            for bikes_loc2 in range(self.MAX_BIKES + 1):
                old_action = policy[bikes_loc1, bikes_loc2]

                # Evaluate all possible actions
                action_values = []
                for action in range(-self.MAX_MOVE, self.MAX_MOVE + 1):

                    if (0 <= bikes_loc1 - action <= self.MAX_BIKES and 
                        0 <= bikes_loc2 + action <= self.MAX_BIKES):

                        q_value = self.expected_return(
                            (bikes_loc1, bikes_loc2), action, V
                        )
                        action_values.append((q_value, action))

                # Choose action with maximum Q-value
                if action_values:
                    best_q_value, best_action = max(action_values, key=lambda x: x[0])
                    policy[bikes_loc1, bikes_loc2] = best_action

                    if old_action != best_action:
                        policy_stable = False

        return policy, policy_stable

    def policy_iteration(self, max_iterations=20):
        """Policy Iteration Algorithm."""
        print("\n" + "="*70)
        print("STARTING POLICY ITERATION (MODIFIED PROBLEM)")
        print("="*70)

        V = np.zeros((self.MAX_BIKES + 1, self.MAX_BIKES + 1))
        policy = np.zeros((self.MAX_BIKES + 1, self.MAX_BIKES + 1), dtype=int)

        start_time = time.time()

        for iteration in range(max_iterations):
            print(f"\n{'─'*70}")
            print(f"Policy Iteration {iteration + 1}/{max_iterations}")
            print('─'*70)

            # Policy Evaluation
            print("Step 1: Evaluating policy...")
            eval_start = time.time()
            V, eval_iterations = self.policy_evaluation(policy, V)
            eval_time = time.time() - eval_start
            print(f"  ✓ Converged in {eval_iterations} iterations ({eval_time:.1f}s)")

            # Policy Improvement
            print("Step 2: Improving policy...")
            improve_start = time.time()
            policy, policy_stable = self.policy_improvement(V, policy)
            improve_time = time.time() - improve_start
            print(f"  ✓ Policy improvement complete ({improve_time:.1f}s)")
            print(f"  Policy stable: {policy_stable}")

            if policy_stable:
                elapsed_time = time.time() - start_time
                print(f"\n{'='*70}")
                print(f"✓ POLICY CONVERGED after {iteration + 1} iterations!")
                print(f"  Total time: {elapsed_time:.1f} seconds")
                print('='*70)
                break

        return policy, V

    def print_policy_summary(self, policy):
        """Print summary of the optimal policy."""
        print("\n" + "="*70)
        print("OPTIMAL POLICY SUMMARY (MODIFIED PROBLEM)")
        print("="*70)
        print("\nSample optimal actions (bikes to move from loc1 → loc2):")
        print("-"*70)

        sample_states = [
            (0, 0), (5, 5), (10, 10), (11, 11), (15, 5), (5, 15), 
            (20, 0), (0, 20), (20, 20), (10, 5), (5, 10), (12, 8)
        ]

        for state in sample_states:
            action = policy[state[0], state[1]]

            # Calculate cost breakdown for this action
            if action > 0:
                move_cost = self.MOVE_COST * max(0, action - 1)
                cost_note = f"(cost: INR {move_cost}, free shuttle used!)" if action > 0 else ""
            else:
                move_cost = self.MOVE_COST * abs(action)
                cost_note = f"(cost: INR {move_cost})" if action != 0 else ""

            # Check parking cost
            loc1_after = state[0] - action
            loc2_after = state[1] + action
            parking_note = ""
            if loc1_after > 10 or loc2_after > 10:
                parking_note = " ⚠️ Parking cost applies"

            direction = "→" if action >= 0 else "←"
            print(f"  State {state}: Move {abs(action)} bikes (loc1 {direction} loc2) {cost_note}{parking_note}")

        print("\n" + "-"*70)
        print("Legend:")
        print("  → : Move from location 1 to location 2")
        print("  ← : Move from location 2 to location 1")
        print("  ⚠️ : Parking cost of INR 4 will be incurred")

    def save_policy(self, policy, filename='gbike_modified_policy.npy'):
        """Save policy to file."""
        np.save(filename, policy)
        print(f"\n✓ Policy saved to: {filename}")

    def plot_policy(self, policy, filename='gbike_modified_policy.png'):
        """Plot policy as a heatmap."""
        plt.figure(figsize=(12, 9))

        sns.heatmap(
            policy, 
            annot=False,
            cmap='RdBu',
            center=0,
            cbar_kws={'label': 'Number of bikes moved (loc1 → loc2)'},
            xticklabels=range(0, 21, 2),
            yticklabels=range(0, 21, 2)
        )

        plt.xlabel('Bikes at Location 2 (end of day)', fontsize=12, fontweight='bold')
        plt.ylabel('Bikes at Location 1 (end of day)', fontsize=12, fontweight='bold')
        plt.title('Optimal Policy: Modified Gbike Rental (Free Shuttle + Parking Cost)', 
                 fontsize=13, fontweight='bold', pad=20)

        # Add modification notes
        plt.text(0.5, -0.15, 
                'MODIFICATIONS: Free shuttle (1st bike loc1→loc2) | Parking cost (>10 bikes)',
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=10, style='italic', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Policy visualization saved to: {filename}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("GBIKE BICYCLE RENTAL - MODIFIED PROBLEM SOLUTION")
    print("="*70)
    print("\nBase Problem Parameters:")
    print("  • Max bikes per location: 20")
    print("  • Max bikes moved per night: 5")
    print("  • Rental reward: INR 10 per bike")
    print("  • Base movement cost: INR 2 per bike")
    print("  • Discount factor (γ): 0.9")
    print("\nPoisson Parameters:")
    print("  • Rental requests: λ₁ = 3 (loc1), λ₂ = 4 (loc2)")
    print("  • Returns: λ₁ = 3 (loc1), λ₂ = 2 (loc2)")
    print("\n" + "─"*70)
    print("MODIFICATIONS:")
    print("─"*70)
    print("  1. FREE SHUTTLE:")
    print("     - Employee shuttles ONE bike from loc1 → loc2 for FREE")
    print("     - Movement cost for loc1→loc2: 0 for 1st bike, INR 2 for each extra")
    print("     - Movement cost for loc2→loc1: Still INR 2 per bike")
    print("\n  2. PARKING COST:")
    print("     - If > 10 bikes at a location overnight: INR 4 extra cost")
    print("     - Applied independently to each location")
    print("     - Cost is flat INR 4 regardless of exact number (11 or 20)")
    print("="*70)

    # Create modified MDP instance
    mdp = GbikeRentalModifiedMDP()

    # Run policy iteration
    policy, V = mdp.policy_iteration(max_iterations=20)

    # Display results
    mdp.print_policy_summary(policy)

    # Save results
    mdp.save_policy(policy, 'gbike_modified_optimal_policy.npy')
    mdp.plot_policy(policy, 'gbike_modified_optimal_policy.png')

    print("\n" + "="*70)
    print("✓ SOLUTION COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print("  1. gbike_modified_optimal_policy.npy - NumPy array of optimal policy")
    print("  2. gbike_modified_optimal_policy.png - Visualization of optimal policy")
    print("\nTo load the policy later:")
    print("  policy = np.load('gbike_modified_optimal_policy.npy')")
    print("\nExpected Differences from Base Problem:")
    print("  • More aggressive moving from loc1→loc2 (free shuttle incentive)")
    print("  • Avoidance of keeping >10 bikes when possible (parking cost)")
    print("  • Asymmetric policy (cheaper to move loc1→loc2 than reverse)")
    print("="*70)


if __name__ == "__main__":
    main()
