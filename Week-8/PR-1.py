#!/usr/bin/env python3
"""
PROBLEM 2: Gbike Bicycle Rental - MDP Solution using Policy Iteration

This code formulates and solves the Gbike bicycle rental problem as a 
continuing finite Markov Decision Process (MDP)
"""

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import time

class GbikeRentalMDP:
    """
    Gbike Bicycle Rental Problem - MDP Formulation

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
        - Constraint: must maintain 0 ≤ n1-a, n2+a ≤ 20

    Transition Function P(s'|s,a):
        - Stochastic transitions due to Poisson-distributed:
          * Rental requests: λ1=3 (loc1), λ2=4 (loc2)
          * Returns: λ1=3 (loc1), λ2=2 (loc2)

    Reward Function R(s,a):
        - R = 10 × (rentals at loc1 + rentals at loc2) - 2 × |a|
        - INR 10 per rental
        - INR 2 per bike moved

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
        self.GAMMA = 0.9             # Discount factor

        # Poisson distribution parameters
        self.RENTAL_REQUEST_LOC1 = 3  # Expected rental requests at location 1
        self.RENTAL_REQUEST_LOC2 = 4  # Expected rental requests at location 2
        self.RETURN_LOC1 = 3          # Expected returns at location 1
        self.RETURN_LOC2 = 2          # Expected returns at location 2

        # Computational efficiency parameter
        # We truncate Poisson distributions at this value
        self.POISSON_UPPER_BOUND = 11

        # Precompute Poisson probabilities for efficiency
        print("Precomputing Poisson probabilities...")
        self.poisson_cache = self._precompute_poisson()
        print("  Done! Cached probabilities for λ ∈ {2, 3, 4}")

    def _precompute_poisson(self):
        """
        Precompute Poisson probabilities for efficiency.

        Returns:
            dict: Cache of (n, lambda) -> probability
        """
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
        """
        Get Poisson probability P(X = n) where X ~ Poisson(λ).

        Args:
            n: number of events
            lam: Poisson parameter λ

        Returns:
            float: P(X = n)
        """
        return self.poisson_cache.get((n, lam), 0.0)

    def expected_return(self, state, action, V):
        """
        Calculate expected return for taking action in state.

        This implements the Bellman equation:
        Q(s,a) = Σ P(s'|s,a) [R(s,a,s') + γ V(s')]

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

        # Immediate cost of moving bikes
        cost_of_moving = self.MOVE_COST * abs(action)
        expected_value -= cost_of_moving

        # State after moving bikes overnight
        bikes_loc1_morning = min(bikes_loc1 - action, self.MAX_BIKES)
        bikes_loc2_morning = min(bikes_loc2 + action, self.MAX_BIKES)

        # Ensure non-negative bikes
        bikes_loc1_morning = max(0, bikes_loc1_morning)
        bikes_loc2_morning = max(0, bikes_loc2_morning)

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
                actual_rentals_loc1 = min(bikes_loc1_morning, rental_request_loc1)
                actual_rentals_loc2 = min(bikes_loc2_morning, rental_request_loc2)

                # Reward from rentals
                rental_reward = (actual_rentals_loc1 + actual_rentals_loc2) * self.RENTAL_REWARD

                # Bikes remaining after rentals
                bikes_after_rental_loc1 = bikes_loc1_morning - actual_rentals_loc1
                bikes_after_rental_loc2 = bikes_loc2_morning - actual_rentals_loc2

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
        """
        Policy Evaluation: Iteratively evaluate value function for given policy.

        Solves: V^π(s) = Σ P(s'|s,π(s)) [R(s,π(s),s') + γ V^π(s')]

        Args:
            policy: numpy array [21×21], current policy
            V: numpy array [21×21], value function to update
            theta: convergence threshold

        Returns:
            V: updated value function
            iterations: number of iterations until convergence
        """
        iterations = 0

        while True:
            delta = 0  # Track maximum change in V

            # Update value for each state
            for bikes_loc1 in range(self.MAX_BIKES + 1):
                for bikes_loc2 in range(self.MAX_BIKES + 1):
                    v_old = V[bikes_loc1, bikes_loc2]

                    # Action prescribed by current policy
                    action = policy[bikes_loc1, bikes_loc2]

                    # Update value using Bellman equation
                    V[bikes_loc1, bikes_loc2] = self.expected_return(
                        (bikes_loc1, bikes_loc2), action, V
                    )

                    # Track maximum change
                    delta = max(delta, abs(v_old - V[bikes_loc1, bikes_loc2]))

            iterations += 1

            # Check convergence
            if delta < theta:
                break

        return V, iterations

    def policy_improvement(self, V, policy):
        """
        Policy Improvement: Update policy to be greedy with respect to V.

        Computes: π'(s) = argmax_a Σ P(s'|s,a) [R(s,a,s') + γ V(s')]

        Args:
            V: numpy array [21×21], current value function
            policy: numpy array [21×21], policy to update

        Returns:
            policy: updated policy
            policy_stable: True if policy didn't change
        """
        policy_stable = True

        # For each state, find the best action
        for bikes_loc1 in range(self.MAX_BIKES + 1):
            for bikes_loc2 in range(self.MAX_BIKES + 1):
                old_action = policy[bikes_loc1, bikes_loc2]

                # Evaluate all possible actions
                action_values = []
                for action in range(-self.MAX_MOVE, self.MAX_MOVE + 1):

                    # Check if action is feasible
                    if (0 <= bikes_loc1 - action <= self.MAX_BIKES and 
                        0 <= bikes_loc2 + action <= self.MAX_BIKES):

                        q_value = self.expected_return(
                            (bikes_loc1, bikes_loc2), action, V
                        )
                        action_values.append((q_value, action))

                # Choose action with maximum Q-value (greedy policy)
                if action_values:
                    best_q_value, best_action = max(action_values, key=lambda x: x[0])
                    policy[bikes_loc1, bikes_loc2] = best_action

                    # Check if policy changed
                    if old_action != best_action:
                        policy_stable = False

        return policy, policy_stable

    def policy_iteration(self, max_iterations=20):
        """
        Policy Iteration Algorithm.

        Alternates between:
        1. Policy Evaluation: compute V^π
        2. Policy Improvement: update π to be greedy w.r.t. V^π

        Returns:
            policy: optimal policy [21×21]
            V: optimal value function [21×21]
        """
        print("\n" + "="*70)
        print("STARTING POLICY ITERATION")
        print("="*70)

        # Initialize value function and policy
        V = np.zeros((self.MAX_BIKES + 1, self.MAX_BIKES + 1))
        policy = np.zeros((self.MAX_BIKES + 1, self.MAX_BIKES + 1), dtype=int)

        start_time = time.time()

        for iteration in range(max_iterations):
            print(f"\n{'─'*70}")
            print(f"Policy Iteration {iteration + 1}/{max_iterations}")
            print('─'*70)

            # Step 1: Policy Evaluation
            print("Step 1: Evaluating policy...")
            eval_start = time.time()
            V, eval_iterations = self.policy_evaluation(policy, V)
            eval_time = time.time() - eval_start
            print(f"  ✓ Converged in {eval_iterations} iterations ({eval_time:.1f}s)")

            # Step 2: Policy Improvement
            print("Step 2: Improving policy...")
            improve_start = time.time()
            policy, policy_stable = self.policy_improvement(V, policy)
            improve_time = time.time() - improve_start
            print(f"  ✓ Policy improvement complete ({improve_time:.1f}s)")
            print(f"  Policy stable: {policy_stable}")

            # Check for convergence
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
        print("OPTIMAL POLICY SUMMARY")
        print("="*70)
        print("\nSample optimal actions (bikes to move from loc1 → loc2):")
        print("-"*70)

        sample_states = [
            (0, 0), (5, 5), (10, 10), (15, 5), (5, 15), 
            (20, 0), (0, 20), (20, 20), (10, 5), (5, 10)
        ]

        for state in sample_states:
            action = policy[state[0], state[1]]
            direction = "→" if action >= 0 else "←"
            print(f"  State {state}: Move {abs(action)} bikes (loc1 {direction} loc2)")

    def save_policy(self, policy, filename='gbike_policy.npy'):
        """Save policy to file."""
        np.save(filename, policy)
        print(f"\n✓ Policy saved to: {filename}")

    def plot_policy(self, policy, filename='gbike_policy.png'):
        """
        Plot policy as a heatmap.

        Args:
            policy: optimal policy [21×21]
            filename: output filename
        """
        plt.figure(figsize=(12, 9))

        # Create heatmap
        sns.heatmap(
            policy, 
            annot=False,
            fmt='d',
            cmap='RdBu',
            center=0,
            cbar_kws={'label': 'Number of bikes moved (loc1 → loc2)'},
            xticklabels=range(0, 21, 2),
            yticklabels=range(0, 21, 2)
        )

        plt.xlabel('Bikes at Location 2 (end of day)', fontsize=12, fontweight='bold')
        plt.ylabel('Bikes at Location 1 (end of day)', fontsize=12, fontweight='bold')
        plt.title('Optimal Policy: Gbike Bicycle Rental Problem', 
                 fontsize=14, fontweight='bold', pad=20)

        # Add interpretation text
        plt.text(0.5, -0.15, 
                'Blue = Move from Loc1 to Loc2  |  Red = Move from Loc2 to Loc1  |  White = No movement',
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=10, style='italic')

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Policy visualization saved to: {filename}")


def main():
    """Main execution function."""
    print("\n" + "="*70)
    print("GBIKE BICYCLE RENTAL - MDP SOLUTION")
    print("="*70)
    print("\nProblem Parameters:")
    print("  • Max bikes per location: 20")
    print("  • Max bikes moved per night: 5")
    print("  • Rental reward: INR 10 per bike")
    print("  • Movement cost: INR 2 per bike")
    print("  • Discount factor (γ): 0.9")
    print("\nPoisson Parameters:")
    print("  • Rental requests: λ₁ = 3 (loc1), λ₂ = 4 (loc2)")
    print("  • Returns: λ₁ = 3 (loc1), λ₂ = 2 (loc2)")
    print("\nState Space: 21 × 21 = 441 states")
    print("Action Space: 11 actions (move -5 to +5 bikes)")

    # Create MDP instance
    mdp = GbikeRentalMDP()

    # Run policy iteration
    policy, V = mdp.policy_iteration(max_iterations=20)

    # Display results
    mdp.print_policy_summary(policy)

    # Save results
    mdp.save_policy(policy, 'gbike_optimal_policy.npy')
    mdp.plot_policy(policy, 'gbike_optimal_policy.png')

    print("\n" + "="*70)
    print("✓ SOLUTION COMPLETE!")
    print("="*70)
    print("\nOutputs:")
    print("  1. gbike_optimal_policy.npy - NumPy array of optimal policy")
    print("  2. gbike_optimal_policy.png - Visualization of optimal policy")
    print("\nTo load the policy later:")
    print("  policy = np.load('gbike_optimal_policy.npy')")
    print("="*70)


if __name__ == "__main__":
    main()
