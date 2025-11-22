import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import factorial, exp
from mpl_toolkits.mplot3d import Axes3D

class GbikeRental:
    def __init__(self):
        self.max_bikes = 20
        self.max_move = 5
        self.move_cost = 2
        self.rental_reward = 10
        self.discount = 0.9
        
        # Poisson parameters
        self.lambda_req1 = 3
        self.lambda_req2 = 4
        self.lambda_ret1 = 3
        self.lambda_ret2 = 2
        
        # Precompute Poisson probabilities to save time
        self.poisson_cache = {}
        self.poisson_upper_bound = 11  # Sufficiently large for lambda <= 4
        
        # State space: (bikes_at_1, bikes_at_2)
        self.states = [(i, j) for i in range(self.max_bikes + 1) for j in range(self.max_bikes + 1)]
        
        # Initialize Value function and Policy
        self.V = np.zeros((self.max_bikes + 1, self.max_bikes + 1))
        self.policy = np.zeros((self.max_bikes + 1, self.max_bikes + 1), dtype=int)
        
        self.actions = np.arange(-self.max_move, self.max_move + 1)

    def poisson(self, n, lam):
        key = (n, lam)
        if key not in self.poisson_cache:
            self.poisson_cache[key] = (lam**n / factorial(n)) * exp(-lam)
        return self.poisson_cache[key]

    def expected_return(self, state, action):
        # Apply action (move bikes)
        # Action > 0: Move from 1 to 2
        # Action < 0: Move from 2 to 1
        
        # Cost of moving
        cost = abs(action) * self.move_cost
        
        # Bikes available after move (clamped to max capacity? No, just clamped to 0 and max if we assume we can't move more than we have or more than fits? 
        # Problem says: "You may move a maximum of 5 bikes". 
        # Implicitly, can't move more than you have.
        # And "No more than 20 bikes can be parked". This usually applies to end of day.
        # But if we move bikes there, do they fit? Assuming they fit or overflow is lost immediately?
        # Sutton & Barto implementation usually assumes we just cap at 20.
        
        b1 = int(state[0] - action)
        b2 = int(state[1] + action)
        
        # Check validity of action (can't have negative bikes)
        if b1 < 0 or b2 < 0:
            return -float('inf') # Invalid move
            
        # Cap at 20? "No more than 20 bikes can be parked". 
        # Usually this check is done after returns. But if we move them, they arrive in the morning.
        # Let's assume we clamp to 20 immediately upon arrival if needed, or just let them be there for the day and clamp at night?
        # Standard interpretation: Morning inventory is min(b, 20).
        b1 = min(b1, self.max_bikes)
        b2 = min(b2, self.max_bikes)
        
        expected_reward = -cost
        
        # Iterate over all possible request/return scenarios
        # We truncate the Poisson distribution for performance
        
        for req1 in range(self.poisson_upper_bound):
            for req2 in range(self.poisson_upper_bound):
                prob_req = self.poisson(req1, self.lambda_req1) * self.poisson(req2, self.lambda_req2)
                
                # Valid rentals
                valid_rentals1 = min(b1, req1)
                valid_rentals2 = min(b2, req2)
                
                reward = (valid_rentals1 + valid_rentals2) * self.rental_reward
                
                # Bikes remaining after rentals
                rem_b1 = b1 - valid_rentals1
                rem_b2 = b2 - valid_rentals2
                
                # Now consider returns
                # We can average the returns or iterate. Iterating 4 loops is slow (11^4 = 14641 iterations per state-action).
                # Optimization: The returns are independent of requests for the *next* state calculation, 
                # BUT the reward depends on requests.
                # Actually, the next state depends on returns.
                # Expected next value = Sum(Prob(req, ret) * (Reward + gamma * V(next_s)))
                # = Expected_Reward + gamma * Expected_Next_V
                
                # Expected Reward is easy:
                # E[R] = E[min(b1, req1)] * 10 + E[min(b2, req2)] * 10 - cost
                # We can compute this separately.
                
                # Expected Next V:
                # We need distribution of (rem_b1 + ret1, rem_b2 + ret2)
                
                # Let's do the full loop but optimized? Or just 4 loops?
                # 21*21 states = 441. 11 actions = ~5000 state-actions.
                # 11^4 inner loop is too big (~14k). Total ops ~ 70 million. Feasible in Python? Maybe slow.
                # Optimization: The transitions for loc 1 and loc 2 are independent.
                # P(s1' | s1) and P(s2' | s2).
                # We can compute transition matrices for each location.
                pass

        # Re-implementation with separated expectations for speed
        
        # 1. Expected Reward
        # R(s, a) = -cost + 10 * (E[rentals1] + E[rentals2])
        # E[rentals1] = sum_k P(req1=k) * min(b1, k)
        expected_rentals1 = sum(self.poisson(k, self.lambda_req1) * min(b1, k) for k in range(self.poisson_upper_bound))
        expected_rentals2 = sum(self.poisson(k, self.lambda_req2) * min(b2, k) for k in range(self.poisson_upper_bound))
        total_reward = -cost + self.rental_reward * (expected_rentals1 + expected_rentals2)
        
        # 2. Expected Next Value
        # Next state s1' = min(20, max(0, b1 - req1 + ret1))
        # We need P(s1' | b1)
        # This can be precomputed or computed on the fly.
        # Since b1 is fixed for this (s, a), we can compute P(s1_next | b1).
        
        # Let's compute P(next_b | current_b, lambda_req, lambda_ret)
        # This is constant for the problem? No, depends on current_b.
        
        prob_next_s1 = np.zeros(self.max_bikes + 1)
        for req in range(self.poisson_upper_bound):
            p_req = self.poisson(req, self.lambda_req1)
            rentals = min(b1, req)
            rem = b1 - rentals
            for ret in range(self.poisson_upper_bound):
                p_ret = self.poisson(ret, self.lambda_ret1)
                next_b = min(self.max_bikes, rem + ret)
                prob_next_s1[next_b] += p_req * p_ret
                
        prob_next_s2 = np.zeros(self.max_bikes + 1)
        for req in range(self.poisson_upper_bound):
            p_req = self.poisson(req, self.lambda_req2)
            rentals = min(b2, req)
            rem = b2 - rentals
            for ret in range(self.poisson_upper_bound):
                p_ret = self.poisson(ret, self.lambda_ret2)
                next_b = min(self.max_bikes, rem + ret)
                prob_next_s2[next_b] += p_req * p_ret
        
        # Now compute expected V
        # E[V] = sum_{s1', s2'} P(s1') * P(s2') * V[s1', s2']
        # This is a matrix multiplication or convolution?
        # V is 2D. P1 is 1D, P2 is 1D.
        # E[V] = P1 . V . P2^T ?
        # Let's check dimensions.
        # V is (21, 21). P1 is (21,). P2 is (21,).
        # sum_i sum_j P1[i] * P2[j] * V[i, j]
        # = sum_i P1[i] * (sum_j P2[j] * V[i, j])
        # = P1 dot (V dot P2)
        
        expected_v = np.dot(prob_next_s1, np.dot(self.V, prob_next_s2))
        
        return total_reward + self.discount * expected_v

    def policy_evaluation(self, epsilon=1e-4):
        print("  Policy Evaluation...")
        while True:
            delta = 0
            # Iterate over all states
            # For speed, we could vectorize, but the logic is complex. 
            # Let's stick to loops for clarity and correctness first.
            for i in range(self.max_bikes + 1):
                for j in range(self.max_bikes + 1):
                    v = self.V[i, j]
                    action = self.policy[i, j]
                    self.V[i, j] = self.expected_return((i, j), action)
                    delta = max(delta, abs(v - self.V[i, j]))
            
            if delta < epsilon:
                break

    def policy_improvement(self):
        print("  Policy Improvement...")
        policy_stable = True
        for i in range(self.max_bikes + 1):
            for j in range(self.max_bikes + 1):
                old_action = self.policy[i, j]
                
                # Find best action
                best_action = old_action
                max_val = -float('inf')
                
                # Valid actions: must not move more than we have?
                # "You may move a maximum of 5 bikes"
                # Also can't move more than i (if moving 1->2) or j (if moving 2->1)
                # Actually, the problem says "If you are out of bikes... business is lost".
                # It doesn't explicitly say you can't move bikes you don't have, but it's physically impossible.
                # So action range is constrained by state.
                
                min_action = max(-self.max_move, -j) # Move from 2 to 1 (negative action)
                max_action_val = min(self.max_move, i) # Move from 1 to 2 (positive action)
                
                # Actually, let's just check all -5 to 5 and handle invalid in expected_return
                # But optimizing the range here saves time.
                
                for action in range(min_action, max_action_val + 1):
                    val = self.expected_return((i, j), action)
                    if val > max_val:
                        max_val = val
                        best_action = action
                
                self.policy[i, j] = best_action
                if old_action != best_action:
                    policy_stable = False
        return policy_stable

    def solve(self):
        iterations = 0
        while True:
            print(f"Iteration {iterations}")
            self.policy_evaluation()
            stable = self.policy_improvement()
            if stable:
                print("Policy stable.")
                break
            iterations += 1
        return self.V, self.policy

def plot_results(V, policy, filename_prefix="gbike_original"):
    # 1. Policy Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(policy, annot=True, fmt="d", cmap="RdBu_r", cbar=True,
                linewidths=0.5, linecolor='gray', square=True,
                cbar_kws={'label': 'Net Bikes Moved (1→2)'}, center=0,
                vmin=-5, vmax=5)
    ax.invert_yaxis()
    plt.title("Optimal Policy: Net Bike Transfers (Location 1 → Location 2)", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Bikes at Location 2 (End of Day)", fontsize=12, fontweight='bold')
    plt.ylabel("Bikes at Location 1 (End of Day)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_policy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved policy heatmap to {filename_prefix}_policy.png")

    # 2. Value Function Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(V, annot=False, cmap="viridis", cbar=True,
                linewidths=0, square=True,
                cbar_kws={'label': 'Expected Return'})
    ax.invert_yaxis()
    plt.title("Value Function: Expected Return", 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel("Bikes at Location 2 (End of Day)", fontsize=12, fontweight='bold')
    plt.ylabel("Bikes at Location 1 (End of Day)", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_value_heatmap.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved value heatmap to {filename_prefix}_value_heatmap.png")

    # 3. Value Function 3D Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    x = np.arange(V.shape[1])
    y = np.arange(V.shape[0])
    X, Y = np.meshgrid(x, y)
    
    surf = ax.plot_surface(X, Y, V, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Bikes at Location 2', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel('Bikes at Location 1', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_zlabel('Expected Return', fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title('Value Function: 3D Surface', fontsize=14, fontweight='bold', pad=20)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='Expected Return')
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.savefig(f"{filename_prefix}_value_3d.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved value 3D plot to {filename_prefix}_value_3d.png")

if __name__ == "__main__":
    print("Solving Gbike Rental Problem (Original)...")
    gbike = GbikeRental()
    V, policy = gbike.solve()
    plot_results(V, policy)
