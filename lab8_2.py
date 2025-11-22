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
        
        self.lambda_req1 = 3
        self.lambda_req2 = 4
        self.lambda_ret1 = 3
        self.lambda_ret2 = 2
        
        self.poisson_cache = {}
        self.poisson_upper_bound = 11
        
        self.states = [(i, j) for i in range(self.max_bikes + 1) for j in range(self.max_bikes + 1)]
        
        self.V = np.zeros((self.max_bikes + 1, self.max_bikes + 1))
        self.policy = np.zeros((self.max_bikes + 1, self.max_bikes + 1), dtype=int)
        
        self.actions = np.arange(-self.max_move, self.max_move + 1)

    def poisson(self, n, lam):
        key = (n, lam)
        if key not in self.poisson_cache:
            self.poisson_cache[key] = (lam**n / factorial(n)) * exp(-lam)
        return self.poisson_cache[key]

    def expected_return(self, state, action):
        
        cost = abs(action) * self.move_cost
        
        
        b1 = int(state[0] - action)
        b2 = int(state[1] + action)
        
        if b1 < 0 or b2 < 0:
            return -float('inf')
            
        b1 = min(b1, self.max_bikes)
        b2 = min(b2, self.max_bikes)
        
        expected_reward = -cost
        
        
        for req1 in range(self.poisson_upper_bound):
            for req2 in range(self.poisson_upper_bound):
                prob_req = self.poisson(req1, self.lambda_req1) * self.poisson(req2, self.lambda_req2)
                
                valid_rentals1 = min(b1, req1)
                valid_rentals2 = min(b2, req2)
                
                reward = (valid_rentals1 + valid_rentals2) * self.rental_reward
                
                rem_b1 = b1 - valid_rentals1
                rem_b2 = b2 - valid_rentals2
                
                
                
                
                pass

        
        expected_rentals1 = sum(self.poisson(k, self.lambda_req1) * min(b1, k) for k in range(self.poisson_upper_bound))
        expected_rentals2 = sum(self.poisson(k, self.lambda_req2) * min(b2, k) for k in range(self.poisson_upper_bound))
        total_reward = -cost + self.rental_reward * (expected_rentals1 + expected_rentals2)
        
        
        
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
        
        
        expected_v = np.dot(prob_next_s1, np.dot(self.V, prob_next_s2))
        
        return total_reward + self.discount * expected_v

    def policy_evaluation(self, epsilon=1e-4):
        print("  Policy Evaluation...")
        while True:
            delta = 0
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
                
                best_action = old_action
                max_val = -float('inf')
                
                
                min_action = max(-self.max_move, -j)
                max_action_val = min(self.max_move, i)
                
                
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
