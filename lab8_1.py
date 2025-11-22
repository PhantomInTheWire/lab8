import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class GridWorld:
    def __init__(self, reward_step):
        self.rows = 3
        self.cols = 4
        self.grid = np.zeros((self.rows, self.cols))
        self.reward_step = reward_step
        self.gamma = 1.0  # Undiscounted as per typical grid world examples unless specified otherwise, usually 1.0 for finite horizon or with terminal states. Problem doesn't specify, assuming 1.0 or close to 1.
        # Actually, for value iteration to converge with negative rewards and no discount, we rely on terminal states.
        # Russell & Norvig usually use gamma=1 for this example or 0.99. Let's use 0.99 to be safe, or 1.0 if guaranteed to terminate.
        # Given the negative rewards, it will try to terminate.
        self.gamma = 1.0 
        
        self.actions = ['U', 'D', 'L', 'R']
        self.terminals = {(0, 3): 1, (1, 3): -1}
        self.walls = {(1, 1)}
        self.start_state = (2, 0)
        
        # Transition probabilities: 0.8 intended, 0.1 left, 0.1 right
        self.transition_probs = {
            'U': {'U': 0.8, 'L': 0.1, 'R': 0.1},
            'D': {'D': 0.8, 'L': 0.1, 'R': 0.1},
            'L': {'L': 0.8, 'U': 0.1, 'D': 0.1},
            'R': {'R': 0.8, 'U': 0.1, 'D': 0.1}
        }

    def is_valid(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols and (r, c) not in self.walls

    def get_next_state(self, r, c, action):
        if action == 'U':
            nr, nc = r - 1, c
        elif action == 'D':
            nr, nc = r + 1, c
        elif action == 'L':
            nr, nc = r, c - 1
        elif action == 'R':
            nr, nc = r, c + 1
        else:
            nr, nc = r, c
            
        if self.is_valid(nr, nc):
            return nr, nc
        else:
            return r, c

    def value_iteration(self, epsilon=1e-4):
        # Initialize utilities
        U = np.zeros((self.rows, self.cols))
        # Set terminal states
        for (r, c), val in self.terminals.items():
            U[r, c] = val
            
        iteration = 0
        while True:
            U_next = U.copy()
            delta = 0
            
            for r in range(self.rows):
                for c in range(self.cols):
                    if (r, c) in self.terminals or (r, c) in self.walls:
                        continue
                    
                    max_val = -float('inf')
                    for action in self.actions:
                        val = 0
                        for actual_move, prob in self.transition_probs[action].items():
                            nr, nc = self.get_next_state(r, c, actual_move)
                            val += prob * U[nr, nc]
                        max_val = max(max_val, val)
                    
                    U_next[r, c] = self.reward_step + self.gamma * max_val
                    delta = max(delta, abs(U_next[r, c] - U[r, c]))
            
            U = U_next
            iteration += 1
            if delta < epsilon * (1 - self.gamma) / self.gamma if self.gamma < 1 else epsilon:
                 break
        
        return U

    def get_optimal_policy(self, U):
        policy = np.full((self.rows, self.cols), '', dtype=object)
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.terminals:
                    policy[r, c] = '*'
                    continue
                if (r, c) in self.walls:
                    policy[r, c] = '#'
                    continue
                
                best_action = None
                max_val = -float('inf')
                
                for action in self.actions:
                    val = 0
                    for actual_move, prob in self.transition_probs[action].items():
                        nr, nc = self.get_next_state(r, c, actual_move)
                        val += prob * U[nr, nc]
                    
                    if val > max_val:
                        max_val = val
                        best_action = action
                
                policy[r, c] = best_action
        return policy

def plot_results(U, policy, reward, title_suffix=""):
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Create a mask for the wall
    mask = np.zeros_like(U, dtype=bool)
    mask[1, 1] = True
    
    # Plot heatmap using seaborn with improved styling
    sns.heatmap(U, annot=False, cmap="RdYlGn", mask=mask, cbar=True,
                linewidths=2, linecolor='black', square=True, 
                cbar_kws={'label': 'Value'}, vmin=U[~mask].min(), vmax=U[~mask].max())
    
    # Add value annotations and terminal state labels
    for r in range(U.shape[0]):
        for c in range(U.shape[1]):
            if mask[r, c]:
                # Wall - fill with gray
                ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=True, color='gray', linewidth=2, edgecolor='black'))
                ax.text(c + 0.5, r + 0.5, 'WALL', ha='center', va='center', 
                       fontweight='bold', fontsize=10, color='white')
                continue
                
            if (r, c) == (0, 3):
                ax.text(c + 0.5, r + 0.2, '+1', ha='center', va='center', 
                       fontweight='bold', fontsize=14, color='darkgreen',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                ax.text(c + 0.5, r + 0.65, f'{U[r, c]:.3f}', ha='center', va='center',
                       fontsize=9, color='black')
                continue
                
            if (r, c) == (1, 3):
                ax.text(c + 0.5, r + 0.2, '-1', ha='center', va='center',
                       fontweight='bold', fontsize=14, color='darkred',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                ax.text(c + 0.5, r + 0.65, f'{U[r, c]:.3f}', ha='center', va='center',
                       fontsize=9, color='black')
                continue
            
            # Value text at top of cell
            ax.text(c + 0.5, r + 0.25, f'{U[r, c]:.3f}', ha='center', va='center',
                   fontsize=10, color='black', fontweight='bold')
            
            # Policy arrow at bottom of cell
            action = policy[r, c]
            arrow_props = dict(arrowstyle='->', lw=2.5, color='blue')
            
            if action == 'U':
                ax.annotate('', xy=(c + 0.5, r + 0.5), xytext=(c + 0.5, r + 0.75),
                          arrowprops=arrow_props)
            elif action == 'D':
                ax.annotate('', xy=(c + 0.5, r + 0.75), xytext=(c + 0.5, r + 0.5),
                          arrowprops=arrow_props)
            elif action == 'L':
                ax.annotate('', xy=(c + 0.25, r + 0.65), xytext=(c + 0.5, r + 0.65),
                          arrowprops=arrow_props)
            elif action == 'R':
                ax.annotate('', xy=(c + 0.75, r + 0.65), xytext=(c + 0.5, r + 0.65),
                          arrowprops=arrow_props)

    plt.title(f"Grid World: Value Function & Optimal Policy (r(s) = {reward})", 
             fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Column', fontsize=12, fontweight='bold')
    plt.ylabel('Row', fontsize=12, fontweight='bold')
    
    # Set tick labels
    ax.set_xticklabels(range(U.shape[1]))
    ax.set_yticklabels(range(U.shape[0]))
    
    plt.tight_layout()
    
    # Create descriptive filename
    if reward < 0:
        filename = f"gridworld_step_cost_neg{abs(reward)}.png"
    else:
        filename = f"gridworld_step_cost_pos{reward}.png"
    
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot for r={reward} to {filename}")

if __name__ == "__main__":
    rewards = [-0.04, -2, 0.1, 0.02, 1]
    
    for r in rewards:
        print(f"Solving for r(s) = {r}...")
        gw = GridWorld(reward_step=r)
        U = gw.value_iteration()
        policy = gw.get_optimal_policy(U)
        plot_results(U, policy, r)
        
    print("Done.")
