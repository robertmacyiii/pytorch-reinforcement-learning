import torch
from torch.distributions import Categorical

def train(num_actions, env, n, epsilon):
    """
    'A simple bandit algorithm' from Sutton and Barto page 25.
    """
    q = torch.zeros(n_actions)
    n = torch.zeros(n_actions)
    for i in range(n):
        
        p = torch.Tensor([1 - epsilon * ((num_actions + 1) / num_actions) if i == np.argmax(q) else epsilon / num_actions for i in range(num_actions)])
        m = Categorical(p)
        a = m.sample()
        r = env.step(a)
        n[a] += 1
        q[a] += (1 / n[a]) * (r - q[a])

    return q

