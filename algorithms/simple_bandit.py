import numpy as np

def train(num_actions, env, n, epsilon):
    """
    'A simple bandit algorithm' from Sutton and Barto page 25.
    """
    q = np.zeros(n_actions)
    n = np.zeros(n_actions)
    for i in range(n):
        p = np.array([1 - epsilon * ((num_actions + 1) / num_actions) if i == np.argmax(q) else epsilon / num_actions for i in range(num_actions)])
        a = np.random.choice(a=range(num_actions), p=p)
        r = env.step(a)
        n[a] += 1
        q[a] += (1 / n[a]) * (r - q[a])

    return q

