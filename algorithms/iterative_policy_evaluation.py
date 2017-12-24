import torch
from itertools import count

def iterative_policy_evaluation(policy, env, gamma, theta):
    """
    'Iterative policy evaluation' from Sutton and Barto page 61.

    Args:
        policy (numpy 2darray): Policy lookup table to be evaluated.

    Returns:
        V (numpy 1darray): Value of each states in the environment.
    """

    num_states = env.nS
    num_actions = env.nA
    V = torch.zeros(env.nS)
    delta = 0.0
    for _ in count(1):
        for s in range(num_states):
            v = V[s]
            V[s] = sum([policy[s][a] * sum([env.P[s][a][i][0] * (env.P[s][a][i][2] + V[env.P[s][a]i][1]) for i in range(len(env.P[s][a]))]) for a in range(env.nA)])
            delta = max(delta, torch.abs(v - V[s]))
        if delta < theta:
            break
    return V
