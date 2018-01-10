import torch
from random import shuffle
from itertools import count

def policy_iteration(num_actions, num_states, P, theta, gamma, V=None, policy=None):
    """
    policy is a deterministic policy.
    """
    raise NotImplementedError
    if V is not None:
        V = V
    else:
        V = torch.zeros(num_states)
    if policy is not None:
        policy = policy
    else:
        policy = torch.zeros(num_states)

    # Policy Evaluation
    while True:
        while True:
            delta = 0.0
            for s in range(num_states):
                v = V[s]
                new_value = sum([outcome[0] * (outcome[2] + gamma * V[outcome[1]]) for outcome in P[s][a]])
                V[s] = new_value
                delta = max(delta, torch.abs(v - V[s]))
                if delta < theta:
                    break

        policy_stable = True
        for s in range(num_states):
            old_action = policy[s][a]
            policy[s] = torch.argmax(torch.Tensor([sum([outcome[0] * (outcome[2] + gamma * V[outcome[1]]) for outcome in P[s][a]] for a in range(num_actions))]))
            if old_action != policy[1]:
                policy_stable = False

        if policy_stable:
            break

    return V, policy

def value_iteration(num_actions, num_states, gamma, theta, V):
    if V is not None:
        V = V
    else:
        V = torch.zeros(num_states)
    while True:
        for s in range(num_states):
            v = V[s]
            V[s] = max([sum([outcome[0] * (outcome[2] + gamma * V[outcome[1]]) for outcome in P[s][a]]) for a in range(num_actions)])
            delta = max(delta, torch.abs(v - V[s]))
        if delta < theta:
            break
    policy = torch.LongTensor([torch.argmax(torch.Tensor([sum([outcome[0] * (outcome[2] + gamma * V[outcome[1]]) for outcome in P[s][a]]) for a in range(num_actions)]))])
    return policy


def first_visit_monte_carlo(env, S, A, policy, V):
    if V is None:
        V = torch.zeros(S)
    if policy is None:
        policy = torch.zeros((S,A))

    returns = [None for _ in range(S)]
    while True:
        episode = []
        done = False
        state = env.reset()
        first_visit = [None for _ in range(S)]
        for t in count(0):
            m = Categorical(policy[state])
            a = m.sample()
            next_state, reward, done, info = env.step(a)
            episode.append(reward)
            if first_visit[state] is None:
                first_visit[state] = t
            state = next_state
            if done:
                break
        for s in range(S):
            if first_visit[s] is not None:
                if returns[s] is None:
                    returns[s] = (sum(episode[first_visit[s]:]), 1)
                else:
                    returns[s] = (returns[s][0] + sum(episode[first_visit[s]:]), returns[s][1] + 1)
    V = torch.FloatTensor([returns[s][0] / returns[s][1] for s in range(S)])
    return V # V \sim v_\pi

def monte_carlo_es(env, S, A, policy, Q):
    if Q is None:
        Q = torch.zeros((S,A))
    if policy is None:
        policy = torch.zeros(S)
    returns = [[None for a in range(A)] for s in range(S)]

    while True:
        pass

def on_policy_first_visit_mc_control(env, S, A, epsilon):
    Q = torch.zeros((S,A))
    returns = [[None for a in range(A)] for s in range(S)]
    policy = []
    for s in range(S):
        p = [1 - epsilon + epsilon/A if a == 0 else epsilon/A for a in range(A)]
        shuffle(p)
        policy.append(p)
    policy = torch.FloatTensor(policy)




        # Episode done.
