import numpy as np
from lib import GridworldMDP, print_value, print_deterministic_policy, init_value, random_policy


def policy_evaluation_one_step(mdp, V, policy, discount=0.99):
    """ Computes one step of policy evaluation.
    Arguments: MDP, value function, policy, discount factor
    Returns: Value function of policy
    """
    # Init value function array
    V_new = V.copy()

    # Iterate over all states
    for s in range(mdp.num_states):
        v = 0
        # Iterate over all actions
        for a, action_prob in enumerate(policy[s]):
            # Iterate over all possible outcomes of action a in state s
            for prob, next_state, reward, done in mdp.P[s][a]:
                v += action_prob * prob * (reward + discount * V[next_state])
        # Update
        V_new[s] = v

    return V_new


def policy_evaluation(mdp, policy, discount=0.99, theta=0.00001):
    """ Computes full policy evaluation until convergence.
    Arguments: MDP, policy, discount factor, theta
    Returns: Value function of policy
    """
    # Init value function array
    V = init_value(mdp)

    # Iterate until convergence
    max_delta = theta + 1
    while max_delta > theta:
        # Do one step of policy evaluation
        V_new = policy_evaluation_one_step(mdp, V, policy, discount)
        # Compute maximum delta between old and new values
        max_delta = np.max(np.abs(V - V_new))
        V = V_new

    return V


def policy_improvement(mdp, V, discount=0.99):
    """ Computes greedy policy w.r.t a given MDP and value function.
    Arguments: MDP, value function, discount factor
    Returns: policy
    """
    # Initialize a policy array in which to save the greed policy 
    policy = np.zeros_like(random_policy(mdp))

    # Iterate over all states
    for s in range(mdp.num_states):
        # action-values for current state s
        Q_s = np.zeros(mdp.num_actions)
        # Iterate over all actions
        for a in range(mdp.num_actions):
            # Iterate over all possible outcomes of action a in state s
            for prob, next_state, reward, done in mdp.P[s][a]:
                Q_s[a] += prob * (reward + discount * V[next_state])
        # Find best greedy action
        greedy_action = np.argmax(Q_s)
        # Set action probability of that action to 1.0
        policy[s] = np.eye(mdp.num_actions)[greedy_action]
    return policy


def policy_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the policy iteration (PI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """

    # Start from random policy
    policy = random_policy(mdp)
    while True:
        # Evaluate current policy
        V = policy_evaluation(mdp, policy, discount, theta)
        # Define greedy policy w.r.t V
        policy_new = policy_improvement(mdp, V, discount)
        # Check if something changed
        if np.array_equal(policy_new, policy):
            break
        policy = policy_new
    return V, policy


def value_iteration(mdp, discount=0.99, theta=0.01):
    """ Computes the value iteration (VI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    # Init value function array
    V = init_value(mdp)

    # Iterate until convergence
    while True:
        # We need a copy in order to calculate some deltas (i.e. we don't do in-place VI)
        V_new = V.copy()
        # Iterate over all states
        for s in range(mdp.num_states):
            Q_s = np.zeros(mdp.num_actions)
            # Iterate over all actions
            for a in range(mdp.num_actions):
                # Iterate over all possible outcomes of action a in state s
                for prob, next_state, reward, done in mdp.P[s][a]:
                    Q_s[a] += prob * (reward + discount * V[next_state])
            # Update
            V_new[s] = np.max(Q_s)
        # Compute deltas
        max_delta = np.max(np.abs(V_new - V))
        V = V_new
        # Check if converged
        if max_delta < theta:
            break

    policy = policy_improvement(mdp, V)

    return V, policy


def value_iteration2(mdp, discount=0.99, theta=0.01):
    """ Computes the value iteration (VI) algorithm.
    Arguments: MDP, discount factor, theta
    Returns: value function, policy
    """
    # Start from random policy
    V = init_value(mdp)
    policy = random_policy(mdp)
    while True:
        # Evaluate current policy
        V = policy_evaluation_one_step(mdp, V, policy)
        # Define greedy policy w.r.t V
        policy_new = policy_improvement(mdp, V, discount)
        # Check if something changed
        if np.array_equal(policy_new, policy):
            break
        policy = policy_new
    return V, policy


if __name__ == "__main__":
    # Create the MDP
    mdp = GridworldMDP([6, 6])
    # discount = 0.99
    discount = 1.0
    theta = 0.000001

    # Print the gridworld to the terminal
    print('---------')
    print('GridWorld')
    print('---------')
    mdp.render()

    # Create a random policy
    V = init_value(mdp)
    policy = random_policy(mdp)

    # Do one step of policy evaluation and print
    print('-----------------------------')
    print('One step of policy evaluation (random policy):')
    print('-----------------------------')
    V = policy_evaluation_one_step(mdp, V, policy, discount=discount)
    print_value(V, mdp)

    # Do a full (random) policy evaluation and print
    print('---------------------------------------')
    print('Full policy evaluation (random policy):')
    print('---------------------------------------')
    V = policy_evaluation(mdp, policy, discount=discount, theta=theta)
    print_value(V, mdp)

    # Do one step of policy improvement and print
    # "Policy improvement" basically means "Take greedy action w.r.t given a given value function"
    print('-------------------')
    print('Policy improvement:')
    print('-------------------')
    policy = policy_improvement(mdp, V, discount=discount)
    print_deterministic_policy(policy, mdp)

    # Do a full PI and print
    print('-----------------')
    print('Policy iteration:')
    print('-----------------')
    V, policy = policy_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)

    # Do a full VI and print
    print('---------------')
    print('Value iteration')
    print('---------------')
    V, policy = value_iteration(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)

    # Do a full VI and print
    print('---------------')
    print('Value iteration (one-step PE + PI)')
    print('---------------')
    V, policy = value_iteration2(mdp, discount=discount, theta=theta)
    print_value(V, mdp)
    print_deterministic_policy(policy, mdp)
