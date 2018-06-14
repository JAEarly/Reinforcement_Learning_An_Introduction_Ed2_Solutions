from chapter02.BanditTestBed import createBandit
import pickle
import os
import numpy as np
import math


def runAndPlot(axes, repeats, plays, n, method, config, kwconfig, label, useSave=False, nonStationary=False):
    if useSave and os.path.isfile(label + '_rewards.p'):
        average_rewards = pickle.load(open(label + '_rewards.p', "rb"))
        average_optimals = pickle.load(open(label + '_optimals.p', "rb"))
    else:
        runRewards = []
        runOptimals = []
        for _ in range(repeats):
            bandit = createBandit(n, nonStationary)
            rewards, optimal_chosen = method(bandit, plays, *config, **kwconfig)
            runRewards.append(rewards)
            runOptimals.append(optimal_chosen)
        average_rewards = np.mean(runRewards, axis=0)
        average_optimals = np.mean(runOptimals, axis=0)*100
        if useSave:
            pickle.dump(average_rewards, open(label + '_rewards.p', 'wb'))
            pickle.dump(average_optimals, open(label + '_optimals.p', 'wb'))
    X = np.arange(0, plays, 1)
    axes[0].plot(X, average_rewards, label=label)
    axes[1].plot(X, average_optimals, label=label)
    axes[2].plot(X, np.cumsum(average_rewards), label=label)
    print("{:^12s}".format(label), "{:1.3f}".format(np.cumsum(average_rewards)[-1]))
    return average_rewards, average_optimals

def softmax(l, tau):
    d = 0
    for b in l:
        d += math.exp(b)/tau
    l1 = []
    for a in l:
        l1.append((math.exp(a)/tau)/d)
    return l1


def runEGreedy(bandit, plays, epsilon, step_method, initial_estimate, random_selection='normal', tau=0, step_constant=0):
    arm_estimates = [initial_estimate] * bandit.size()
    num_pulls = [0] * bandit.size()
    rewards = []
    optimal_chosen = []
    for _ in range(plays):
        if np.random.uniform(0, 1) < epsilon:
            if random_selection == 'softmax':
                selection_probs = softmax(arm_estimates, tau)
                action_value = np.random.choice(selection_probs, p=selection_probs)
                chosen_arm = np.argmax(selection_probs == action_value)
            elif random_selection == 'normal':
                chosen_arm = np.random.randint(bandit.size())
            else:
                raise ValueError('Unknown random selection method:', random_selection)
        else:
            m = max(arm_estimates)
            chosen_arm = np.random.choice([i for i, j in enumerate(arm_estimates) if j == m])
        reward = bandit.pull(chosen_arm)
        num_pulls[chosen_arm] += 1
        if step_method == 'sample_average':
            step_size = 1/num_pulls[chosen_arm]
        elif step_method == 'constant':
            step_size = step_constant
        else:
            raise ValueError('Unknown step method:', step_method)
        arm_estimates[chosen_arm] += step_size*(reward - arm_estimates[chosen_arm])
        rewards.append(reward)
        optimal_chosen.append(1 if chosen_arm == bandit.getOptimalArmId() else 0)
    return rewards, optimal_chosen

def runReinforcementComparison(bandit, plays, alpha, beta, initialReferenceReward, useCrowdingCountermeasure=False):
    if not 0 < beta <= 1:
        raise ValueError('Beta must be between 0 and 1.')
    if not 0 < alpha <= 1:
        raise ValueError('Alpha must be between 0 and 1.')
    arm_preferences = [1/bandit.size()] * bandit.size()
    reference_rewards = [initialReferenceReward] * bandit.size()
    rewards = []
    optimal_chosen = []
    for _ in range(plays):
        selection_probs = softmax(arm_preferences, 1)
        action_value = np.random.choice(selection_probs, p=selection_probs)
        chosen_arm = np.argmax(selection_probs == action_value)
        reward = bandit.pull(chosen_arm)
        if useCrowdingCountermeasure:
            arm_preferences[chosen_arm] += beta*(reward - reference_rewards[chosen_arm])*(1 - selection_probs[chosen_arm])
        else:
            arm_preferences[chosen_arm] += beta*(reward - reference_rewards[chosen_arm])
        reference_rewards[chosen_arm] += alpha*(reward - reference_rewards[chosen_arm])
        rewards.append(reward)
        optimal_chosen.append(1 if chosen_arm == bandit.getOptimalArmId() else 0)
    return rewards, optimal_chosen

def runPursuitMethod(bandit, plays, beta):
    if beta <= 0:
        raise ValueError('Beta must be greater than zero.')
    arm_probabilities = [1/bandit.size()] * bandit.size()
    arm_estimates = [0] * bandit.size()
    num_pulls = [0] * bandit.size()
    rewards = []
    optimal_chosen = []
    #print(arm_probabilities)
    #print(arm_estimates)
    #print(num_pulls)
    #print()
    for _ in range(plays):
        # Choose arm based on probabilities
        chosen_arm = np.random.choice(range(bandit.size()), p=arm_probabilities)

        # Pull arm and update estimates
        reward = bandit.pull(chosen_arm)
        num_pulls[chosen_arm] += 1
        arm_estimates[chosen_arm] += (1/num_pulls[chosen_arm])*(reward - arm_estimates[chosen_arm])

        # Find greedy arm
        m = max(arm_estimates)
        greedy_arm = np.random.choice([i for i, j in enumerate(arm_estimates) if j == m])

        # Update arm probabilities based on greedy/not greedy
        arm_probabilities[greedy_arm] += beta * (1 - arm_probabilities[greedy_arm])
        for arm in range(bandit.size()):
            if arm != greedy_arm:
                arm_probabilities[arm] += beta * (0 - arm_probabilities[arm])

        rewards.append(reward)
        optimal_chosen.append(1 if chosen_arm == bandit.getOptimalArmId() else 0)
        # print('Pulled ', chosen_arm, 'Value', reward)
        # print('Greedy', greedy_arm)
        # print(arm_probabilities)
        # print(arm_estimates)
        # print(num_pulls)
        # print()
    return rewards, optimal_chosen
