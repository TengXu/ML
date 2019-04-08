import gym
import numpy as np
from numpy import random
from collections import Counter
from functools import reduce
import time

def valueIter(env, gamma = 1.0, theta = 1e-10):
    nStates = env.nS
    nActions = env.nA
    P = env.P
    v = np.zeros(nStates)
    iters = 0
    converged = False
    while not converged:
        v_old = np.copy(v)
        Q = np.zeros((nStates, nActions))
        for s in range(nStates):
            for a in range(nActions):
                for prob, s_next, reward, done in P[s][a]:
                    if not done:
                        Q[s][a] += prob * (reward + (gamma* v_old[s_next]))
                    else:
                        Q[s][a] += prob * (reward)
        v = np.max(Q,1)
        if(np.max(np.abs(v - v_old)) < theta):
            converged = True
        iters += 1
    return v, iters

def getPolicy(env,v, gamma = 1.0):
    nStates = env.nS
    nActions = env.nA
    P = env.P
    policy = np.zeros(nStates)
    for s in range(nStates):
        action_values = np.zeros(nActions)
        for a in range(nActions):
            for prob, s_next, r, done in P[s][a]:
                if not done:
                    action_values[a] += (prob * (r + gamma * v[s_next]))
                else:
                    action_values[a] += (prob * (r))
        policy[s] = np.argmax(action_values)
    return policy

def runPolicy(env, policy, useDiscount = True, gamma = 1.0):
    reward = 0
    s = env.reset()
    i = 0
    done = False
    while not done:
        s, r, done, _ = env.step(policy[s])
        if useDiscount:
            reward += (gamma**i * r)
        else:
            reward += r
        i += 1
    return reward

def scorePolicy(env, policy, useDiscount = True, gamma = 1.0,  nSamples = 100):
    return np.mean([runPolicy(env, policy, useDiscount, gamma) for _ in range(nSamples)])

def getPolicyV(env, policy, gamma = 1.0, theta = 1e-10, maxIters = 2000):
    nStates = env.nS
    nActions = env.nA
    P = env.P
    v = np.zeros(nStates)
    iters = 0
    converged = False
    while not converged and iters < maxIters:
        v_prev = np.copy(v)
        for s in range(nStates):
            a = policy[s]
            action_value = 0
            for prob, s_next, r, done in P[s][a]:
                if not done:
                    action_value += prob * (r + gamma * v_prev[s_next])
                else:
                    action_value += prob * (r)
            v[s] = action_value
        if(np.max(np.abs(v - v_prev)) < theta):
            converged = True
        iters += 1
    return v


def policyIter(env, gamma = 1.0, maxIters = 200000,  theta = 1e-10, nSamples = 100):
    nStates = env.nS
    nActions = env.nA
    P = env.P
    converged = False
    policy = np.random.choice(nActions, nStates)
    i = 0
    while i < maxIters and not converged:
        policy_prev = np.copy(policy)
        v = getPolicyV(env, policy, gamma, theta)
        policy = getPolicy(env, v, gamma)

        if(np.all(policy == policy_prev)):
            converged = True
        i += 1
    return policy, i

def QLearning(problem, maxIters = 200, gamma = .99, alpha = .2, episodes = 500000, epsilon = 1, epsilon_min = .2, epsilon_decay = .999):
    nState = problem.observation_space.n
    nAction = problem.action_space.n
    #Q = np.zeros((nState, nAction))
    Q = np.random.rand(nState, nAction)
    cnt = Counter()
    for i in range(episodes):
        s = problem.reset()
        done = False
        Q_old = Q.copy()
        step = 0
        while not done:
            a = problem.action_space.sample()
            if random.uniform(0,1) > epsilon:            
                a = np.argmax(Q[s])
            s_, r, done, info = problem.step(a)
            q_old = Q[s,a]
            q_new = (1-alpha)*q_old + alpha*(r+gamma*np.max(Q[s_,:]))
            Q[s,a] = q_new
            s = s_
            step += 1
        epsilon = max(epsilon_min, epsilon*epsilon_decay)

        pol = np.argmax(Q, 1)
        cnt[str(pol)] += 1
        # if(i%100 == 99):
        #     most_common = cnt.most_common(3)
        #     spol = reduce(lambda x,y: str(x)+str(y), pol)
        #     score = scorePolicy(problem, pol)
        #     print("iteration: {}\t epsilon: {}\t policy: {}\t score: {} \t most common: {}".format(i,epsilon,spol,score,[x[1] for x in most_common]))
        
    return np.argmax(Q, 1), episodes

#   2 mdps  
frozen_lake1 = gym.make("FrozenLake-v0")
frozen_lake1_env = frozen_lake1.env
frozen_lake2 = gym.make("FrozenLake8x8-v0")
frozen_lake2_env = frozen_lake2.env

problems = [frozen_lake1, frozen_lake2]
#problems = [frozen_lake2]

for problem in problems:
    env = problem.env
    print("-----" , str(env), "-----")
    print("states: {}\nactions: {}".format(env.nS, env.nA))
    #####
    # do value iteration
    t0 = time.perf_counter()
    v_value, i_value = valueIter(env, 1.0)
    tf = time.perf_counter()
    t_value = tf-t0
    # get policy
    p_star_value = getPolicy(env, v_value)
    score_value = scorePolicy(env, p_star_value, False)
    #####
    print("Value Iteration:\n\ttime: {}\n\titers: {}\n\tscore: {}".format(t_value, i_value,score_value))
    ############

    ############
    # do policy iteration
    t0 = time.perf_counter()
    p_star_policy, i_policy = policyIter(env)
    tf = time.perf_counter()
    t_policy = tf-t0
    # derive value matrix
    v_policy = getPolicyV(env, p_star_policy)
    score_policy = scorePolicy(env, p_star_policy, False)
    print("Policy Iteration:\n\ttime: {}\n\titers: {}\n\tscore: {}".format(t_policy, i_policy,score_policy))
    #####    
    #print(v_policy, "\n", p_star_policy)
    print("Same Policy: {}".format(np.all(p_star_value == p_star_policy)))
    ##########
    ############
    # do Q-learning
    t0 = time.perf_counter()
    p_star_Q, i_Q = QLearning(problem, alpha = 0.2)
    tf = time.perf_counter()
    t_Q = tf-t0
    score_Q = scorePolicy(env, p_star_Q, False)
    print('alpha = 0.2')
    print("Q-learning:\n\ttime: {}\n\titers: {}\n\tscore: {}".format(t_Q, i_Q,score_Q))
    t0 = time.perf_counter()
    p_star_Q, i_Q = QLearning(problem, alpha = 0.3)
    tf = time.perf_counter()
    t_Q = tf-t0
    score_Q = scorePolicy(env, p_star_Q, False)
    print('alpha = 0.3')
    print("Q-learning:\n\ttime: {}\n\titers: {}\n\tscore: {}".format(t_Q, i_Q,score_Q))
    t0 = time.perf_counter()
    p_star_Q, i_Q = QLearning(problem, alpha = 0.5)
    tf = time.perf_counter()
    t_Q = tf-t0
    score_Q = scorePolicy(env, p_star_Q, False)
    print('alpha = 0.5')
    print("Q-learning:\n\ttime: {}\n\titers: {}\n\tscore: {}".format(t_Q, i_Q,score_Q))
    print("")
    print("Sample of policies:\tValue Iteration\tPolicy Iteration\tQ-Learning policy")
#    if(len(p_star_Q) > 8):
#        print(p_star_value[:8], p_star_policy[:8], p_star_Q[:8])
#    else:
#        print(p_star_value, p_star_policy, p_star_Q)
    
