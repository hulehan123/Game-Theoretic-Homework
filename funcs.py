import numpy as np
from config import *

# 成本函数
def cost_function(w):
    """计算被攻陷节点的成本"""
    return alpha * (np.exp(beta * w) - 1)

# 状态转移概率
def state_transition_prob(w, s1, s2, p):
    """计算状态转移概率"""
    prob = np.zeros(W + 1)
    if w > 0:
        prob[w - 1] = (w / W) * p  # 恢复一个节点的概率
    prob[w] = (w / W) * (1 - p) + ((W - w) / W) * p  # 状态不变的概率
    if w < W:
        prob[w + 1] = ((W - w) / W) * (1 - p)  # 攻陷一个节点的概率
    return prob

# 生成状态转移矩阵
def generate_transition_matrix(s1, s2, p):
    """生成状态转移矩阵"""
    transition_matrix = np.zeros((W + 1, W + 1))
    for w in range(W + 1):
        transition_matrix[w] = state_transition_prob(w, s1, s2, p)
    return transition_matrix

# 玩家即时成本函数
def immediate_cost(w, s1, s2):
    """计算即时成本"""
    attacker_cost = -cost_function(w) + (C1_H if s1 == s1_high else C1_L)
    ids_cost = cost_function(w) + (C2_H if s2 == s2_high else C2_L)
    return attacker_cost, ids_cost

# 总成本计算
def compute_total_cost(transition_matrix, initial_state, iterations=100):
    """计算总成本"""
    state = initial_state
    total_attacker_cost = 0
    total_ids_cost = 0

    for t in range(iterations):
        attacker_cost, ids_cost = immediate_cost(state, s1, s2)
        total_attacker_cost += (delta ** t) * attacker_cost
        total_ids_cost += (delta ** t) * ids_cost

        # 随机转移到下一状态
        state = np.random.choice(range(W + 1), p=transition_matrix[state])

    return total_attacker_cost, total_ids_cost