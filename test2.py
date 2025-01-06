#实验 2: 支付值对比
import random
from config import *
from funcs import *

def calculate_total_utility(strategy, steps=100):
    """
    计算给定策略的 IDS 和攻击者的总支付值
    """
    w = 0  # 初始状态
    total_utility_ids = 0
    total_utility_attacker = 0

    for _ in range(steps):
        if strategy == "fixed":
            detection_rate = 0.8
        elif strategy == "random":
            detection_rate = random.uniform(0.6, 0.9)
        elif strategy == "robust":
            detection_rate = P_D[1]

        attack_success_rate = random.uniform(P_A[0], P_A[1])

        # 计算支付值
        utility_ids = utility_ids(w, "high", "high", alpha_D, beta_D, detection_rate)
        utility_attacker = utility_intruder(w, "high", "high", alpha_A, beta_A, attack_success_rate)

        total_utility_ids += utility_ids
        total_utility_attacker += utility_attacker

        # 状态转移
        probs = transition_probability(w, "high", "high", P_A, [detection_rate, detection_rate], W)
        w = random.choices([max(w - 1, 0), w, min(w + 1, W)], weights=probs)[0]

    return total_utility_ids, total_utility_attacker

# 计算三种策略的支付值
fixed_utilities = calculate_total_utility("fixed", steps)
random_utilities = calculate_total_utility("random", steps)
robust_utilities = calculate_total_utility("robust", steps)

# 打印结果
print(f"Fixed Strategy (IDS, Attacker): {fixed_utilities}")
print(f"Random Strategy (IDS, Attacker): {random_utilities}")
print(f"Robust Strategy (IDS, Attacker): {robust_utilities}")
