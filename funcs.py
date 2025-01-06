def transition_probability(w, s1, s2, P_A, P_D, W):
    """
    计算状态转移概率 P(w' | w, s1, s2)
    """
    if s1 == "high":
        attack_success_rate = P_A[1]
    else:
        attack_success_rate = P_A[0]

    if s2 == "high":
        detection_rate = P_D[1]
    else:
        detection_rate = P_D[0]

    # 状态转移概率
    if w == 0:  # 最小状态
        return [1 - detection_rate, detection_rate]  # [保持, 增加]
    elif w == W:  # 最大状态
        return [detection_rate, 1 - detection_rate]  # [减少, 保持]
    else:
        return [detection_rate / 2, 1 - detection_rate - attack_success_rate, attack_success_rate / 2]  # [减少, 保持, 增加]
    
def cost_intruder(s1, alpha_A):
    """
    入侵者的攻击成本
    """
    return alpha_A * (1 if s1 == "high" else 0.5)

def cost_ids(s2, alpha_D):
    """
    IDS 的检测成本
    """
    return alpha_D * (1 if s2 == "high" else 0.5)

def reward_intruder(w, beta_A, attack_success_rate):
    """
    入侵者的攻击收益
    """
    return beta_A * w * attack_success_rate

def reward_ids(w, beta_D, detection_rate):
    """
    IDS 的损失减少
    """
    return beta_D * w * detection_rate

def utility_intruder(w, s1, s2, alpha_A, beta_A, attack_success_rate):
    """
    入侵者的支付函数 U_1
    """
    return -cost_intruder(s1, alpha_A) + reward_intruder(w, beta_A, attack_success_rate)

def utility_ids(w, s1, s2, alpha_D, beta_D, detection_rate):
    """
    IDS 的支付函数 U_2
    """
    return -cost_ids(s2, alpha_D) - reward_ids(w, beta_D, detection_rate)

def robust_value_iteration(W, delta, P_A, P_D, alpha_D, beta_D, alpha_A, beta_A, iterations=100):
    """
    鲁棒动态规划求解 V(w)
    """
    V = [0] * (W + 1)  # 初始化 V(w)

    for _ in range(iterations):
        new_V = [0] * (W + 1)
        for w in range(W + 1):
            max_utility = float('-inf')  # 对于 IDS，取最大值
            for s2 in ["low", "high"]:
                min_utility = float('inf')  # 对于入侵者，取最小值
                for s1 in ["low", "high"]:
                    # 获取转移概率
                    probs = transition_probability(w, s1, s2, P_A, P_D, W)
                    # 计算支付
                    utility = utility_ids(w, s1, s2, alpha_D, beta_D, P_D[1])
                    # 加入未来价值的期望
                    expected_value = sum(p * V[w_next] for p, w_next in zip(probs, [w - 1, w, w + 1]))
                    total_utility = utility + delta * expected_value
                    min_utility = min(min_utility, total_utility)
                max_utility = max(max_utility, min_utility)
            new_V[w] = max_utility
        V = new_V  # 更新价值函数

    return V
