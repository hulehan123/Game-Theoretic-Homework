#实验 1: 被攻陷节点数量随时间变化
import random
from config import *
from funcs import *
import matplotlib.pyplot as plt

def simulate_game(strategy, steps=100):
    """
    模拟博弈过程，记录每个时间步的被攻陷节点数量
    """
    w = 0  # 初始状态
    history = [w]

    for _ in range(steps):
        if strategy == "fixed":
            detection_rate = 0.8  # 固定检测率
        elif strategy == "random":
            detection_rate = random.uniform(0.6, 0.9)  # 随机检测率
        elif strategy == "robust":
            detection_rate = P_D[1]  # 鲁棒优化检测率（使用高检测率）

        attack_success_rate = random.uniform(P_A[0], P_A[1])  # 攻击成功率

        # 状态转移
        probs = transition_probability(w, "high", "high", P_A, [detection_rate, detection_rate], W)
        next_state = random.choices([max(w - 1, 0), w, min(w + 1, W)], weights=probs)[0]
        history.append(next_state)
        w = next_state

    return history

# 模拟三种策略
steps = 100
fixed_history = simulate_game("fixed", steps)
random_history = simulate_game("random", steps)
robust_history = simulate_game("robust", steps)

# 绘制结果
plt.plot(fixed_history, label="Fixed Detection Rate")
plt.plot(random_history, label="Random Detection Rate")
plt.plot(robust_history, label="Robust Detection Rate")
plt.xlabel("Time Steps")
plt.ylabel("Compromised Nodes")
plt.title("State Dynamics under Different Strategies")
plt.legend()
plt.show()
