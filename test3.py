#实验 3: 参数敏感性分析
from config import *
from funcs import *
from test2 import calculate_total_utility

sensitivities = []
for pa_min in [0.2, 0.3, 0.4]:
    for pd_min in [0.5, 0.6, 0.7]:
        P_A = [pa_min, 0.7]
        P_D = [pd_min, 0.9]
        robust_utilities = calculate_total_utility("robust", steps)
        sensitivities.append((P_A, P_D, robust_utilities))

# 打印结果
for pa, pd, utility in sensitivities:
    print(f"P_A: {pa}, P_D: {pd}, Robust Utilities: {utility}")
