import matplotlib.pyplot as plt
from config import *  # 导入实验参数
from funcs import *

# 初始化价值函数
V = robust_value_iteration(W, delta, P_A, P_D, alpha_D, beta_D, alpha_A, beta_A)

# 绘制价值函数
plt.plot(range(W + 1), V, label="Robust Value Function")
plt.xlabel("State (w)")
plt.ylabel("Value (V(w))")
plt.title("Value Function for IDS")
plt.legend()
plt.show()
