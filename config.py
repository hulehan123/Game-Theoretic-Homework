# 初始化参数
W = 30  # 最大被攻陷节点数
delta = 0.9  # 折扣因子
P_A = [0.3, 0.7]  # 攻击成功率范围
P_D = [0.6, 0.9]  # 检测成功率范围

# 策略集合
strategies_attacker = ["low", "high"]  # 攻击者策略：低攻击率和高攻击率
strategies_ids = ["low", "high"]  # IDS 策略：低扫描率和高扫描率

# 成本系数
alpha_A = 10  # 攻击成本系数
alpha_D = 8  # 检测成本系数
beta_A = 5   # 攻击收益系数
beta_D = 3   # 检测损失系数
