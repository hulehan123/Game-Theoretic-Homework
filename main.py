#################################################
# (A) 全局参数配置: 状态集、动作集、及相关常量
#################################################

# 下面这些与论文中的Notation和Section III的Simulation Setup相对应。
# 在论文中，W 表示最大可能被攻陷的节点数，delta 是折扣因子，p_min/p_max 是不确定检测率区间等等。

W = 5                       # 状态最大值: 最多30个节点被攻陷(论文示例)
States = range(W+1)         # 状态集合 \Omega = {0,1,...,W}

delta = 0.9                 # 折扣因子 (论文中示例是 0.9)

# 检测率区间 (p_min, p_max), 论文中在表I中给了多个区间实例，
# 例如 [0.79,0.81], [0.77,0.83], [0.74,0.86], ...
# 这里先示例性给一个区间, 你可以在实验时多次切换。
p_min, p_max = 0.74, 0.86

# 入侵者和IDS的动作集 (论文示例: 两种纯策略, Low/High)
AttackerActions = [0, 1]  # 0->低攻击, 1->高攻击
IDSactions = [0, 1]       # 0->低扫描, 1->高扫描

# 论文里(18)、(19) 式分别给出了入侵者和IDS在每个状态及动作组合下的即时收益/成本，
# 同时还带了一些常量(如 C^H_1, C^L_1, C^H_2, C^L_2, 以及 c(w) 等)。
# 在此处我们先定义一些可能用到的常数:
C_H1 = 8.0  # 入侵者高攻击时的收益(论文示例)
C_L1 = 2.0  # 入侵者低攻击时的收益
C_H2 = 4.0  # IDS高扫描的开销(或对攻击者的额外压制)
C_L2 = 1.0  # IDS低扫描的开销

# c(w) 表示若有 w 个节点被攻陷时，对IDS或网络的额外开销(论文中可能是 c(w)).
# 这里也可以对应文中"c(w)+C_i^H"或"c(w)+C_i^L"的组合。
def c_w(w):
    # 论文里并未给出严格的函数形式，有些地方用线性、有些用指数。此处演示成线性：
    return 3.0 * w

#################################################
# (B) 定义入侵者 & IDS的即时收益/成本函数
#################################################

# 对应论文中 \(\chi^1(w, s^1, s^2)\) 和 \(\chi^2(w, s^1, s^2)\)：
# (18)式与(19)式等说法，可根据论文中具体的计费、收益形式去修改。这里是示例。

def attacker_payoff(w, a1, a2):
    """
    入侵者在状态 w 时, 选择 a1(0=低攻击,1=高攻击),
    而 IDS选择 a2(0=低扫描,1=高扫描) 时的即时收益(论文中的 \chi^1)。
    """
    # 可直接对照论文(18)式:
    #   \chi^1(w, s^1, s^2) = ...
    # 这里仅作示例性写法:
    if a1 == 1:
        base = C_H1   # 高攻击
    else:
        base = C_L1   # 低攻击

    # 假设当IDS高扫描时, 入侵者收益会被额外减少:
    if a2 == 1:
        base -= (C_H2 / 2.0)
    else:
        base += 0.5  # IDS低扫描, 入侵者可稍微获益

    # 也可以让收益跟 w 再挂钩(若 w 已经很大, 入侵者多半也获益?),可自行修改
    # 这里演示加上一点 c_w(w)*0.01
    payoff = base + 0.01 * c_w(w)

    return payoff


def ids_cost(w, a1, a2):
    """
    IDS在状态 w 时, 动作组合 (a1,a2) 下的即时成本 \chi^2(w, s^1, s^2).
    与论文(19)式对应, 也常带 c(w) + C^H_2 / C^L_2 等项.
    """
    # 先把 w 引起的风险/损失: c_w(w)
    cost = c_w(w)

    # 当 a2=1(高扫描)时, 多消耗资源
    if a2 == 1:
        cost += C_H2
    else:
        cost += C_L2

    # 如果入侵者在高攻击 a1=1，则可能让IDS额外负担
    if a1 == 1:
        cost += 2.0

    return cost

#################################################
# (C) 状态转移概率: 结合不确定检测率
#################################################

def transition_prob(w, a1, a2, pD):
    """
    给定当前状态 w, 入侵者动作 a1, IDS动作 a2, 以及检测率 pD(在 [p_min, p_max]),
    返回一个字典 { 下一个状态w': 概率 }.
    
    论文(20)式和(21)式中描述了:
       - w->w+1: 新节点被攻陷(attack成功)
       - w->w-1: 被攻陷节点被恢复(recover)
       - w->w  : 也可能停留原状态(或者其中之一概率为0)
    边界: w=0时无法再-1, w=W时无法再+1.
    """
    # 这里是一个示例, 你可以直接把(20),(21)式的公式实现进去。
    # 例如论文中写到 p^(s^1,s^2) (w->w+1) = 1 - p̂; p^(s^1,s^2) (w->w-1) = p̂ ...
    # 并让 p̂ 跟 pD、a1,a2 挂钩。

    # 先写一个简单版本:
    p_succ = 0.5  # 攻击者成功的基本概率
    if a1 == 1:
        p_succ += 0.3   # 高攻击 => 成功率上升
    if a2 == 1:
        p_succ -= 0.2   # 高扫描 => 攻击成功率下降

    # 考虑检测率 pD: 若 pD 高, 攻击者成功率进一步下降
    p_succ -= (pD - 0.5)*0.5
    p_succ = max(min(p_succ, 1.0), 0.0)

    # 下面构造概率dict
    probs = {}
    if w == 0:
        # 无法 -1
        # 假设 w->w+1 的概率 = p_succ
        # 剩余 (1 - p_succ) 不变(留在0)
        probs[w+1] = p_succ
        probs[w] = 1.0 - p_succ
    elif w == W:
        # 无法 +1
        # 假设 w->w-1 的概率 = (1 - p_succ)
        # 剩余 p_succ 留在W
        probs[w-1] = (1.0 - p_succ)
        probs[w] = p_succ
    else:
        # 一般情况
        # 可让 w->w+1 = p_succ,
        # w->w-1 = (1-p_succ)*0.5, w->w = (1-p_succ)*0.5
        # 只是个示例，可自由改成(21)式那种 w->w+1 = (W - w)/W * (1 - p̂), ...
        probs[w+1] = p_succ
        probs[w-1] = (1.0 - p_succ)*0.5
        probs[w] = (1.0 - p_succ)*0.5

    return probs

#################################################
# (D) Pyomo模型主体
#################################################
import pyomo.environ as pyo

def build_robust_stochastic_game_model_no_minexpr(
    States, 
    AttackerActions, 
    IDSactions,
    attacker_payoff_func, 
    ids_cost_func,
    transition_prob_func,
    p_candidates,
    delta
):
    """
    构建一个简化版(示例)的鲁棒随机对策模型,
    不使用 pyo.min_expression / pyo.max_expression,
    而是用“辅助变量 + 约束”来实现对表达式的 min/max。
    """
    model = pyo.ConcreteModel("RobustGame_NoMinExpr")

    # ========== 1) 定义决策变量 ================
    model.xAtt = pyo.Var(States, AttackerActions, domain=pyo.NonNegativeReals)
    model.xIDS = pyo.Var(States, IDSactions,      domain=pyo.NonNegativeReals)

    # VAtt[w], VIDS[w]: 在状态 w 时, 入侵者/IDS 的鲁棒期望价值
    model.VAtt = pyo.Var(States, domain=pyo.Reals)
    model.VIDS = pyo.Var(States, domain=pyo.Reals)

    #
    # 另外，我们需要在 “取 min” 或 “取 max” 的地方引入辅助变量
    # 比如 attacker_minVal[w,a1,a2], ids_maxVal[w,a1,a2] 等
    #
    # attacker_minVal[w,a1,a2] =  min_{pD in p_candidates}  [ sum_{w'}( prob(w->w') * VAtt[w'] ) ]
    #
    # 这样我们就能把 “min(...)” 变成若干约束。
    #
    # 同理，对于 IDS 侧，要取 max(...) => 引入 ids_maxVal[w,a1,a2].
    #

    model.attacker_minVal = pyo.Var(
        [(w,a1,a2) for w in States for a1 in AttackerActions for a2 in IDSactions],
        domain=pyo.Reals
    )
    model.ids_maxVal = pyo.Var(
        [(w,a1,a2) for w in States for a1 in AttackerActions for a2 in IDSactions],
        domain=pyo.Reals
    )

    # ========== 2) 定义混合策略概率和约束 ================
    def attacker_prob_sum_rule(m, w):
        return sum(m.xAtt[w,a1] for a1 in AttackerActions) == 1
    model.attacker_prob_sum_con = pyo.Constraint(States, rule=attacker_prob_sum_rule)

    def ids_prob_sum_rule(m, w):
        return sum(m.xIDS[w,a2] for a2 in IDSactions) == 1
    model.ids_prob_sum_con = pyo.Constraint(States, rule=ids_prob_sum_rule)

    # ========== 3) 对 attacker_minVal[w,a1,a2] 做“取最小”约束 ================
    #
    # attacker_minVal[w,a1,a2] <= sum_{w'}( prob(w->w') * VAtt[w'] )  对每个 pD
    # 这样在后面为了满足 VAtt[w] >= ... * attacker_minVal 的不等式， 
    # attacker_minVal 就会被 solver 尽量推大，但又不能超过任何 scenario 的表达式 => 等效 min(...)
    #

    def attacker_minVal_con_rule(m, w, a1, a2, i):
        """
        i 用来索引 p_candidates: i=0..(len(p_candidates)-1)
        """
        pd = p_candidates[i]
        # 计算 scenarioExp = sum_{w'}( prob(w->w'|a1,a2,pd) * VAtt[w'] )
        trans = transition_prob_func(w,a1,a2,pd)
        scenario_value = sum(trans[wn]*m.VAtt[wn] for wn in trans.keys())
        return m.attacker_minVal[w,a1,a2] <= scenario_value

    # 我们需要为 (w,a1,a2) 的每个 p_candidates 都加一条约束
    model.attacker_minVal_con = pyo.Constraint(
        [(w,a1,a2,i) 
         for w in States 
         for a1 in AttackerActions 
         for a2 in IDSactions
         for i in range(len(p_candidates))],
        rule=attacker_minVal_con_rule
    )

    # ========== 4) 对 IDS 的 ids_maxVal[w,a1,a2] 做“取最大”约束 ================
    #
    # 如果想取 max( scenarioExp ), 那就：  ids_maxVal[w,a1,a2] >= scenarioExp
    # 同理，在后续不等式中 ids_maxVal 可能被“尽量往下推”，从而 = max(...) 。

    def ids_maxVal_con_rule(m, w, a1, a2, i):
        pd = p_candidates[i]
        trans = transition_prob_func(w,a1,a2,pd)
        scenario_value = sum(trans[wn]*m.VIDS[wn] for wn in trans.keys())
        # 对于“取最大” => ids_maxVal[w,a1,a2] >= scenario_value
        return m.ids_maxVal[w,a1,a2] >= scenario_value

    model.ids_maxVal_con = pyo.Constraint(
        [(w,a1,a2,i) 
         for w in States 
         for a1 in AttackerActions 
         for a2 in IDSactions
         for i in range(len(p_candidates))],
        rule=ids_maxVal_con_rule
    )

    # ========== 5) Bellman-like 不等式 ================
    #
    # attacker:
    #   VAtt[w] >= sum_{a1,a2} [ xAtt[w,a1]*xIDS[w,a2] * ( immediate + delta * attacker_minVal[w,a1,a2] ) ]
    #
    # IDS:
    #   VIDS[w] >= sum_{a1,a2} [ xAtt[w,a1]*xIDS[w,a2] * ( cost + delta * ids_maxVal[w,a1,a2] ) ]

    def attacker_value_rule(m, w):
        sum_expr = []
        for a1 in AttackerActions:
            for a2 in IDSactions:
                immediate = attacker_payoff_func(w,a1,a2)
                combo_val = immediate + delta * m.attacker_minVal[w,a1,a2]
                prob = m.xAtt[w,a1] * m.xIDS[w,a2]
                sum_expr.append(prob * combo_val)
        return m.VAtt[w] >= sum(sum_expr)

    model.attacker_value_con = pyo.Constraint(States, rule=attacker_value_rule)

    def ids_value_rule(m, w):
        sum_expr = []
        for a1 in AttackerActions:
            for a2 in IDSactions:
                immediate = ids_cost_func(w,a1,a2)
                combo_val = immediate + delta * m.ids_maxVal[w,a1,a2]
                prob = m.xAtt[w,a1] * m.xIDS[w,a2]
                sum_expr.append(prob * combo_val)
        return m.VIDS[w] >= sum(sum_expr)

    model.ids_value_con = pyo.Constraint(States, rule=ids_value_rule)

    # ========== 6) 定义一个目标函数(示例) ================
    def obj_rule(m):
        # 举例: minimize sum_{w}(VIDS[w] - VAtt[w])
        return sum(m.VIDS[w] - m.VAtt[w] for w in States)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model

# main.py

import pyomo.environ as pyo
if __name__ == "__main__":
    # 准备参数
    W = 30
    States = range(W+1)
    AttackerActions = [0,1]
    IDSactions = [0,1]
    p_candidates = [0.74, 0.86]  # 不确定区间两端点
    delta = 0.9

    # 构建模型 (不使用 min_expression/max_expression)
    model = build_robust_stochastic_game_model_no_minexpr(
        States,
        AttackerActions,
        IDSactions,
        attacker_payoff,
        ids_cost,
        transition_prob,
        p_candidates,
        delta
    )

    # 求解
    solver = pyo.SolverFactory('ipopt')
    result = solver.solve(model, tee=True)
    print(result)

    # 查看决策变量
    print("==== Attacker's strategies ====")
    for w in States:
        p_low  = pyo.value(model.xAtt[w,0])
        p_high = pyo.value(model.xAtt[w,1])
        print(f"w={w}, Att(Low,High)=({p_low:.4f}, {p_high:.4f})")

    print("\n==== IDS's strategies ====")
    for w in States:
        p_low  = pyo.value(model.xIDS[w,0])
        p_high = pyo.value(model.xIDS[w,1])
        print(f"w={w}, IDS(Low,High)=({p_low:.4f}, {p_high:.4f})")

    print("\n==== Value functions ====")
    for w in States:
        vA = pyo.value(model.VAtt[w])
        vI = pyo.value(model.VIDS[w])
        print(f"w={w}, VAtt={vA:.4f}, VIDS={vI:.4f}")

