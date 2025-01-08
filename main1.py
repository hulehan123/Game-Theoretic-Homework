# main.py

import pyomo.environ as pyo
import math
############################################
# 1. 全局参数
############################################

W = 30
States = range(W+1)

delta = 0.6

# 可以先只用 p_nominal=0.8 测试名义情况, 然后再切换到 p_candidates=[p_min,p_max].
p_min, p_max = 0.77, 0.83
# p_min, p_max = 0.8, 0.8
p_candidates = [0.8]  # <-- 名义场景（无区间不确定）
# p_candidates = [p_min, p_max]  # <-- 打开鲁棒区间

# 动作集
AttackerActions = [0,1]  # 0->low attack, 1->high attack
IDSactions      = [0,1]  # 0->low scan,   1->high scan

# 缩减一下常量, 避免出现过大值
C_H1 = 8  # 入侵者高攻击时的收益(调小一点)
C_L1 = 0  # 入侵者低攻击时的收益
C_H2 = 10  # IDS高扫描的开销(原是4)
C_L2 = 2  # IDS低扫描的开销(原是1)

# c(w) 改成 w 而非 3.0*w, 减少量级
def c_w(w):
    return 10 * (math.exp(0.05 * w) - 1)

############################################
# 2. 即时收益/成本函数
############################################

# def attacker_payoff(w, a1, a2):

#     if a1 == 1:
#         base = C_H1  # high attack
#     else:
#         base = C_L1  # low attack

#     # 若IDS高扫描, 入侵者收益被额外扣一些
#     if a2 == 1:
#         base -= (C_H2 / 2.0)  # 让它负一点
#     else:
#         base += 0.2  # IDS低扫描, 再给一点好处

#     # e.g. w越大, 入侵者可能更容易利用, + 0.05*w
#     payoff = base + 0.05 * w
#     return payoff

# def ids_cost(w, a1, a2):

#     cost = c_w(w)

#     # IDS高扫描 => 多开销
#     if a2 == 1:
#         cost += C_H2
#     else:
#         cost += C_L2

#     # 如果入侵者在高攻击
#     if a1 == 1:
#         cost += 1.0

#     return cost

def attacker_payoff(w, a1, a2):
    payoff = -c_w(w)
    if a1 == 1:
        payoff = payoff + C_H1
    else:
        payoff = payoff + C_L1
    return payoff

def ids_cost(w, a1, a2):
    cost = c_w(w)
    if a2 == 1:
        cost += C_H2
    else:
        cost += C_L2
    return cost
        

############################################
# 3. 状态转移概率
############################################

def transition_prob(w, a1, a2, pD):
    """
    根据论文公式 (21) 实现状态转移概率
    :param w: 当前状态
    :param a1: 攻击者的行为 (0: low attack, 1: high attack)
    :param a2: IDS 的行为 (0: low scan, 1: high scan)
    :param pD: 攻击成功的基础概率
    :return: 下一状态的转移概率字典
    """
    # 攻击成功率调整
    p_succ = pD
    if a1 == 1:  # 高攻击
        p_succ += 0.1
    if a2 == 1:  # 高扫描
        p_succ -= 0.1
    p_succ = max(min(p_succ, 1.0), 0.0)  # 限制范围 [0, 1]

    # 状态转移概率
    probs = {}

    if w == 0:
        # 初始状态，只可能转移到 w 和 w+1
        probs[w] = (1 - p_succ)
        probs[w + 1] = p_succ
    elif w == W:
        # 最大状态，只可能转移到 w 和 w-1
        probs[w] = p_succ
        probs[w - 1] = (1 - p_succ)
    else:
        # 中间状态
        probs[w - 1] = (w / W) * p_succ
        probs[w] = (w / W) * (1 - p_succ) + ((W - w) / W) * p_succ
        probs[w + 1] = ((W - w) / W) * (1 - p_succ)

    return probs


############################################
# 4. 搭建 Pyomo 模型
############################################

def build_robust_stochastic_game_model():
    model = pyo.ConcreteModel("RobustGame_NoMinExpr")

    # 4.1) 决策变量
    #     给 xAtt, xIDS 加 bounds=(0,1)
    model.xAtt = pyo.Var(States, AttackerActions, bounds=(0,1), domain=pyo.NonNegativeReals)
    model.xIDS = pyo.Var(States, IDSactions,      bounds=(0,1), domain=pyo.NonNegativeReals)

    model.VAtt = pyo.Var(States, domain=pyo.Reals)
    model.VIDS = pyo.Var(States, domain=pyo.Reals)

    # 辅助变量: attacker_minVal / ids_maxVal
    model.attacker_minVal = pyo.Var([(w,a1,a2) 
                                    for w in States
                                    for a1 in AttackerActions
                                    for a2 in IDSactions],
                                    domain=pyo.Reals)
    model.ids_maxVal = pyo.Var([(w,a1,a2)
                                for w in States
                                for a1 in AttackerActions
                                for a2 in IDSactions],
                                domain=pyo.Reals)

    # 4.2) 约束: sum_{a1} xAtt[w,a1] ==1, sum_{a2} xIDS[w,a2]==1
    def attacker_prob_sum_rule(m, w):
        return sum(m.xAtt[w,a1] for a1 in AttackerActions) == 1
    model.attacker_prob_sum_con = pyo.Constraint(States, rule=attacker_prob_sum_rule)

    def ids_prob_sum_rule(m, w):
        return sum(m.xIDS[w,a2] for a2 in IDSactions) == 1
    model.ids_prob_sum_con = pyo.Constraint(States, rule=ids_prob_sum_rule)

    # 4.3) attacker_minVal <= sum_{w'}(prob*w') for every pD
    def attacker_minVal_con_rule(m, w, a1, a2, i):
        pd = p_candidates[i]
        trans = transition_prob(w,a1,a2,pd)
        scenario_val = sum(prob * m.VAtt[wn] for (wn,prob) in trans.items())
        return m.attacker_minVal[w,a1,a2] <= scenario_val

    model.attacker_minVal_con = pyo.Constraint(
        [(w,a1,a2,i)
         for w in States
         for a1 in AttackerActions
         for a2 in IDSactions
         for i in range(len(p_candidates))],
        rule=attacker_minVal_con_rule
    )

    # 4.4) ids_maxVal >= sum_{w'}(prob*w') for every pD
    def ids_maxVal_con_rule(m, w, a1, a2, i):
        pd = p_candidates[i]
        trans = transition_prob(w,a1,a2,pd)
        scenario_val = sum(prob * m.VIDS[wn] for (wn,prob) in trans.items())
        return m.ids_maxVal[w,a1,a2] >= scenario_val

    model.ids_maxVal_con = pyo.Constraint(
        [(w,a1,a2,i)
         for w in States
         for a1 in AttackerActions
         for a2 in IDSactions
         for i in range(len(p_candidates))],
        rule=ids_maxVal_con_rule
    )

    # 4.5) Bellman-like 不等式
    def attacker_value_rule(m, w):
        sum_expr = []
        for a1 in AttackerActions:
            for a2 in IDSactions:
                imm = attacker_payoff(w,a1,a2)
                val = imm + delta*m.attacker_minVal[w,a1,a2]
                prob= m.xAtt[w,a1]*m.xIDS[w,a2]
                sum_expr.append(prob*val)
        return m.VAtt[w] >= sum(sum_expr)
    model.attacker_value_con = pyo.Constraint(States, rule=attacker_value_rule)

    def ids_value_rule(m, w):
        sum_expr = []
        for a1 in AttackerActions:
            for a2 in IDSactions:
                imm = ids_cost(w,a1,a2)
                val = imm + delta*m.ids_maxVal[w,a1,a2]
                prob= m.xAtt[w,a1]*m.xIDS[w,a2]
                sum_expr.append(prob*val)
        return m.VIDS[w] >= sum(sum_expr)
    model.ids_value_con = pyo.Constraint(States, rule=ids_value_rule)

    # 4.6) 目标函数
    def obj_rule(m):
        # minimize sum_{w} [ VIDS[w] - VAtt[w] ]
        return sum(m.VIDS[w] - m.VAtt[w] for w in States)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model




############################################
# 5. 主流程
############################################

if __name__ == "__main__":
    model = build_robust_stochastic_game_model()

    solver = pyo.SolverFactory('ipopt')
    # 可以试试 'glpk'、'cbc' (不过它们是LP/MIP solver, 
    # 而这里有非线性的Bellman方程, 需要NLP solver, 
    # 'ipopt'或'couenne'或'bonmin'之类).
    solver.options['max_iter'] = 3000
    solver.options['tol'] = 1e-5
    solver.options['acceptable_tol'] = 1e-3
    solver.options['linear_solver'] = 'mumps'
    # 可以 solver.options['print_level'] = 5 看更详细日志

    result = solver.solve(model, tee=True)

    result = solver.solve(model, tee=True)
    print(result)

    # 打印结果
    print("\n=== Attacker's strategy (xAtt) ===")
    for w in States:
        lowA = pyo.value(model.xAtt[w,0])
        highA= pyo.value(model.xAtt[w,1])
        print(f" w={w}, (low={lowA:.4f}, high={highA:.4f}) sum={lowA+highA:.4f}")

    print("\n=== IDS's strategy (xIDS) ===")
    for w in States:
        lowI = pyo.value(model.xIDS[w,0])
        highI= pyo.value(model.xIDS[w,1])
        print(f" w={w}, (low={lowI:.4f}, high={highI:.4f}) sum={lowI+highI:.4f}")

    print("\n=== Values: VAtt, VIDS ===")
    for w in States:
        vA = pyo.value(model.VAtt[w])
        vI = pyo.value(model.VIDS[w])
        print(f" w={w}, VAtt={vA:.2f}, VIDS={vI:.2f}")
