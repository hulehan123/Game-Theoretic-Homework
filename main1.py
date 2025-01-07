# main.py

import pyomo.environ as pyo

############################################
# 1. 全局参数
############################################

W = 2
States = range(W+1)

delta = 0.9

# 可以先只用 p_nominal=0.8 测试名义情况, 然后再切换到 p_candidates=[p_min,p_max].
p_min, p_max = 0.77, 0.83
# p_candidates = [0.8]  # <-- 名义场景（无区间不确定）
p_candidates = [p_min, p_max]  # <-- 打开鲁棒区间

# 动作集
AttackerActions = [0,1]  # 0->low attack, 1->high attack
IDSactions      = [0,1]  # 0->low scan,   1->high scan

# 缩减一下常量, 避免出现过大值
C_H1 = 4.0  # 入侵者高攻击时的收益(调小一点)
C_L1 = 1.0  # 入侵者低攻击时的收益
C_H2 = 2.0  # IDS高扫描的开销(原是4)
C_L2 = 0.5  # IDS低扫描的开销(原是1)

# c(w) 改成 w 而非 3.0*w, 减少量级
def c_w(w):
    return w

############################################
# 2. 即时收益/成本函数
############################################

def attacker_payoff(w, a1, a2):
    """
    对应论文(18), 这里只是一个较小的示例版本
    """
    if a1 == 1:
        base = C_H1  # high attack
    else:
        base = C_L1  # low attack

    # 若IDS高扫描, 入侵者收益被额外扣一些
    if a2 == 1:
        base -= (C_H2 / 2.0)  # 让它负一点
    else:
        base += 0.2  # IDS低扫描, 再给一点好处

    # 让它跟 w 也稍微挂钩
    # e.g. w越大, 入侵者可能更容易利用, + 0.05*w
    payoff = base + 0.05 * w
    return payoff

def ids_cost(w, a1, a2):
    """
    对应论文(19), 同样简化
    """
    cost = c_w(w)

    # IDS高扫描 => 多开销
    if a2 == 1:
        cost += C_H2
    else:
        cost += C_L2

    # 如果入侵者在高攻击
    if a1 == 1:
        cost += 1.0

    return cost

############################################
# 3. 状态转移概率
############################################

def transition_prob(w, a1, a2, pD):
    """
    参考论文(20),(21). 此处仍是演示写法.
    """
    # 基础成功率
    p_succ = 0.5
    if a1 == 1:  # high attack
        p_succ += 0.2
    if a2 == 1:  # high scan
        p_succ -= 0.2

    # 考虑 pD
    p_succ -= (pD - 0.5)*0.4
    # clip
    p_succ = max(min(p_succ, 1.0), 0.0)

    probs = {}
    if w == 0:
        # w->w+1 = p_succ,  w->w = (1 - p_succ)
        probs[w+1] = p_succ
        probs[w]   = 1.0 - p_succ
    elif w == W:
        # w->w-1 = (1 - p_succ), w->w = p_succ
        probs[w-1] = 1.0 - p_succ
        probs[w]   = p_succ
    else:
        # w->w+1 = p_succ * 0.8, w->w-1 = (1-p_succ)*0.8, w->w=0.2 ...
        # 你可按论文(21)更精准写. 这里就演示.
        # 先来个最简单: w->w+1 = p_succ, w->w-1 = 0.2*(1-p_succ), w->w = 0.8*(1-p_succ)
        # 这样保证和=1
        p_fail = 1.0 - p_succ
        probs[w+1] = p_succ
        probs[w-1] = 0.2*p_fail
        probs[w]   = 0.8*p_fail

    return probs

############################################
# 4. 搭建 Pyomo 模型 (不使用 min_expression/max_expression)
############################################

def build_robust_stochastic_game_model():
    model = pyo.ConcreteModel("RobustGame_NoMinExpr")

    # 4.1) 决策变量
    #     给 xAtt, xIDS 加 bounds=(0,1)!
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

    # 4.6) 目标函数(示例)
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
    # 你也可以试试 'glpk'、'cbc' (不过它们是LP/MIP solver, 
    # 而这里有非线性的Bellman方程, 需要NLP solver, 
    # 'ipopt'或'couenne'或'bonmin'之类).

    result = solver.solve(model, tee=True)
    print(result)

    # 如果 solver 报 infeasible，可以逐条看提示，并尝试:
    # 1) 改 W=5
    # 2) 改 p_candidates = [0.8]
    # 3) 调小 c_w(w) or payoff 的系数
    # 4) 改换 solver

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
