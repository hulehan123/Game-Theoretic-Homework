#################################################
# (D) Pyomo模型主体
#################################################
import pyomo.environ as pyo
from config import *
from rewards import *
from trans import *

def build_robust_stochastic_game_model():
    """
    构建一个简化版(示例)的鲁棒随机对策模型, 包含:
      - 混合策略变量
      - 价值函数变量
      - 不等式(简化的Bellman-like)约束
      - 目标函数(为了让solver搜到某个可行解)
    返回: 一个 Pyomo ConcreteModel
    """
    model = pyo.ConcreteModel("RobustStochasticGame")

    # 1) 定义决策变量
    model.xAtt = pyo.Var(States, AttackerActions, domain=pyo.NonNegativeReals)
    model.xIDS = pyo.Var(States, IDSactions,      domain=pyo.NonNegativeReals)

    # VAtt[w], VIDS[w]: 在状态 w 时, 入侵者/IDS 的鲁棒期望价值
    model.VAtt = pyo.Var(States, domain=pyo.Reals)
    model.VIDS = pyo.Var(States, domain=pyo.Reals)

    # 2) 每个状态下, 混合策略概率之和 = 1
    def attacker_prob_sum_rule(m, w):
        return sum(m.xAtt[w,a1] for a1 in AttackerActions) == 1
    model.attacker_prob_sum_con = pyo.Constraint(
        States, rule=attacker_prob_sum_rule
    )

    def ids_prob_sum_rule(m, w):
        return sum(m.xIDS[w,a2] for a2 in IDSactions) == 1
    model.ids_prob_sum_con = pyo.Constraint(
        States, rule=ids_prob_sum_rule
    )

    # 3) 定义鲁棒Bellman-like不等式(简化版)
    #    对入侵者: 
    #    VAtt[w] >= sum_{a1,a2} xAtt[w,a1]*xIDS[w,a2] * [ immediate + delta * worstNextVal ]
    #    worstNextVal = min_{pD \in [p_min,p_max]} E_{w'}[VAtt[w']] 
    #  (如果想做“最有利”, 就取 max; 论文中 attacker/IDS对不确定性的处理略有区别)
    p_candidates = [p_min, p_max]  # 离散近似
    
    def attacker_value_rule(m, w):
        sums_a1_a2 = []
        for a1 in AttackerActions:
            for a2 in IDSactions:
                immediate = attacker_payoff(w,a1,a2)
                # 对不确定性: attacker对检测率最坏的情况 => min_{pD} ...
                next_val_candidates = []
                for pd in p_candidates:
                    trans = transition_prob(w,a1,a2,pd)
                    val = 0.0
                    for wn, prob in trans.items():
                        val += prob * m.VAtt[wn]
                    next_val_candidates.append(val)
                worst_next_val = pyo.min_expression(next_val_candidates)

                combo_value = immediate + delta * worst_next_val
                prob_combo  = m.xAtt[w,a1] * m.xIDS[w,a2]
                sums_a1_a2.append(prob_combo * combo_value)

        return m.VAtt[w] >= sum(sums_a1_a2)
    model.attacker_value_con = pyo.Constraint(States, rule=attacker_value_rule)

    # 对IDS: VIDS[w] >= sum_{a1,a2} xAtt[w,a1]*xIDS[w,a2] 
    # [ cost + delta * (对IDS最坏的pD => max_{pD} E_{w'} VIDS[w'] ) ]
    def ids_value_rule(m, w):
        sums_a1_a2 = []
        for a1 in AttackerActions:
            for a2 in IDSactions:
                immediate = ids_cost(w,a1,a2)
                next_val_candidates = []
                for pd in p_candidates:
                    trans = transition_prob(w,a1,a2,pd)
                    val = sum(prob * m.VIDS[wn] for wn, prob in trans.items())
                    next_val_candidates.append(val)
                # IDS的worst case(花费更大) => max
                worst_next_val = pyo.max_expression(next_val_candidates)

                combo_value = immediate + delta * worst_next_val
                prob_combo  = m.xAtt[w,a1] * m.xIDS[w,a2]
                sums_a1_a2.append(prob_combo * combo_value)

        return m.VIDS[w] >= sum(sums_a1_a2)
    model.ids_value_con = pyo.Constraint(States, rule=ids_value_rule)

    # 4) 定义一个目标函数(非论文原始做法，而是为了让求解器找解):
    #    这里示例: minimize \sum (VIDS[w] - VAtt[w])
    def obj_rule(m):
        return sum(m.VIDS[w] - m.VAtt[w] for w in States)
    model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    return model
