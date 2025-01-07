import pyomo.environ as pyo
import math

#############################
# 1. 全局参数配置
#############################

# 最大可被攻陷节点数 (论文示例: W=30)
W = 30
# 状态集 Ω = {0,1,...,W}
States = range(W+1)

# 折扣因子 delta
delta = 0.9

# 不确定检测率区间 (p_min, p_max)
p_min = 0.74
p_max = 0.86
# 可以一次性尝试多组区间，比如 [0.79,0.81], [0.77,0.83], ...
# 在实验时可循环地改变这些值

# 入侵者(Attacker)的动作集合: a1=0 -> 低攻击, a1=1 -> 高攻击
AttackerActions = [0, 1]
# IDS的动作集合: a2=0 -> 低扫描, a2=1 -> 高扫描
IDSactions = [0, 1]

# 论文中示例的成本/收益常数：
# 比如 入侵者在状态 w 时行动 (a1=0/1) 并且 IDS 行动 (a2=0/1) 的即时收益
# 这里给一个可能的定义(仅供示例)。你也可以将其改为函数形式：
# chi^1(w,a1,a2) 以及 chi^2(w,a1,a2).
# -------------------------------------
# c(w) =  cost if w nodes are compromised, 论文里似乎有个 c(w)，可自行定义：
def c_w(w):
    # 在论文示例中有时令 c(w) 与 w 成正比, 也可做别的定义
    return 3.0 * w  # 示例：线性随 w 增长

# 常数参数 (可对照论文中类似 C^H_1, C^L_1, C^H_2, C^L_2 等)
C_H1 = 8.0   # 入侵者使用高攻击时的额外获益 (或IDS多损失)
C_L1 = 2.0   # 入侵者使用低攻击时的相对收益
C_H2 = 4.0   # IDS使用高扫描的成本(或者说 attacker 额外受损)
C_L2 = 1.0   # IDS使用低扫描的成本(或 attacker 多收益)...

# 入侵者即时收益函数 chi^1
def attacker_payoff(w, a1, a2):
    """
    a1 = 0/1 -> low/high attack
    a2 = 0/1 -> low/high scanning
    """
    # 下面是一个非常简化的示例逻辑, 仅演示写法
    # 你可以替换成论文里更精确的公式(见 18~19 式等)
    if a1 == 1:
        # high attack
        base = C_H1
    else:
        # low attack
        base = C_L1

    # 假设IDS的高扫描会减少入侵者收益
    if a2 == 1:
        # high scanning
        base = base - C_H2/2.0  # IDS高扫描压制了攻击者收益
    else:
        # low scanning
        base = base + 0.5       # IDS低扫描给攻击者一点加成

    # 这里还可以加入对 w 的依赖, e.g. 攻击越多节点可能越赚钱
    # 作为演示, 简单地加上 c_w(w)*0.01
    return base + 0.01*c_w(w)

# IDS即时成本(或负收益)函数 chi^2
def ids_cost(w, a1, a2):
    """
    a1 = 0/1 -> low/high attack
    a2 = 0/1 -> low/high scanning
    """
    # 同样示例写法:
    # IDS希望不让 w 太大, 但 scanning 也有代价
    cost = c_w(w)  # 当 w 大时, IDS认为安全风险高, cost 高

    if a2 == 1:
        # high scanning => cost maybe higher in terms of resource usage
        cost += C_H2
    else:
        cost += C_L2

    # 如果对方高攻击, cost再增加一点
    if a1 == 1:
        cost += 2.0

    return cost


# 状态转移概率:
# w -> w+1 : attacker成功攻陷新节点
# w -> w-1 : 之前被攻陷节点被recover
# boundary: w=0时, 无法再往w-1; w=W时, 无法再往w+1
# 论文中 (21) 式等: p( w -> w+1 ) = 1 - \tilde{p}, p( w -> w-1 ) = \tilde{p} ...
# 其中 \tilde{p} 依赖不确定检测率 p_D、a1, a2...
def transition_prob(w, a1, a2, pD):
    """
    返回一个 dict: {w': prob}, 对所有 w' in {0..W}
    pD 在 [p_min, p_max] 范围
    """
    # 在这里, 先根据 (a1,a2) 大致定义 attacker成功/失败的概率
    # 你可以对照论文Eq. (20)-(21)等, 更精确地写
    # 例如 p_success = 0.16 * pD^( a1 ) * ...
    # 这里我们简化演示:
    # attacker成功破坏 => w -> w+1
    # attacker失败/IDS成功检测 => w -> w-1
    # remain w => 0 (也可以自己定义)
    p_succ = 0.5  # base
    if a1 == 1:
        p_succ += 0.3  # high attack => success up
    if a2 == 1:
        p_succ -= 0.2  # high scanning => success down

    # 再因不确定检测率 pD 改变:
    # 如果 pD 大 => IDS强 => attacker成功率低
    # (仅做演示; 具体函数可自行定义)
    p_succ -= (pD - 0.5)*0.5

    # clip:
    p_succ = max(min(p_succ, 1.0), 0.0)

    # boundary check:
    probs = {}
    if w == 0:
        # w=0 时无法再 -1
        # 只能 stay or go to w+1
        probs[w+1] = p_succ
        probs[w]   = 1.0 - p_succ
    elif w == W:
        # w=W 时无法再 +1
        # 只能 stay or go to w-1
        probs[w-1] = (1.0 - p_succ)
        probs[w]   = p_succ
    else:
        # 一般情况
        probs[w+1] = p_succ
        probs[w-1] = (1.0 - p_succ)*0.5
        probs[w]   = (1.0 - p_succ)*0.5

    return probs


#############################
# 2. 用 Pyomo 搭建模型
#############################

model = pyo.ConcreteModel("Robust_Stochastic_Game")

# 为简化, 我们假设各状态下玩家的混合策略用 xAtt[w,a1], xIDS[w,a2] 表示.
# 并确保 sum_{a1} xAtt[w,a1] = 1, sum_{a2} xIDS[w,a2] = 1
model.xAtt = pyo.Var( States, AttackerActions, domain=pyo.NonNegativeReals )
model.xIDS = pyo.Var( States, IDSactions,      domain=pyo.NonNegativeReals )

# 我们还需要“价值函数”/“成本函数”变量, 表示在状态 w 的(鲁棒)价值:
# VAtt[w] = attacker在状态w的期望折扣收益(最优策略下)
# VIDS[w] = IDS在状态w的期望折扣成本
model.VAtt = pyo.Var( States, domain=pyo.Reals )
model.VIDS = pyo.Var( States, domain=pyo.Reals )

# 混合策略约束: sum_{a1} xAtt[w,a1] = 1, sum_{a2} xIDS[w,a2] = 1
def attacker_prob_sum_rule(m, w):
    return sum(m.xAtt[w,a1] for a1 in AttackerActions) == 1
model.attacker_prob_sum_con = pyo.Constraint( States, rule=attacker_prob_sum_rule )

def ids_prob_sum_rule(m, w):
    return sum(m.xIDS[w,a2] for a2 in IDSactions) == 1
model.ids_prob_sum_con = pyo.Constraint( States, rule=ids_prob_sum_rule )

# 为了演示鲁棒性, 我们需要对 pD in [p_min,p_max] 做 max-min。
# 在较简单(但不太高效)的做法中，可把 pD 当成一个对手“最坏的选择”
# 并引入辅助变量/不等式, 或者做二层循环(外层枚举pD,内层做Bellman更新)等。
# 论文中有更系统的对偶转换, 这里仅用一种可行的“离散近似”做示例:
p_candidates = [p_min, p_max]  # 只取端点近似
# 你也可以更细分区间多点离散, 但会使模型复杂度提高。

# 我们需要写“Bellman方程”形式:
# VAtt[w] >= InstantPayoff^Att(w,a1,a2) + delta * sum_{w'} prob( w->w' ) * VAtt[w']
# IDS 同理(只不过可能是cost最小化, 这里演示成 "cost = value" ).
# 由于是nonzero-sum, 各自的Bellman不等式要结合混合策略与最坏pD.
# 这里为了简化, 演示成: 
#    VAtt[w] >= sum_{a1} xAtt[w,a1] sum_{a2} xIDS[w,a2] [ \chi^1(w,a1,a2)
#       + delta * min_{pD in [p_min,p_max]}(  sum_{w'} P_{pD}(w->w'|a1,a2) * VAtt[w'] ) ]
# 并对所有 a1,a2, w 施加 KKT条件 or 取最大? 这是一个多层结构.
# 在完整实现中, 你要参照论文附录, 把内层 min_{pD} 通过线性对偶引入.
# 下列仅给出一个“内层枚举 p_min, p_max 并取最不利pD”的做法:

BIGM = 1e5  # 辅助大常数

def attacker_value_rule(m, w):
    """
    构造一个不等式: 
      VAtt[w] >= sum_{a1} sum_{a2} xAtt[w,a1]*xIDS[w,a2]*[ immediate payoff + delta * worstNextValue ]
    """
    # 先把各(a1,a2)的“worstNextValue”算出来(枚举p_min, p_max).
    # 以入侵者的视角, worstNextValue = min_{pD} E_{w'} [ VAtt[w'] ],
    # 因为不确定对入侵者不利(IDS会选最坏pD).
    # 当然, 如果是 attacker robust, 可能是 max_{pD}, 需看论文具体定义.
    # 这里演示成: attacker对不确定检测率最保守 => 取 min_{pD}.
    
    sums_a1_a2 = []
    for a1 in AttackerActions:
        for a2 in IDSactions:
            immediate = attacker_payoff(w,a1,a2)
            # 计算在 pD = p_min 与 pD = p_max 时, 期望后续价值
            next_val_candidates = []
            for pd in p_candidates:
                trans = transition_prob(w,a1,a2,pd)
                tmp_val = sum(trans[w_next]*m.VAtt[w_next] for w_next in trans.keys())
                next_val_candidates.append(tmp_val)
            # 取最小(最坏情况)
            worst_next_val = pyo.min_expression(next_val_candidates)
            
            # 该动作组合下的(即时收益 + 折扣 * worstNextVal)
            total_combo = immediate + delta * worst_next_val
            # 再乘以该动作组合被选中的概率
            prob_combo = m.xAtt[w,a1]*m.xIDS[w,a2]
            sums_a1_a2.append( prob_combo * total_combo )
    
    return m.VAtt[w] >= sum(sums_a1_a2)

model.attacker_value_con = pyo.Constraint( States, rule=attacker_value_rule )

# 对IDS的价值函数(或成本), 也做类似:
def ids_value_rule(m, w):
    """
    VIDS[w] >= sum_{a1,a2} xAtt[w,a1]*xIDS[w,a2] [ ids_cost(...) + delta*worstNextVal_ids ]
    (同理)
    """
    sums_a1_a2 = []
    for a1 in AttackerActions:
        for a2 in IDSactions:
            immediate = ids_cost(w,a1,a2)
            next_val_candidates = []
            for pd in p_candidates:
                trans = transition_prob(w,a1,a2,pd)
                tmp_val = sum(trans[w_next]*m.VIDS[w_next] for w_next in trans.keys())
                next_val_candidates.append(tmp_val)
            worst_next_val = pyo.max_expression(next_val_candidates)
            # 这里演示为 IDS 担心不确定性朝坏方向(使其cost更大), 于是 取 max
            total_combo = immediate + delta * worst_next_val
            sums_a1_a2.append( m.xAtt[w,a1]*m.xIDS[w,a2]*total_combo )
    
    return m.VIDS[w] >= sum(sums_a1_a2)

model.ids_value_con = pyo.Constraint( States, rule=ids_value_rule )

# 上面 attacker_value_rule / ids_value_rule 只是构造了
# VAtt[w],VIDS[w] 的“自洽”下界. 其实为了找到博弈均衡,
# 需要更多KKT或最优性条件(或借助论文中的多线性系统).
# 这里给出的仅是个简化“可行解”条件；要逼近真实均衡，还需其他约束。
# 完整实现请参考论文附录(Definition 2, Corollary 1~2等)把max-min(min-max)转化为
# 对偶LP or NLP, 并添加同伴偏离不可盈利的约束。

#############################
# 3. 定义(或近似定义)目标函数并求解
#############################

# 在nonzero-sum博弈中, 我们往往不是简单地“maximize VAtt + VIDS”.
# 论文做法：寻找满足上面那套多线性方程组(或不等式)的策略, 
# 使之构成Markov Perfect Equilibrium (MPE) in robust sense.  
# 这里演示一下: 我们可能让求解器 try “minimize sum_{w} ( - VAtt[w] + VIDS[w] ) ”
# 作为一个“引导”目标, 让它倾向于在可行集中找某种(平衡感)解。
# 这并非论文的标准做法(它用固定点定理 + KKT对偶).
# 如果想完整还原论文结果, 需要把他们的等式系统原封不动写成 constraints, 
# 并用 solver 求 feasibility or use a specialized MPE solver.

def obj_rule(m):
    return sum(m.VIDS[w] - m.VAtt[w] for w in States)  # IDS想最小, attacker想最大
model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

#############################
# 4. 调用求解器
#############################
solver = pyo.SolverFactory('ipopt')  # 也可换cplex, gurobi, etc.
result = solver.solve(model, tee=True)
print(result)

# 打印决策变量
print("------ Attacker's stationary strategies ------")
for w in States:
    # xAtt[w,0], xAtt[w,1]
    p_low = pyo.value(model.xAtt[w,0])
    p_high= pyo.value(model.xAtt[w,1])
    print(f"State w={w}: Attacker(Low,High)=({p_low:.4f},{p_high:.4f})")

print("\n------ IDS's stationary strategies ------")
for w in States:
    p_low = pyo.value(model.xIDS[w,0])
    p_high= pyo.value(model.xIDS[w,1])
    print(f"State w={w}: IDS(Low,High)=({p_low:.4f},{p_high:.4f})")

print("\n------ Value functions ------")
for w in States:
    vA = pyo.value(model.VAtt[w])
    vI = pyo.value(model.VIDS[w])
    print(f"State w={w}: VAtt={vA:.4f}, VIDS={vI:.4f}")
