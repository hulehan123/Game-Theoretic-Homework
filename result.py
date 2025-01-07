#################################################
# (E) 调用求解器 & 查看输出
#################################################
import pyomo.environ as pyo
from model import *


def solve_and_report(model):
    solver = pyo.SolverFactory('ipopt')  # 可改成 'cplex', 'gurobi', 'loqo' 等
    result = solver.solve(model, tee=True)
    print(result)

    # 打印混合策略
    print("==== Attacker's stationary strategies ====")
    for w in States:
        p_low  = pyo.value(model.xAtt[w,0])
        p_high = pyo.value(model.xAtt[w,1])
        print(f"State {w}: (Low={p_low:.4f}, High={p_high:.4f})")

    print("\n==== IDS's stationary strategies ====")
    for w in States:
        p_low  = pyo.value(model.xIDS[w,0])
        p_high = pyo.value(model.xIDS[w,1])
        print(f"State {w}: (Low={p_low:.4f}, High={p_high:.4f})")

    print("\n==== Value functions ====")
    for w in States:
        vA = pyo.value(model.VAtt[w])
        vI = pyo.value(model.VIDS[w])
        print(f"State {w}: VAtt={vA:.4f}, VIDS={vI:.4f}")

#################################################
# main.py 或者在同一文件中直接写
#################################################

if __name__ == "__main__":
    # 1) 构建模型
    model = build_robust_stochastic_game_model()

    # 2) 求解
    solve_and_report(model)

    # 若要画类似论文Fig.3~6的图，可以在solve_and_report后追加：
    #  (1) 读取各 w 的策略 xAtt[w,1] => 入侵者“高攻击”概率 vs. w => 对应Fig.3
    #  (2) 读取 xIDS[w,1] => IDS“高扫描”概率 vs. w => 对应Fig.4
    #  (3) 计算或模拟平均被攻陷节点 vs. 不同区间 => Fig.5
    #  (4) 计算或比较双方收益 => Fig.6
    #  这里就不赘述了，你可用 matplotlib 做可视化
