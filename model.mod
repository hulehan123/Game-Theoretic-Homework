############################################
# model.mod / data.dat  (AMPL Model)
# ------------------------------------------
############################################

# =============================
# 1. 参数定义 (param)
# =============================

# W 表示状态空间最大值(比如最多被攻陷节点数)，
#   所以 w in 0..W 即可枚举所有状态。
param W;

# alpha, beta 用于 inst_cost[w] = alpha*(exp(beta*w)-1)，
#   表示随 w 增加而指数增长的系统损失/开销。
param alpha;
param beta;

# inst_cost[w] 即“环境”或“系统”随着被攻陷节点 w 的增多而增加的损失。
param inst_cost {w in 0..W} = alpha * (exp(beta*w)-1);

# Ch1, Ch2, Cl1, Cl2 分别表示在入侵者/IDS选择(high/low)动作时的常数收益或成本。
#   例如 Ch1, Cl1 针对入侵者 (高攻/低攻)，Ch2, Cl2 针对 IDS (高扫描/低扫描)。
param Ch1;
param Ch2;
param Cl1;
param Cl2;

# discount 为折扣因子(0<discount<1), 表示无限期博弈的未来收益折扣。
param discount;

# actions 集合：入侵者/IDS 各有两个动作：low(低) 或 high(高)。
set actions;

# s1[a], s2[b] 在有些版本里代表入侵者/IDS各动作的速率(或扫描/攻击强度)。
#   这里不一定用到，但可以用于计算 defend_probab。
param s1 {actions};
param s2 {actions};

# Pmin, Pmax, Pd：检测率/防御率的最小值、最大值和名义值(或当前值)。
param Pmin;
param Pmax; 
param Pd;

# 如果想让 Pd 在 [Pmin,Pmax] 间浮动，可取消注释:
# param Pd = Uniform(Pmin,Pmax); 
# var Pd <= Pmax, >=Pmin;

# =====================================
# 2. 定义即时收益(chi1, chi2)与转移概率
# =====================================

# Player1 = 入侵者(Attacker)
# Player2 = IDS(Defender)

# chi1[w,a,b]：在状态 w 下，当入侵者动作=a, IDS动作=b 时, 入侵者的即时收益。
#   这里使用 -inst_cost[w] + (if a="high" then Ch1 else Cl1)
#   表示系统损失越大时, 对入侵者收益越不利(取负)。
param chi1 {w in 0..W, a in actions, b in actions} 
    = -inst_cost[w] 
      + (if a="high" then Ch1 else Cl1);

# chi2[w,a,b]：类似，表示在状态 w 下, IDS 的即时收益或负成本。
#   用 inst_cost[w] + (if b="high" then Ch2 else Cl2)
param chi2 {w in 0..W, a in actions, b in actions}
    = inst_cost[w] 
      + (if b="high" then Ch2 else Cl2);

# defend_probab[a,b]：入侵者/IDS 动作组合 (a,b) 时的“防御成功率” (或检测成功率)
#   若 s1[a]=0 (如 a=low 攻击?), 则赋值=1
#   否则 min(0.16*Pd*s2[b]/s1[a],1)，代表防御率上限为1。
param defend_probab {a in actions, b in actions}
    = ( if s1[a] =0 
        then 1 
        else min( 0.16*Pd*s2[b]/s1[a], 1) );

# transition_probab[w,w_next,a,b]：从状态 w 跳转到 w_next 的概率，
#   当入侵者= a, IDS= b，
#   根据 defend_probab[a,b] (成功防御导致 w->w-1？) 等逻辑。
param transition_probab {w in 0..W, w_next in 0..W, a in actions, b in actions} =  
    ( if w_next = w-1 then (w*defend_probab[a,b]/W) 
      else if w_next = w then ( w*(1-defend_probab[a,b])/W  
                                + (W-w)*defend_probab[a,b]/W )
      else if w_next = w+1 then ( (W-w)*(1-defend_probab[a,b])/W )
      else 0 );

# =============================
# 3. 变量定义 (var)
# =============================

# sigma[i,w,a]: 第 i 个玩家(1=Attacker,2=IDS)在状态 w 下，对动作 a 的混合策略概率。
#   初始化为0.5
var sigma {i in 1..2, w in 0..W, a in actions} := 0.5;

# value[i,w]: 第 i 个玩家在状态 w 下的(长期)价值函数(折扣后的期望收益)。
var value {i in 1..2, w in 0..W};

# C[i,w,a,b]: 用来表示在状态 w 时, 如果 i=1(Attacker)则= chi1[w,a,b]*sigma[2,w,b],
#   如果 i=2(IDS)则= chi2[w,a,b]*sigma[1,w,a]。
#   这样可以方便构造期望收益表达式。
var C { i in 1..2, w in 0..W, a in actions, b in actions } 
    = ( if i=1 
        then chi1[w,a,b]*sigma[2,w,b] 
        else chi2[w,a,b]*sigma[1,w,a] );

# transitn_probab_total[w,w_next]: 从状态 w 跳转到 w_next 的总概率(考虑双方策略)。
#   就是对所有 (a,b) 做加权 sum( sigma[1,w,a]* sigma[2,w,b] * transition_probab[w,w_next,a,b] )。
var transitn_probab_total {w in 0..W, w_next in 0..W}
    = sum { a in actions, b in actions } 
          ( sigma[1,w,a]*sigma[2,w,b] * transition_probab[w,w_next,a,b] );


# =============================
# 4. 目标函数 (minimize)
# =============================

# total_val: 这里把“玩家的即时期望 + discount * 后续价值 - value[i,w]”之和做一个最小化，
#  让 solver 来找一个自洽解(类似Bellman方程)。
minimize total_val : 
    sum {i in 1..2, w in 0..W}
        (  sum {a in actions, b in actions}
             ( C[i,w,a,b]*(if i =1 then sigma[i,w,a] else sigma[i,w,b]) )
         +  discount* sum { w_next in 0..W}
                ( transitn_probab_total[w, w_next] * value [i,w_next] )
         -  value[i,w] );


# =============================
# 5. 约束 (subject to)
# =============================

# indiv_value: “个体价值”不等式(类似Bellman约束)，
#   表示 value[i,w] <= 期望即时收益 + discount * 期望下一状态价值
#   具体： value[i,w] <= sum_{b}( chi1 or chi2 ) + discount * sum_{w_next}(... value[i,w_next])
subject to indiv_value {i in 1..2, w in 0..W, a in actions}:
    value [i,w] <=  
        sum {b in actions} ( if i=1 then C[i,w,a,b] else C[i,w,b,a] )
      + discount* sum { w_next in 0..W }
        (  ( sum {b in actions}
             ( transition_probab[w, w_next,a,b] * (if i=1 then sigma[2,w,b] else sigma[1,w,b]) )
           )
         * value [i,w_next] ) ;


# probab_sum: 确保对任意(i,w), sum_{a} sigma[i,w,a] = 1
subject to probab_sum { i in 1..2, w in 0..W } :
    sum {a in actions} sigma[i,w,a] = 1 ;

# nonnegative: 确保 0 <= sigma[i,w,a] <= 1
subject to nonnegative {i in 1..2, w in 0..W, a in actions} :
    1>= sigma[i,w,a] >= 0;


# ~~~ 结束 ~~~


