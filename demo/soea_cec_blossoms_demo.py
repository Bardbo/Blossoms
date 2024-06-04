import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import geatpy as ea
import opfunu
from algorithms.soea_Blossoms_templet import AlgorithmNode, AlgorithmParam, soea_Blossoms_templet
from algorithms.soea_bridge2mealpy_templet import soea_bridge2mealpy_templet
from algorithms.soea_GWO_templet import soea_GWO_templet
from algorithms.soea_PSO_templet import soea_PSO_templet
from algorithms.soea_SA_templet import soea_SA_templet


"""
使用blossoms协同算法运行CEC测试问题的示例
支持自定义算法和桥接的mealpy库中的算法
"""

# 通过opfunu构建CEC测试问题
FUN_NAME = "F12017"
ndim = 10

funcs = opfunu.get_functions_by_classname(FUN_NAME)
func = funcs[0](ndim=ndim)
@ea.Problem.single
def evalVars(Vars):  # 定义目标函数
    f = func.evaluate(Vars)
    return f

problem = ea.Problem(name='soea_cec_blossoms_demo',
                        M=1,  # 目标维数
                        maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                        Dim=ndim,  # 决策变量维数
                        varTypes=[0] * ndim,  # 决策变量的类型列表，0：实数；1：整数
                        lb=[-100] * ndim,  # 决策变量下界
                        ub=[100] * ndim,  # 决策变量上界
                        evalVars=evalVars)

# 单个节点算法运行
# nodeA = AlgorithmNode(problem, "A", ea.soea_DE_best_1_L_templet, {"NIND":10, "MAXGEN":20})
# nodeA.run()
# print(nodeA.trace)

# 设置算法节点, 当前设置三个节点 AlgorithmParam = namedtuple("AlgorithmParam", ["node_name", "algorithm", "algorithm_param"])
nodes = [
    AlgorithmParam("A", ea.soea_DE_best_1_bin_templet, {"NIND":10, "MAXGEN":20}),
    AlgorithmParam("B", soea_GWO_templet, {"NIND":20, "MAXGEN":30}),
    AlgorithmParam("C", soea_bridge2mealpy_templet, {"NIND":20, "MAXGEN":30, 
                                                     "optimizer_name":"OriginalPSO", 
                                                     "fake_geatpy":False, 
                                                     "log_to":None}),
]

# 设置算法节点的连接方式, 当前三个节点的连接方式表示节点C要等节点AB运行后才运行, 但如果是串行模式则会忽略该参数
# 当edges为None时, 默认节点并行
edges = None
# edges = [
#     ("A", "C"),
#     ("B", "C"),
# ]

# 是否是链式执行各算法节点(串行)
isChain = True

if isChain:
    blossoms_algorithm = soea_Blossoms_templet(problem, ea.Population(Encoding="RI", NIND=50), nodes, edges, MAXGEN=100, mode="chain")
else:
    blossoms_algorithm = soea_Blossoms_templet(problem, ea.Population(Encoding="RI", NIND=50), nodes, edges, MAXGEN=100)

res = ea.optimize(blossoms_algorithm, saveFlag=False)