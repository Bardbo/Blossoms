import geatpy as ea
import opfunu

"""
使用geatpy运行CEC测试问题的示例
"""

# 通过opfunu构建CEC测试问题
FUN_NAME = "F12022"
ndim = 10

funcs = opfunu.get_functions_by_classname(FUN_NAME)
func = funcs[0](ndim=ndim)
@ea.Problem.single
def evalVars(Vars):  # 定义目标函数
    f = func.evaluate(Vars)
    return f

problem = ea.Problem(name='soea_cec_demo',
                        M=1,  # 目标维数
                        maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                        Dim=ndim,  # 决策变量维数
                        varTypes=[0] * ndim,  # 决策变量的类型列表，0：实数；1：整数
                        lb=[-100] * ndim,  # 决策变量下界
                        ub=[100] * ndim,  # 决策变量上界
                        evalVars=evalVars)


# 构建算法
algorithm = ea.soea_SGA_templet(
                                problem,
                                ea.Population(Encoding='RI', NIND=50),
                                MAXGEN=2,  # 最大进化代数。
                                # MAXTIME=2, # 最大可优化时间
                                # MAXEVALS=10000,
                                logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                                )

# 求解
res = ea.optimize(algorithm, seed=42, verbose=True, drawing=1, outputMsg=True, saveFlag=False)