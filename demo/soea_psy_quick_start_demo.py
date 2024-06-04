import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from psy_optimize import optimize
from algorithms.soea_psy_GWO_templet import soea_psy_GWO_templet

import geatpy as ea
import numpy as np

"""
单目标带约束问题的快速入门简单示例的多染色体版本
"""

# 构建问题
r = 1  # 模拟该案例问题计算目标函数时需要用到的额外数据

@ea.Problem.single
def evalVars(Vars):  # 定义目标函数（含约束）
    f = np.sum((Vars - r) ** 2)  # 计算目标函数值
    cv = 0 if Vars[0] + Vars[1] > 1.5 else 1
    return f, cv

problem = ea.Problem(
    name='soea psy quick start demo',
    M=1,  # 目标维数
    maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
    Dim=5,  # 决策变量维数
    varTypes=[0, 0, 1, 1, 1],  # 决策变量的类型列表，0：实数；1：整数
    lb=[-1, 1, 0, 0, 0],  # 决策变量下界
    ub=[1, 4, 2, 1, 1],  # 决策变量上界
    evalVars=evalVars)
# 构建算法
algorithm = ea.soea_psy_SEGA_templet(
    problem,
    ea.PsyPopulation(Encodings=['RI', 'P'], NIND=20, EncoIdxs=[[0, 1, 2], [3, 4]]),
    MAXGEN=50,  # 最大进化代数。
    logTras=1,  # 表示每隔多少代记录一次日志信息，0表示不记录。
    trappedValue=1e-6,  # 单目标优化陷入停滞的判断阈值。
    maxTrappedCount=10)  # 进化停滞计数器最大上限值。

# 求解
res = optimize(algorithm, seed=None, verbose=True, drawing=1, outputMsg=True, saveFlag=False)