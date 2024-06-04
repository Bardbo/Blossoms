import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from algorithms.soea_GWO_templet import soea_GWO_templet
from algorithms.soea_PSO_templet import soea_PSO_templet
from algorithms.soea_AFSA_templet import soea_AFSA_templet
from algorithms.soea_SA_templet import soea_SA_templet
from algorithms.soea_JADE_currentToBest_1_bin_templet import soea_JADE_currentToBest_1_bin_templet
from algorithms.soea_L_SHADE_currentToBest_1_bin_templet import soea_L_SHADE_currentToBest_1_bin_templet
from algorithms.soea_GSK_templet import soea_GSK_templet
from algorithms.soea_AGSK_templet import soea_AGSK_templet
from algorithms.soea_SSA_templet import soea_SSA_templet
from algorithms.soea_DBO_templet import soea_DBO_templet
from algorithms.soea_HO_templet import soea_HO_templet

import geatpy as ea
import numpy as np

"""
单目标带约束问题的快速入门简单示例
"""

# 构建问题
r = 1  # 目标函数需要用到的额外数据
@ea.Problem.single
def evalVars(Vars):  # 定义目标函数（含约束）
    f = np.sum((Vars - r) ** 2)  # 计算目标函数值
    cv = 0 if Vars[0] + Vars[1] > 1.5 else 1
    return f, cv

problem = ea.Problem(name='soea quick start demo',
                        M=1,  # 目标维数
                        maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                        Dim=5,  # 决策变量维数
                        varTypes=[0, 0, 0, 0, 0],  # 决策变量的类型列表，0：实数；1：整数
                        lb=[-10] * 4 + [2],  # 决策变量下界
                        ub=[10] * 5,  # 决策变量上界
                        evalVars=evalVars)

# 构建算法
algorithm = soea_HO_templet(problem,
                            ea.Population(Encoding='RI', NIND=50),
                            MAXGEN=20  # 最大进化代数。
                            )

# 求解
res = ea.optimize(algorithm, seed=None, verbose=True, drawing=1, outputMsg=True, saveFlag=False)

