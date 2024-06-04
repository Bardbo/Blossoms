# 单个算法实验
import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from algorithms.soea_GWO_templet import soea_GWO_templet
from algorithms.soea_PSO_templet import soea_PSO_templet
from algorithms.soea_JADE_currentToBest_1_bin_templet import soea_JADE_currentToBest_1_bin_templet
from algorithms.soea_L_SHADE_currentToBest_1_bin_templet import soea_L_SHADE_currentToBest_1_bin_templet
from algorithms.soea_GSK_templet import soea_GSK_templet
from algorithms.soea_AGSK_templet import soea_AGSK_templet
from algorithms.soea_SSA_templet import soea_SSA_templet
from algorithms.soea_DBO_templet import soea_DBO_templet
from algorithms.soea_HO_templet import soea_HO_templet

import re
import inspect
import geatpy as ea
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import Pool, Manager
import opfunu
from opfunu.cec_based import *

CEC_YEAR = "2017"
ndims = [10, 30, 50, 100]
MAXEVALS = 10000
NIND = 50
USE_Algorithms = [soea_GWO_templet, soea_PSO_templet,
                  soea_JADE_currentToBest_1_bin_templet, soea_L_SHADE_currentToBest_1_bin_templet, 
                  soea_GSK_templet, soea_AGSK_templet,
                  soea_SSA_templet, soea_DBO_templet, soea_HO_templet]
runs = 51
FUN_NAMES = [member[0] for member in inspect.getmembers(cec2017) if re.match(f"F\d+{CEC_YEAR}", member[0])]
num_tasks = len(USE_Algorithms) * len(FUN_NAMES) * len(ndims) * runs

def run_algorithm_fun_dim(q, use_algorithm, fun_name, dim):
    print(f"{use_algorithm.__name__} {fun_name} {dim} start!")
    funcs = opfunu.get_functions_by_classname(fun_name)
    func = funcs[0](ndim=dim)
    @ea.Problem.single
    def evalVars(Vars):  # 定义目标函数（含约束）
        f = func.evaluate(Vars)
        return f

    problem = ea.Problem(name=CEC_YEAR,
                            M=1,  # 目标维数
                            maxormins=[1],  # 目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标
                            Dim=dim,  # 决策变量维数
                            varTypes=[0] * dim,  # 决策变量的类型列表，0：实数；1：整数
                            lb=[-100] * dim,  # 决策变量下界
                            ub=[100] * dim,  # 决策变量上界
                            evalVars=evalVars)
    
    runs_trace = dict()
    for i in range(runs):
        algorithm = use_algorithm(
                            problem,
                            ea.Population(Encoding='RI', NIND=NIND),
                            MAXEVALS=MAXEVALS,
                            logTras=0,  # 表示每隔多少代记录一次日志信息，0表示不记录。
                            )
        # 求解
        res = ea.optimize(algorithm, verbose=False, drawing=0, outputMsg=0, saveFlag=False)
    
        runs_trace[f"{fun_name}-{dim}-{i}"] = algorithm.trace["f_best"]
        q.put(f"{fun_name:<7} | {dim:<3} | {i:<2} | {res['ObjV'].item()}")
        
    if not os.path.exists(f"result\{algorithm.name}"):
        os.makedirs(f"result\{algorithm.name}")
    pd.DataFrame(runs_trace.values(), index=runs_trace.keys()).T.to_csv(f"result\{algorithm.name}\{fun_name}-{dim}-{runs}.csv", index=False)
    print(f"{use_algorithm.__name__} {fun_name} {dim} finished!")
    
    
if __name__ == '__main__':
    q = Manager().Queue()
    num_cores = int(mp.cpu_count())
    with Pool(int(num_cores)) as pool:
        pbar = tqdm(num_tasks)
        pool.starmap_async(run_algorithm_fun_dim, [(q, i, j, k) for i in USE_Algorithms for j in FUN_NAMES for k in ndims])
        count = 0
        while count != num_tasks:
            if not q.empty():
                pbar.update()
                pbar.set_description_str(f" {count:>4}/{num_tasks} | " + q.get() + "\n")
                count += 1
        pool.close()
        pool.join()
        pbar.close()
    
    print(f"{CEC_YEAR} all finished!")
    
        
        
            
