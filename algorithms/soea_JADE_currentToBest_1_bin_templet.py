# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from geatpy.core.xovbd import xovbd


class soea_JADE_currentToBest_1_bin_templet(ea.SoeaAlgorithm):
    """
soea_JADE_currentToBest_1_bin_templet : class - 差分进化变体JADE/current-to-best/1/bin算法算法类

算法描述:
    本算法类实现的是JADE/current-to-best/1/bin单目标差分进化算法。算法流程如下:
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 采用current-to-best的方法选择差分变异的各个向量,对当前种群进行差分变异,得到变异个体。
    5) 将当前种群和变异个体合并，采用二项式分布交叉方法得到试验种群。
    6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    7) 回到第2步。
    JADE的变异与原始变异方法不同,Xbest会从适应度排名前100p%中随机选取,Xr2从P∪A中随机选取,
    其中P是当前种群,A是一个受维护的历代被淘汰父代个体组成的种群;
    Cr和F均可以自适应。

参考文献:
    [1] Zhang, J. and A. C. Sanderson (2009). "JADE: Adaptive Differential Evolution With Optional 
    External Archive." IEEE Transactions on Evolutionary Computation 13(5): 945-958.

"""

    def __init__(self,
                 problem,
                 population,
                 MAXGEN=None,
                 MAXTIME=None,
                 MAXEVALS=None,
                 MAXSIZE=None,
                 logTras=None,
                 verbose=None,
                 outFunc=None,
                 drawing=None,
                 trappedValue=None,
                 maxTrappedCount=None,
                 dirName=None,
                 **kwargs):
        # 先调用父类构造方法
        super().__init__(problem, population, MAXGEN, MAXTIME, MAXEVALS, MAXSIZE, logTras, verbose, outFunc, drawing, trappedValue, maxTrappedCount, dirName)
        if population.ChromNum != 1:
            raise RuntimeError('传入的种群对象必须是单染色体的种群类型。')
        self.name = 'JADE'
        self.Cr = 0.5
        self.F = 0.5
        self.p = 0.05 # 最优个体选择的百分比 [0.05, 0.2]
        self.c = 0.05 # 自适应计算中的常数 [0.05, 0.2]
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        population.initChrom(NIND)  # 初始化种群染色体矩阵
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        self.call_aimFunc(population)  # 计算种群的目标函数值
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        pbest_nums = max(1, int(NIND * self.p))
        A = ea.Population(population.Encoding, population.Field, 0)
        r0 = np.arange(NIND)
        # ===========================开始进化============================
        while not self.terminated(population):
            # 进行差分进化操作
            r_best = np.argpartition(-population.FitnV.ravel(), pbest_nums)[:pbest_nums]
            r_best = np.random.choice(r_best, NIND)
            r1 = np.array([np.random.choice(r0[r0 != i], 1) for i in range(NIND)]).flatten()
            PandA = population + A if A.sizes else population
            r2 = np.arange(PandA.sizes)
            r2 = np.array([np.random.choice(r2[(r2 != i) & (r2 != j)], 1) for i,j in zip(r0, r1)]).flatten()
            F = self.F + 0.1 * np.random.standard_cauchy(NIND)
            while (F > 0).sum() < NIND:
                F = np.concatenate((F, self.F + 0.1 * np.random.standard_cauchy(NIND)))
            F = F[np.where(F > 0)[0][:NIND]].reshape(-1, 1)
            F = np.clip(F, None, 1)
            Cr = np.random.normal(self.Cr, 0.1, NIND)
            Cr = np.clip(Cr, 0, 1)
            experimentPop = ea.Population(population.Encoding, population.Field, NIND)  # 存储试验个体
            experimentPop.Chrom = population.Chrom + F * (population.Chrom[r_best] - population.Chrom) + \
                F * (population.Chrom[r1] - PandA.Chrom[r2]) # 变异
            experimentPop.Chrom = ea.boundfix(population.Encoding, experimentPop.Chrom, population.Field)
            for i, pi, ei, Cri in zip(r0, population.Chrom, experimentPop.Chrom, Cr):
                experimentPop.Chrom[i] = xovbd(np.vstack((pi, ei)), Cri, True, None, None) # 重组
            experimentPop.Chrom = ea.boundfix(population.Encoding, experimentPop.Chrom, population.Field) 
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            worse_ind = np.where(tempPop.FitnV[:NIND] < tempPop.FitnV[NIND:])[0]
            population = tempPop[r0]
            if worse_ind.size:
                A = A + population[worse_ind] if A.sizes else population[worse_ind]
                population[worse_ind] = tempPop[worse_ind+NIND]
                Sf, Scr = F[worse_ind].flatten(), Cr[worse_ind]
                self.F = (1-self.c) * self.F + self.c * (Sf ** 2).sum() / Sf.sum()
                self.Cr = (1-self.c) * self.Cr + self.c * (Scr ** 2).sum() / Scr.sum()
            if A.sizes > NIND:
                A_ind = np.arange(A.sizes)
                np.random.shuffle(A_ind)
                A = A[A_ind[:NIND]]
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
