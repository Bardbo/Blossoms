# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库
from geatpy.core.xovbd import xovbd


class soea_L_SHADE_currentToBest_1_bin_templet(ea.SoeaAlgorithm):
    """
soea_L_SHADE_currentToBest_1_bin_templet : class - 差分进化变体L-SHADE/current-to-best/1/bin算法算法类

算法描述:
    本算法类实现的是L-SHADE/current-to-best/1/bin单目标差分进化算法。算法流程如下:
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 采用current-to-best的方法选择差分变异的各个向量,对当前种群进行差分变异,得到变异个体。
    5) 将当前种群和变异个体合并，采用二项式分布交叉方法得到试验种群。
    6) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    7) 回到第2步。
    L-SHADE的变异与JADE相同;
    Cr和F均可以自适应, 用了加权;
    种群规模会线性缩减。

参考文献:
    [1] Tanabe R , Fukunaga A S .Improving the search performance of SHADE using linear 
    population size reduction[C]//Evolutionary Computation.IEEE, 2014.DOI:10.1109/CEC.2014.6900380.

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
        self.name = 'L-SHADE'
        self.p = 0.05 # 最优个体选择的百分比 [0.05, 0.2]
        self.H = 100
        self.Mcr = 0.5
        self.Mf = 0.5
        self.Nmin = 4 # 最小种群大小
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
        self.Mcr = self.Mcr * np.ones(self.H)
        self.Mf = self.Mf * np.ones(self.H)
        self.H = np.arange(self.H)
        pbest_nums = max(1, int(NIND * self.p))
        A = ea.Population(population.Encoding, population.Field, 0)
        k = 0
        self.Ninit = NIND
        self.max_eval = NIND * self.MAXGEN if self.MAXEVALS is None else self.MAXEVALS
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
            ri = np.random.choice(self.H, NIND)
            Cr = self.Mcr[ri]
            Cr = np.random.normal(Cr, 0.1) 
            Cr = np.clip(Cr, 0, 1)
            F = []
            for fi in self.Mf[ri]:
                while (Fi := fi + 0.1 * np.random.standard_cauchy()) < 0:pass
                F.append(Fi)
            F = np.clip(F, None, 1).reshape(-1, 1)
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
                delta_fitv = np.abs(tempPop.FitnV[worse_ind] - tempPop.FitnV[worse_ind+NIND]).flatten()
                A = A + population[worse_ind] if A.sizes else population[worse_ind]
                population[worse_ind] = tempPop[worse_ind+NIND]
                Sf, Scr = F[worse_ind].flatten(), Cr[worse_ind]
                w = delta_fitv / delta_fitv.sum()
                if self.Mcr[k] < 0 or max(Scr) == 0:
                    self.Mcr[k] = -np.inf
                else:
                    self.Mcr[k] = (w * Scr ** 2).sum() / (w * Scr).sum()
                self.Mf[k] = (w * Sf ** 2).sum() / (w * Sf).sum()
                k += 1
                if k > self.H[-1]:
                    k = 0
            population = population[population.FitnV.flatten().argsort()[::-1]]
            NIND = round((self.Nmin-self.Ninit) * self.evalsNum / self.max_eval + self.Ninit)
            r0 = np.arange(NIND)
            population = population[r0]
            if A.sizes > NIND:
                A_ind = np.arange(A.sizes)
                np.random.shuffle(A_ind)
                A = A[A_ind[:NIND]]
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
