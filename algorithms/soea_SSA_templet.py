# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库


class soea_SSA_templet(ea.SoeaAlgorithm):
    """
soea_SSA_templet : class - Sparrow Search Algorithm麻雀搜索算法类

算法描述:
    本算法类实现的是Sparrow Search Algorithm麻雀搜索算法。算法流程如下:
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 发现者更新。
    5) 加入者更新。
    6) 警戒者更新。
    7) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    8) 回到第2步。

参考文献:
    [1] XUE J K, ShEN B. A novel swarm intelligence optimization approach: sparrow search algorithm [J]. 
    Systems Science & Control Engineering, 2020, 8(1): 22-34.

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
        self.name = 'SSA'
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')
        self.PD = 0.2 # 生产者/发现者占比
        self.SD = 0.15 # 随机警戒者占比,一般为0.1-0.2
        self.ST = 0.7 # 安全值 [0.5, 1]

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
        # SSA参数
        self.PD = int(NIND * self.PD)
        self.SD = int(NIND * self.SD)
        self.PD_ind = np.arange(self.PD).reshape(-1, 1)
        self.FD_ind = np.arange(self.PD, NIND)
        self.n = len(self.FD_ind)
        self.epsilon = 1e-10
        dim = self.problem.Dim
        if self.MAXGEN is not None:
            MAXGEN = self.MAXGEN
        else:
            MAXGEN = 100 if self.MAXEVALS is None else np.ceil(self.MAXEVALS / NIND)
        # ===========================开始进化============================
        while not self.terminated(population):
            # 个体排序 最优到最差
            population = population[population.FitnV.flatten().argsort()[::-1]]
            Xbest = population.Chrom[0]
            Xworst = population.Chrom[-1]
            # 发现者位置更新
            PD = population[self.PD_ind]
            alpha = np.random.rand(self.PD, dim)
            R2 = np.random.rand()
            Q = np.random.normal(size=(self.PD, dim))
            if R2 < self.ST:
                PD.Chrom = PD.Chrom * np.exp(-self.PD_ind / (alpha * MAXGEN + self.epsilon))
            else:
                PD.Chrom = PD.Chrom + Q
            PD.Chrom = ea.boundfix(PD.Encoding, PD.Chrom, PD.Field)
            # 加入者/追随者位置更新
            FD = population[self.FD_ind]
            halfn = int(self.n / 2)
            Q = np.random.normal(size=(halfn, dim))
            FD.Chrom[halfn:] = Q * np.exp((Xworst - FD.Chrom[halfn:]) / (np.arange(halfn, self.n).reshape(-1, 1) ** 2 + self.epsilon))
            A = np.random.randint(0, 2, (halfn, dim))
            A[np.where(A == 0)] = -1
            FD.Chrom[:halfn] = Xbest + ((np.abs(FD.Chrom[:halfn] - Xbest) * A).sum(axis=1) / dim).reshape(-1, 1)
            FD.Chrom = ea.boundfix(FD.Encoding, FD.Chrom, FD.Field)
            # 警戒者位置更新
            experimentPop = PD + FD
            SD_ind = np.random.choice(np.arange(NIND), self.SD, replace=False)
            mask = experimentPop.FitnV.flatten()[SD_ind] != population[0].FitnV.flatten()
            beta = np.random.normal(size=(mask.sum(), dim))
            k = np.random.uniform(-1, 1, size=((~mask).sum(), dim))
            experimentPop.Chrom[SD_ind[mask]] = Xbest + beta * np.abs(experimentPop.Chrom[SD_ind[mask]] - Xbest)
            if len(k):
                experimentPop.Chrom[SD_ind[~mask]] = experimentPop.Chrom[SD_ind[~mask]] + \
                    k * np.abs(experimentPop.Chrom[SD_ind[~mask]] - Xworst) / ((population.FitnV[SD_ind[~mask]] - population[NIND-1].FitnV) + self.epsilon)
            experimentPop.Chrom[SD_ind] = ea.boundfix(experimentPop.Encoding, experimentPop.Chrom[SD_ind], experimentPop.Field)
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果