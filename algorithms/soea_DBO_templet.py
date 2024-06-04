# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库


class soea_DBO_templet(ea.SoeaAlgorithm):
    """
soea_DBO_templet : class - Dung Beetle Optimizer蜣螂优化算法类

算法描述:
    本算法类实现的是Dung Beetle Optimizer蜣螂优化算法。算法流程如下:
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 滚球行为。
    5) 繁殖行为。
    6) 觅食行为。
    7) 偷窃行为。
    8) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    9) 回到第2步。

参考文献:
    [1] Xue J , Shen B .Dung beetle optimizer: a new meta-heuristic algorithm for global optimization[J].
    The Journal of Supercomputing, 2022:1-32.DOI:10.1007/s11227-022-04959-6.

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
        self.name = 'DBO'
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')
        self.dbr = [6, 6, 7, 11] # 滚球蜣螂 繁殖蜣螂 小蜣螂 偷窃蜣螂的数目占比
        self.k = 0.1 # 偏转系数 (0, 0.2]
        self.b = 0.3 # (0, 1)
        self.S = 1
        self.lambda_ = 0.5 # 控制alpha为-1或1的概率参数
        self.obstacle = 0.9 # 无障碍物的概率

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
        # DBO参数
        self.dbr = np.array(self.dbr)
        self.dbr = (self.dbr / self.dbr.sum() * NIND).astype(int)
        self.dbr[-1] = NIND - self.dbr[:3].sum()
        brdb_nums = self.dbr[0]
        sdb_nums = self.dbr[2]
        self.dbr = self.dbr.cumsum()
        brdb_i = np.arange(brdb_nums) # 滚球蜣螂
        bb_i = np.arange(brdb_nums, self.dbr[1]) # 繁殖蜣螂
        sdb_i = np.arange(self.dbr[1], self.dbr[2]) # 小蜣螂
        thief_i = np.arange(self.dbr[2], self.dbr[3]) # 偷窃蜣螂
        Lb = self.problem.ranges[0]
        Ub = self.problem.ranges[1]
        if self.MAXGEN is not None:
            MAXGEN = self.MAXGEN
        else:
            MAXGEN = 100 if self.MAXEVALS is None else np.ceil(self.MAXEVALS / NIND)
        # ===========================开始进化============================
        while not self.terminated(population):
            Xworst = population[np.argmin(population.FitnV)].Chrom
            Xbest = population[np.argmax(population.FitnV)].Chrom
            brdb, bb_t, sdb_t, thief_t = population[brdb_i], population[bb_i], population[sdb_i], population[thief_i]
            brdb_t = brdb.copy()
            # 滚球行为
            alpha = np.random.rand(brdb_nums, 1)
            mask = alpha > self.lambda_
            alpha[mask] = 1
            alpha[~mask] = -1
            mask = np.random.rand(brdb_nums) < self.obstacle
            # 无障碍物
            brdb_t.Chrom[mask] = brdb_t.Chrom[mask] + alpha[mask] * self.k * brdb[mask].Chrom + \
                self.b * np.abs(brdb_t.Chrom[mask] - Xworst)
            # 有障碍物 跳舞
            if (~mask).sum():
                theta = np.random.randint(0, 180, size=((~mask).sum(), 1))
                theta[theta == 90] = 0
                brdb_t.Chrom[~mask] = brdb_t.Chrom[~mask] + \
                    np.tan(np.deg2rad(theta)) * np.abs(brdb_t.Chrom[~mask] - brdb.Chrom[~mask])
            # 繁殖行为
            R = 1 - self.currentGen / MAXGEN
            lb, ub = Xbest * (1 - R), Xbest * (1 + R)
            ea.Problem
            lb = np.maximum(lb, Lb)
            ub = np.minimum(ub, Ub)
            b1 = np.random.random(size=bb_t.Chrom.shape)
            b2 = np.random.random(size=bb_t.Chrom.shape)
            bb_t.Chrom = Xbest + b1 * (bb_t.Chrom - lb) + b2 * (bb_t.Chrom - ub)
            # 觅食行为
            Xgbest = self.BestIndi.Chrom if self.BestIndi.sizes != 0 else Xbest
            lb, ub = Xgbest * (1 - R), Xgbest * (1 + R)
            lb = np.maximum(lb, Lb)
            ub = np.minimum(ub, Ub)
            C1 = np.random.normal(size=(sdb_nums, 1))
            C2 = np.random.random(size=sdb_t.Chrom.shape)
            sdb_t.Chrom = sdb_t.Chrom + C1 * (sdb_t.Chrom - lb) + C2 * (sdb_t.Chrom - ub)
            # 偷窃行为
            g = np.random.normal(size=thief_t.Chrom.shape)
            thief_t.Chrom = Xgbest + self.S * g * (np.abs(thief_t.Chrom - Xbest) + \
                np.abs(thief_t.Chrom - Xgbest))
            experimentPop = brdb_t + bb_t + sdb_t + thief_t
            experimentPop.Chrom = ea.boundfix(experimentPop.Encoding, experimentPop.Chrom, experimentPop.Field)
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果