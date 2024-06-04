# -*- coding: utf-8 -*-
import numpy as np
from scipy.stats import levy
import geatpy as ea  # 导入geatpy库


class soea_HO_templet(ea.SoeaAlgorithm):
    """
soea_HO_templet : class - Hippopotamus Optimization Algorithm河马优化算法类

算法描述:
    本算法类实现的是Hippopotamus Optimization Algorithm河马优化算法。算法流程如下:
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 河马在河流的位置更新 探索阶段1。
    5) 河马防御捕食者 探索阶段2。
    6) 河马逃离捕食者 开发阶段。
    7) 回到第2步。

参考文献:
    [1] Amiri, M. H., Mehrabi Hashjin, N., Montazeri, M., Mirjalili, S., & Khodadadi, N. (2024). 
    Hippopotamus optimization algorithm: a novel nature-inspired optimization algorithm. Scientific reports, 
    14(1), 5032. https://doi.org/10.1038/s41598-024-54910-3

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
        self.name = 'HO'
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
        # HO参数
        self.ind = np.arange(NIND)
        halfn = int(NIND / 2)
        self.Phase1_ind = np.arange(halfn) # 探索阶段1
        dim = self.problem.Dim
        self.Phase2_ind = np.arange(halfn, NIND) # 探索阶段2
        phase2_nind = NIND - halfn
        Lb = self.problem.ranges[0]
        Ub = self.problem.ranges[1]
        if self.MAXGEN is not None:
            MAXGEN = self.MAXGEN
        else:
            MAXGEN = 100 if self.MAXEVALS is None else np.ceil(self.MAXEVALS / NIND)
        # ===========================开始进化============================
        while not self.terminated(population):
            Xbest = population[np.argmax(population.FitnV)].Chrom
            # Phase1 河流的位置更新 探索阶段
            Phase1 = population[self.Phase1_ind]
            I1, I2 = np.random.randint(1,3, size=(halfn, 1)), np.random.randint(1,3, size=(halfn, 1))
            Q1, Q2 = np.random.randint(0, 2, size=(halfn, 1)), np.random.randint(0, 2, size=(halfn, 1))
            rand_nums = np.random.randint(1, NIND+1, size=halfn) # 随机河马的数目
            rand_ind = np.random.randint(0, NIND, size=(halfn, NIND))
            mG = np.zeros_like(Phase1.Chrom)
            for i, rgi in enumerate(rand_ind):
                mG[i]= population[rgi[:rand_nums[i]]].Chrom.mean(axis=0)
            # 公式4
            h1 = I2 *  np.random.random(size=(halfn, dim)) + Q1
            h2 = 2 * np.random.random(size=(halfn, dim)) - 1
            h3 = np.random.random(size=(halfn, dim))
            h4 = I1 * np.random.random(size=(halfn, dim)) + Q2
            h5 = np.tile(np.random.random(size=(halfn, 1)), (1, dim))
            h = np.array([h1, h2, h3, h4, h5])
            A = np.random.randint(0, 5, size=(halfn,))
            B = np.random.randint(0, 5, size=(halfn,))
            A = h[A, self.Phase1_ind]
            B = h[B, self.Phase1_ind]
            y1 = np.random.random(size=(halfn, 1))
            Xm = ea.Population(population.Encoding, population.Field, halfn)  # 存储雄性个体
            Xm.Chrom = Phase1.Chrom + y1 * (Xbest - I1 * Phase1.Chrom) # 公式3
            Xm.Chrom = ea.boundfix(Xm.Encoding, Xm.Chrom, Xm.Field)
            self.call_aimFunc(Xm)
            T = np.exp(-self.currentGen / MAXGEN) # 公式5
            Xf = Phase1.copy()  # 存储雌性或未成熟个体
            if T > 0.6:
                Xf.Chrom = Phase1.Chrom + A * (Xbest - I2 * mG) # 公式6
            else: # 公式7
                r6 = np.random.random(size=(halfn,))
                mask = r6 > 0.5
                Xf.Chrom[mask] = Phase1.Chrom[mask] + B[mask] * (mG[mask] - Xbest)
                Xf.Chrom[~mask] = (Ub - Lb) * np.random.random(size=((~mask).sum(), 1)) + Lb
            Xf.Chrom = ea.boundfix(Xf.Encoding, Xf.Chrom, Xf.Field)
            self.call_aimFunc(Xf)
            # 公式8
            tempPop = Phase1 + Xm
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins) 
            Phase1 = tempPop[ea.selecting('otos', tempPop.FitnV, halfn)]
            # 公式9
            tempPop = Phase1 + Xf
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins) 
            Phase1 = tempPop[ea.selecting('otos', tempPop.FitnV, halfn)]
            
            # Phase2 防御捕食者 探索阶段
            Phase2 = population[self.Phase2_ind]
            predator = ea.Population(population.Encoding, population.Field, phase2_nind)
            predator.Chrom = Lb + np.random.random(size=(phase2_nind, dim)) * (Ub - Lb) # 公式10
            self.call_aimFunc(predator)
            tempPop = Phase2 + predator
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)
            mask = tempPop.FitnV[:phase2_nind].flatten() > tempPop.FitnV[phase2_nind:].flatten()
            Distance = np.abs(predator.Chrom - Phase2.Chrom) # 公式11
            b = np.random.uniform(2, 4, size=(phase2_nind, 1))
            c = np.random.uniform(1, 1.5, size=(phase2_nind, 1))
            d = np.random.uniform(2, 3, size=(phase2_nind, 1))
            l = np.random.uniform(-2*np.pi, 2*np.pi, size=(phase2_nind, 1))
            RL = 0.05 * levy.rvs(1.5, size=(phase2_nind, dim))
            # 公式12
            predator.Chrom[mask] = RL[mask] * predator.Chrom[mask] + \
                (b[mask] / (c[mask] - d[mask] * np.cos(l[mask]))) * (1 / Distance[mask])
            predator.Chrom[~mask] = RL[~mask] * predator.Chrom[~mask] + \
                (b[~mask] / (c[~mask] - d[~mask] * np.cos(l[~mask]))) * (1 / (2 * Distance[~mask] + np.random.rand((~mask).sum(), dim)))
            predator.Chrom = ea.boundfix(predator.Encoding, predator.Chrom, predator.Field)
            self.call_aimFunc(predator)
            tempPop = Phase2 + predator
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins) 
            Phase2 = tempPop[ea.selecting('otos', tempPop.FitnV, phase2_nind)]
            population = Phase1 + Phase2

            # Phase3 逃离捕食者 开发阶段
            Lb_local = Lb / self.currentGen
            Ub_local = Ub / self.currentGen
            # 公式18
            h1 = 2 *  np.random.random(size=(NIND, dim)) - 1
            h2 = np.tile(np.random.normal(size=(NIND, 1)), (1, dim))
            h3 = np.tile(np.random.random(size=(NIND, 1)), (1, dim))
            h = np.array([h1, h2, h3])
            A = np.random.randint(0, 3, size=(NIND,))
            A = h[A, self.ind]
            X = population.copy()
            X.Chrom = population.Chrom + np.random.rand(NIND, 1) * (Lb_local + A * (Ub_local - Lb_local))
            X.Chrom = ea.boundfix(X.Encoding, X.Chrom, X.Field)
            tempPop = population + X
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins) 
            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果