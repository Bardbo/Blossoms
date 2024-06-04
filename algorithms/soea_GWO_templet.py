# -*- coding: utf-8 -*-
import geatpy as ea  # 导入geatpy库
import numpy as np

class soea_GWO_templet(ea.SoeaAlgorithm):
    """
soea_GWO_templet : class - GWO Algorithm(灰狼优化算法类)

注意：仅支持实整数编码

算法描述:
    本算法类实现的是单目标GWO算法。算法流程如下:
    1) 初始化N头狼的位置。
    2) 若满足停止条件则停止，否则继续执行。
    3) 根据适应度值排序得到头狼，即alpha, beta, delta。
    4) 计算参数a,A,C,结合头狼位置更新其余普通狼位置。
    5) 回到第2步。

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
            self.name = 'GWO'
            self.head_wolves_nums = 3 # 头狼个数
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
        other_wolves_nums = NIND - self.head_wolves_nums # 普通狼的数量
        if self.MAXGEN is not None:
            MAXGEN = self.MAXGEN
        else:
            MAXGEN = 100 if self.MAXEVALS is None else np.ceil(self.MAXEVALS / NIND)
        # ===========================开始进化============================
        while not self.terminated(population):
            # 计算参数
            a = 2 - 2 * self.currentGen / MAXGEN # 收敛因子
            r1 = np.random.rand(other_wolves_nums, self.head_wolves_nums)
            A = 2 * a * r1 - a
            C = 2 * np.random.rand(other_wolves_nums, self.head_wolves_nums)
            # 取出头狼和普通狼的染色体矩阵
            population = population[population.FitnV.argsort(axis=0)[::-1]] # 将种群按照适应度从大到小的顺序排列
            head_wolves_Chrom = population.Chrom[:self.head_wolves_nums] # 头狼的Chrom矩阵,对应灰狼算法中的alpha, beta, delta
            other_wolves_Chrom = population.Chrom[self.head_wolves_nums:] # 普通狼的Chrom矩阵
            # 更新普通狼的位置
            for owi in range(other_wolves_nums):
                X = []
                for hwi in range(self.head_wolves_nums):
                    Di = np.abs(C[owi, hwi] * head_wolves_Chrom[hwi] - other_wolves_Chrom[owi])
                    X.append(head_wolves_Chrom[hwi] - A[owi, hwi] * Di)
                other_wolves_Chrom[owi] = np.array(X).mean(axis=0)
            population.Chrom = np.vstack((head_wolves_Chrom, other_wolves_Chrom))
            population.Chrom = ea.boundfix("RI", population.Chrom, population.Field)
            self.call_aimFunc(population)  # 计算目标函数值
            population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
