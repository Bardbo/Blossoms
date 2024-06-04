# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库


class soea_GSK_templet(ea.SoeaAlgorithm):
    """
soea_GSK_templet : class - Gaining Sharing Knowledge based Algorithm算法类

算法描述:
    本算法类实现的是Gaining Sharing Knowledge based Algorithm。算法流程如下:
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 通过初级获取和共享知识阶段和高级获取和共享知识阶段得到试验种群。
    5) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    6) 回到第2步。

参考文献:
    [1] Mohamed A W , Hadi A A , Mohamed A K .Gaining-sharing knowledge based 
    algorithm for solving optimization problems: a novel nature-inspired 
    algorithm[J].International Journal of Machine Learning and Cybernetics, 
    2020, 11(7):1501-1529.DOI:10.1007/s13042-019-01053-x.

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
        self.name = 'GSK'
        self.selFunc = 'ecs'  # 基向量的选择方式，采用精英复制选择
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')
        self.k = 1 # 知识率 > 0
        self.kf = 0.5 # 知识因素
        self.kr = 0.9 # 知识比率 [0, 1]
        self.p = 0.1 # 最优和最差个体比例 [0, 1]
    
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
        # GSK参数
        self.problemsize = self.problem.Dim
        self.best_nind = int(NIND * self.p)
        self.worst_nind = self.best_nind
        self.middle_nind = NIND - self.best_nind - self.worst_nind
        if self.MAXGEN is not None:
            MAXGEN = self.MAXGEN
        else:
            MAXGEN = 100 if self.MAXEVALS is None else np.ceil(self.MAXEVALS / NIND)
        # ===========================开始进化============================
        while not self.terminated(population):
            # 获取初级要素数目和高级要素数目
            Djunior = int(self.problemsize * (1 - self.currentGen / MAXGEN) ** self.k)
            Dsenior = self.problemsize - Djunior
            # 个体排序 最优到最差
            population = population[population.FitnV.flatten().argsort()[::-1]]
            experimentPop = population.copy()  # 存储试验个体
            if Djunior > 0:
                # 初级获取和共享知识阶段
                x_i0 = population[[1] + list(range(NIND-2)) + [NIND-3]] # 最近的更优个体
                x_i1 = population[[2] + list(range(2, NIND)) + [NIND-2]] # 最近的更差个体
                x_r = population[np.random.choice(np.arange(NIND), NIND)] # 随机个体
                rand_mat = np.random.rand(NIND, Djunior) <= self.kr
                better_ind = population.FitnV.flatten() > x_r.FitnV.flatten()
                better_rand_ind = rand_mat[better_ind]
                worse_rand_ind = rand_mat[~better_ind]
                xij = population.Chrom[:, :Djunior]
                x_i0, x_i1, x_r = x_i0.Chrom[:, :Djunior], x_i1.Chrom[:, :Djunior], x_r.Chrom[:, :Djunior]
                experimentPop.Chrom[:, :Djunior][rand_mat] = xij[rand_mat] + self.kf * (x_i0[rand_mat] - x_i1[rand_mat])
                experimentPop.Chrom[:, :Djunior][better_ind][better_rand_ind] += (self.kf * (x_r[better_ind][better_rand_ind] - xij[better_ind][better_rand_ind]))
                experimentPop.Chrom[:, :Djunior][~better_ind][worse_rand_ind] += (self.kf * (xij[~better_ind][worse_rand_ind] - x_r[~better_ind][worse_rand_ind]))
            # 高级获取和共享知识阶段
            x_i0 = population[np.random.choice(np.arange(self.best_nind), NIND)] # 更优个体
            x_i1 = population[np.random.choice(np.arange(self.best_nind+self.middle_nind, NIND), NIND)]  # 更差个体
            x_r = population[np.random.choice(np.arange(self.best_nind, self.best_nind+self.middle_nind), NIND)] # 中等个体
            rand_mat = np.random.rand(NIND, Dsenior) <= self.kr
            better_ind = population.FitnV.flatten() > x_r.FitnV.flatten()
            better_rand_ind = rand_mat[better_ind]
            worse_rand_ind = rand_mat[~better_ind]
            xij = population.Chrom[:, Djunior:]
            x_i0, x_i1, x_r = x_i0.Chrom[:, Djunior:], x_i1.Chrom[:, Djunior:], x_r.Chrom[:, Djunior:]
            experimentPop.Chrom[:, Djunior:][rand_mat] = xij[rand_mat] + self.kf * (x_i0[rand_mat] - x_i1[rand_mat])
            experimentPop.Chrom[:, Djunior:][better_ind][better_rand_ind] += (self.kf * (x_r[better_ind][better_rand_ind] - xij[better_ind][better_rand_ind]))
            experimentPop.Chrom[:, Djunior:][~better_ind][worse_rand_ind] += (self.kf * (xij[~better_ind][worse_rand_ind] - x_r[~better_ind][worse_rand_ind]))
            experimentPop.Chrom = ea.boundfix(experimentPop.Encoding, experimentPop.Chrom, experimentPop.Field)
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
