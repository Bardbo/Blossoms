# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库


class soea_SCO_templet(ea.SoeaAlgorithm):
    """
soea_SCO_templet : class - 筋斗云优化算法类(SomersaultCloud)

算法描述:
    本算法类实现的是SCO单目标筋斗云优化算法。算法流程如下:
    1) 均匀初始化候选解种群, 在初始化种群的每个个体附近生成一个搭档个体, 初始个体和其对应的搭档个体成为一组。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 同组内的两个个体比较适应度, 适应度低的朝适应度高的个体移动。
    5) 退出机制:设置固定轮次后淘汰最优解排在末位的组合,并通过筋斗云机制和随机化方法补充新的组合。最差的组合使用筋斗云机制。
    筋斗云机制：通过比较这组解的最终位置和初始位置生成一个反向距离的新解并在其附近生成搭档个体, 加入迭代。
    6) 回到第2步。

由于组内个体迭代更新过程像是两个个体形成的线段在决策空间内翻筋斗,同时存在种群大小个翻筋斗的个体组,因此算法命名为筋斗云。
其中步骤5)中新个体的加入也像一个筋斗云可以十万八千里。

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
        self.name = 'SCO'
        self.r = 5 # 邻域距离, 用于生成搭档个体
        self.lmax_rate = 0.2 # 最大翻筋斗长度为决策变量范围百分比
        self.out_epoch = 10 # 解多少轮退出一次
        self.sc_out_rate = 0.4 # 筋斗云机制淘汰的比例
        self.random_out_rate = 0.4 # 随机生成机制淘汰的比例
        self.group_best_weight = 0.5
        self.global_best_weight = 0.5
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')
    
    def get_partner(self, population):
        offspring = population.copy()
        offspring.Chrom = offspring.Chrom + np.random.random(offspring.Chrom.shape) * self.r
        offspring.Chrom = ea.boundfix(offspring.Encoding, offspring.Chrom, offspring.Field)
        return offspring

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
        self.call_aimFunc(population)
        offspring = self.get_partner(population)
        self.call_aimFunc(offspring)
        # 初始化相关参数
        self.lmax = (self.lmax_rate * (self.problem.ranges[1] - self.problem.ranges[0])).reshape(1, -1) # 最大翻筋斗长度
        self.init_Chrom = population.Chrom # 初始种群保留, 用于退出机制生成新的个体
        self.sc_out_nums = int(NIND * self.sc_out_rate)
        self.random_out_nums = int(NIND * self.random_out_rate)
        # ===========================开始进化============================
        pop_group = population + offspring
        pop_group.FitnV = ea.scaling(pop_group.ObjV, pop_group.CV, self.problem.maxormins)  # 计算适应度
        while not self.terminated(pop_group):
            # population 总是存储对应位置组合中更优的个体, offspring 则存储另一个个体
            worse_i = np.where(pop_group.FitnV[:NIND] < pop_group.FitnV[NIND:])[0]
            population, offspring = pop_group[slice(0, NIND)], pop_group[slice(NIND, 2*NIND)]
            population[worse_i] = offspring[worse_i]
            offspring[worse_i] = pop_group[worse_i]
            if self.currentGen % self.out_epoch == 0:
                # 组合的退出与加入机制: 其中省去了offspring的目标函数值计算和population、offspring的适应度计算
                argsort_i = population.FitnV.argsort(axis=0).flatten()
                sc_out_group_i = argsort_i[:self.sc_out_nums]
                random_out_group_i = argsort_i[self.sc_out_nums:self.sc_out_nums+self.random_out_nums]
                # 筋斗云机制生成新组合, 分别放入population, offspring中对应位置(此处不管组合中哪个更优)
                sc_pop = population[sc_out_group_i].copy()
                step = np.random.random(sc_pop.Chrom.shape) + 1
                l = (self.init_Chrom[sc_out_group_i]- sc_pop.Chrom) * step
                sc_pop.Chrom = sc_pop.Chrom + np.clip(l, -self.lmax, self.lmax)
                sc_pop.Chrom = ea.boundfix(sc_pop.Encoding, sc_pop.Chrom, sc_pop.Field)   
                self.call_aimFunc(sc_pop)
                population[sc_out_group_i] = sc_pop
                offspring[sc_out_group_i].Chrom = self.get_partner(sc_pop).Chrom
                self.init_Chrom[sc_out_group_i] = sc_pop.Chrom
                # 随机机制生成新组合
                random_pop = population[random_out_group_i].copy()
                sc_pop.initChrom(self.random_out_nums)
                random_pop.Chrom = sc_pop.Chrom
                self.call_aimFunc(random_pop)
                population[random_out_group_i] = random_pop
                offspring[random_out_group_i].Chrom = self.get_partner(random_pop).Chrom
                self.init_Chrom[random_out_group_i] = random_pop.Chrom
            # 翻筋斗
            l1 = (population.Chrom - offspring.Chrom) * (np.random.random(population.Chrom.shape) + 1) # 朝组合更优翻筋斗
            l2 = (self.BestIndi.Chrom - offspring.Chrom) * (np.random.random(population.Chrom.shape) + 1) # 朝全局最优翻筋斗
            # 差的解移动
            offspring.Chrom = offspring.Chrom + np.clip(self.group_best_weight * l1 + self.global_best_weight * l2, -self.lmax, self.lmax)
            offspring.Chrom = ea.boundfix(offspring.Encoding, offspring.Chrom, offspring.Field)
            self.call_aimFunc(offspring)
            pop_group = population + offspring
            pop_group.FitnV = ea.scaling(pop_group.ObjV, pop_group.CV, self.problem.maxormins)  # 计算适应度
        return self.finishing(pop_group)  # 调用finishing完成后续工作并返回结果
