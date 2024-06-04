# -*- coding: utf-8 -*-
import geatpy as ea  # 导入geatpy库
import numpy as np

class soea_psy_GWO_templet(ea.SoeaAlgorithm):
    """
soea_psy_GWO_templet : class - Polysomy GWO Algorithm(多染色体灰狼优化算法类)

算法类说明:
    该算法类是算法类soea_GWO_templet的多染色体版本,
    因此里面的种群对象为支持混合编码的多染色体种群类PsyPopulation类的对象。(注意：不支持二进制\格雷编码)
    
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
        if population.ChromNum == 1:
            raise RuntimeError('传入的种群对象必须是多染色体的种群类型。')
        self.name = 'psy-GWO'
        self.head_wolves_nums = 3 # 头狼个数
        # 排序编码的交叉变异算子
        self.recOper = ea.Xovpmx(XOVR=0.7)  # 生成部分匹配交叉算子对象
        self.mutOper = ea.Mutinv(Pm=0.5)  # 生成逆转变异算子对象
        for i in range(population.ChromNum):
            if population.Encodings[i] not in ["P", "RI"]:
                raise RuntimeError('编码方式必须为''RI''或''P''.')

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
        # ===========================开始进化============================
        other_wolves_nums = NIND - self.head_wolves_nums # 普通狼的数量
        while not self.terminated(population):
            a = 2 - 2 * self.currentGen / self.MAXGEN # 收敛因子
            population = population[population.FitnV.argsort(axis=0)[::-1]] # 将种群按照适应度从大到小的顺序排列
            # 进行进化操作,分别对各种编码的染色体进行重组和变异
            for i in range(population.ChromNum):
                if population.Encodings[i] == "P": # 排序编码仍然按照排序编码去进化
                    population.Chroms[i] = self.recOper.do(population.Chroms[i])  # 重组
                    population.Chroms[i] = self.mutOper.do(population.Encodings[i], population.Chroms[i],
                                                           population.Fields[i])  # 变异
                else: # 实整数编码使用灰狼算法
                    # 计算参数
                    r1 = np.random.rand(other_wolves_nums, self.head_wolves_nums)
                    A = 2 * a * r1 - a
                    C = 2 * np.random.rand(other_wolves_nums, self.head_wolves_nums)
                    # 取出头狼和普通狼的染色体矩阵
                    head_wolves_Chrom = population.Chroms[i][:self.head_wolves_nums] # 头狼的Chrom矩阵,对应灰狼算法中的alpha, beta, delta
                    other_wolves_Chrom = population.Chroms[i][self.head_wolves_nums:] # 普通狼的Chrom矩阵
                    # 更新普通狼的位置
                    for owi in range(other_wolves_nums):
                        X = []
                        for hwi in range(self.head_wolves_nums):
                            Di = np.abs(C[owi, hwi] * head_wolves_Chrom[hwi] - other_wolves_Chrom[owi])
                            X.append(head_wolves_Chrom[hwi] - A[owi, hwi] * Di)
                        other_wolves_Chrom[owi] = np.array(X).mean(axis=0)
                    population.Chroms[i] = np.vstack((head_wolves_Chrom, other_wolves_Chrom))
                    population.Chroms[i] = ea.boundfix(population.Encodings[i], population.Chroms[i], population.Fields[i])
            self.call_aimFunc(population)  # 计算目标函数值
            population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
