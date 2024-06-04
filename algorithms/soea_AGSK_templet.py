# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea  # 导入geatpy库


class soea_AGSK_templet(ea.SoeaAlgorithm):
    """
soea_AGSK_templet : class - Adaptive Gaining Sharing Knowledge based Algorithm算法类

算法描述:
    本算法类实现的是Adaptive Gaining Sharing Knowledge based Algorithm。算法流程如下:
    1) 初始化候选解种群。
    2) 若满足停止条件则停止，否则继续执行。
    3) 对当前种群进行统计分析，比如记录其最优个体、平均适应度等等。
    4) 通过初级获取和共享知识阶段和高级获取和共享知识阶段得到试验种群。
    5) 在当前种群和实验种群之间采用一对一生存者选择方法得到新一代种群。
    6) 回到第2步。
    基本与GSK相同
    kf和kr可以池自适应, 会更改参数池对应参数被选择的概率;
    k知识率在种群有(0,1)和[1,20]两种取值, 概率各为0.5;
    种群规模会线性缩减。

参考文献:
    [1] Mohamed A W , Hadi A A , Mohamed A K ,et al.Evaluating the Performance of Adaptive GainingSharing 
    Knowledge Based Algorithm on CEC 2020 Benchmark Problems[C]//2020 IEEE Congress on Evolutionary Computation (CEC)
    .0[2024-03-16].DOI:10.1109/CEC48606.2020.9185901.

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
        self.name = 'AGSK'
        self.selFunc = 'ecs'  # 基向量的选择方式，采用精英复制选择
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')
        self.kf = [0.1, 1.0, 0.5, 1.0] # 知识因素池
        self.kr = [0.2, 0.1, 0.9, 0.9] # 知识比率池 [0, 1]
        self.p = 0.1 # 最优和最差个体比例 [0, 1]
        self.kw_p = [0.85, 0.05, 0.05, 0.05] # 对应kf, kr被选择的概率
        self.Nmin = 12 # 最小种群大小
        self.c = 0.05
        self.pool_ind = np.arange(len(self.kf))

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
        self.kf = np.array(self.kf)
        self.kr = np.array(self.kr)
        self.kw_p = np.array(self.kw_p)
        self.w_ps = np.zeros_like(self.kw_p)
        kind = np.random.rand(NIND)
        self.k = np.zeros_like(kind) # 知识率
        self.k[kind < 0.5] = np.random.rand((kind < 0.5).sum())
        self.k[kind >= 0.5] = np.random.uniform(1, 20, (kind >= 0.5).sum())
        self.problemsize = self.problem.Dim
        self.Ninit = NIND
        self.Nmin = min(self.Nmin, NIND)
        if self.MAXEVALS is None:
            MAXEVALS = NIND * 100 if self.MAXGEN is None else NIND * self.MAXGEN
        else:
            MAXEVALS = self.MAXEVALS
        # ===========================开始进化============================
        while not self.terminated(population):
            if self.evalsNum >= 0.1 * MAXEVALS:
                self.kw_p = (1 - self.c) * self.kw_p + self.c * self.w_ps
            k_rand_ind = np.random.choice(self.pool_ind, NIND, p=self.kw_p / self.kw_p.sum())
            kf = self.kf[k_rand_ind].reshape(-1, 1)
            kf = np.tile(kf, (1, self.problemsize))
            kr = self.kr[k_rand_ind].reshape(-1, 1)
            # 获取初级要素数目和高级要素数目
            Djunior = (self.problemsize * (1 - self.evalsNum / MAXEVALS) ** self.k[:NIND]).astype(int)
            mask_junior = np.zeros_like(population.Chrom, dtype=bool)
            for i,j in enumerate(Djunior):
                mask_junior[i, :j] = True
            # 个体排序 最优到最差
            population = population[population.FitnV.flatten().argsort()[::-1]]
            experimentPop = population.copy()  # 存储试验个体
            # 初级获取和共享知识阶段
            x_i0 = population[[1] + list(range(NIND-2)) + [NIND-3]] # 最近的更优个体
            x_i1 = population[[2] + list(range(2, NIND)) + [NIND-2]] # 最近的更差个体
            x_r = population[np.random.choice(np.arange(NIND), NIND)] # 随机个体
            rand_mat = (np.random.rand(NIND, self.problemsize) <= kr) & mask_junior
            better_ind = population.FitnV.flatten() > x_r.FitnV.flatten()
            better_rand_ind = rand_mat[better_ind]
            worse_rand_ind = rand_mat[~better_ind]
            xij = population.Chrom
            x_i0, x_i1, x_r = x_i0.Chrom, x_i1.Chrom, x_r.Chrom
            experimentPop.Chrom[rand_mat] = xij[rand_mat] + kf[rand_mat] * (x_i0[rand_mat] - x_i1[rand_mat])
            experimentPop.Chrom[better_ind][better_rand_ind] += (kf[better_ind][better_rand_ind] * (x_r[better_ind][better_rand_ind] - xij[better_ind][better_rand_ind]))
            experimentPop.Chrom[~better_ind][worse_rand_ind] += (kf[~better_ind][worse_rand_ind] * (xij[~better_ind][worse_rand_ind] - x_r[~better_ind][worse_rand_ind]))
            # 高级获取和共享知识阶段
            self.best_nind = int(NIND * self.p)
            self.worst_nind = self.best_nind
            self.middle_nind = NIND - self.best_nind - self.worst_nind
            x_i0 = population[np.random.choice(np.arange(self.best_nind), NIND)] # 更优个体
            x_i1 = population[np.random.choice(np.arange(self.best_nind+self.middle_nind, NIND), NIND)]  # 更差个体
            x_r = population[np.random.choice(np.arange(self.best_nind, self.best_nind+self.middle_nind), NIND)] # 中等个体
            rand_mat = ~((np.random.rand(NIND, self.problemsize) <= kr) & mask_junior)
            better_ind = population.FitnV.flatten() > x_r.FitnV.flatten()
            better_rand_ind = rand_mat[better_ind]
            worse_rand_ind = rand_mat[~better_ind]
            xij = population.Chrom
            x_i0, x_i1, x_r = x_i0.Chrom, x_i1.Chrom, x_r.Chrom
            experimentPop.Chrom[rand_mat] = xij[rand_mat] + kf[rand_mat] * (x_i0[rand_mat] - x_i1[rand_mat])
            experimentPop.Chrom[better_ind][better_rand_ind] += (kf[better_ind][better_rand_ind] * (x_r[better_ind][better_rand_ind] - xij[better_ind][better_rand_ind]))
            experimentPop.Chrom[~better_ind][worse_rand_ind] += (kf[~better_ind][worse_rand_ind] * (xij[~better_ind][worse_rand_ind] - x_r[~better_ind][worse_rand_ind]))
            experimentPop.Chrom = ea.boundfix(experimentPop.Encoding, experimentPop.Chrom, experimentPop.Field)
            self.call_aimFunc(experimentPop)  # 计算目标函数值
            tempPop = population + experimentPop  # 临时合并，以调用otos进行一对一生存者选择
            tempPop.FitnV = ea.scaling(tempPop.ObjV, tempPop.CV, self.problem.maxormins)  # 计算适应度
            population = tempPop[ea.selecting('otos', tempPop.FitnV, NIND)]  # 采用One-to-One Survivor选择，产生新一代种群
            FitnV = tempPop.FitnV[:NIND] # old fitness
            self.w_ps = np.zeros_like(self.kw_p)
            for i in self.pool_ind:
                ind = np.where(k_rand_ind == i)[0]
                self.w_ps[i] = (population.FitnV[ind] - FitnV[ind]).sum()
            self.w_ps = self.w_ps / (self.w_ps.sum() + 1e-5)
            self.w_ps = np.clip(self.w_ps, 0.05, None)
            NIND = round((self.Nmin-self.Ninit) * self.evalsNum / MAXEVALS + self.Ninit)
            population = population[population.FitnV.flatten().argpartition(-NIND)[-NIND:]]
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果
