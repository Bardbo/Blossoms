import geatpy as ea  # 导入geatpy库
import numpy as np

class soea_PSO_templet(ea.SoeaAlgorithm):
    """
soea_PSO_templet : class - PSO Algorithm(粒子群优化算法类)

注意：仅支持实整数编码;

算法描述:
    本算法类实现的是单目标PSO算法。算法流程如下:
    1) 初始化N个粒子,包括位置和速度。
    2) 若满足停止条件则停止，否则继续执行。
    3) 更新所有粒子的速度，基于速度更新所有粒子的位置。
    4) 更新历代全局最优粒子、每个粒子的历代最优。
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
        self.name = 'PSO'
        self.w_start = 0.9 # 初始惯性权重
        self.w_end = 0.4 # 最终惯性权重
        self.c1 = 1 # 个体学习因子
        self.c2 = 1 # 群体学习因子
        self.vmax_rate = 0.15
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群
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
        # PSO相关参数的计算 
        self.vmax = (self.vmax_rate * (self.problem.ranges[1] - self.problem.ranges[0])).reshape(1, -1) # 最大速度阈值
        pop_shape = population.Chrom.shape
        v = np.random.uniform(-1, 1, size=pop_shape) * self.vmax # 速度初始化
        if self.w_end > self.w_start:
            raise RuntimeError("w_end应该小于等于w_start")
        if self.MAXGEN is not None:
            MAXGEN = self.MAXGEN
        else:
            MAXGEN = 100 if self.MAXEVALS is None else np.ceil(self.MAXEVALS / NIND)
        self.w_step = (self.w_start - self.w_end) / (MAXGEN - 1) # 惯性因子线性递减
        pBest = population.copy() # 当前各粒子历代最优
        # ===========================开始进化============================
        self.w_t = self.w_start
        while not self.terminated(population):
            gBest = self.BestIndi # 历代全局最优
            offspring = population.copy() # 实际上粒子群没有子代的概念，这里是为了比较得出各粒子移动前后的历代最优
            # 更新速度
            v = self.w_t * v + \
                self.c1 * np.random.random(pop_shape) * (pBest.Chrom - offspring.Chrom) + \
                self.c2 * np.random.random(pop_shape) * (gBest.Chrom - offspring.Chrom)
            v = np.clip(v, -self.vmax, self.vmax)
            self.w_t -= self.w_step
            # 更新位置
            offspring.Chrom += v
            offspring.Chrom = ea.boundfix("RI", offspring.Chrom, offspring.Field)
            self.call_aimFunc(offspring)  # 计算目标函数值
            # 更新pBest 
            population = offspring + pBest
            population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
            pBesti = np.where(population.FitnV[:NIND] >= population.FitnV[-NIND:])[0]
            if pBesti.size:
                pBest[pBesti] = population[pBesti]
            population = population[np.arange(NIND)]
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果