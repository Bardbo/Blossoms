import time
import geatpy as ea  # 导入geatpy库
import numpy as np

class soea_SA_templet(ea.SoeaAlgorithm):
    """
soea_SA_templet : class - SA Algorithm(模拟退火算法类)

注意：仅支持实整数编码;

算法描述:
    本算法类实现的是单目标SA算法。与原始算法不同,该可以设置种群大小,种群大小为1时即是原始算法情况。
    算法流程如下:
    1) 初始化N个个体。
    2) 若满足停止条件则停止，否则继续执行。
    3) 生成邻域个体,按照Metropolis准则接受新解。
    4) 回到第2步。
    
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
        self.name = 'SA'
        self.t_now = 3000 # 初始温度
        self.t_final = 10 # 终止温度
        self.alpha = 0.98 # 冷却系数
        self.inner_iter = 200 # 内层迭代次数
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')
        self.NIND = population.sizes
    
    def generate_neighbors(self, pop):
        neighbors = pop.copy()
        lb_range = neighbors.Chrom - self.problem.ranges[0]
        ub_range = self.problem.ranges[1] - neighbors.Chrom
        # neighbors.Chrom += np.random.normal(size=neighbors.Chrom.shape) * np.maximum(lb_range, ub_range) / 3
        neighbors.Chrom += np.random.uniform(-1, 1, size=neighbors.Chrom.shape) * np.maximum(lb_range, ub_range) / 3
        neighbors.Chrom = ea.boundfix(neighbors.Encoding, neighbors.Chrom, neighbors.Field)
        return neighbors
    
    def metropolis(self, population):
        feasible = np.where(np.all(population.CV <= 0, 1))[0] if population.CV is not None else np.arange(population.sizes)  # 找到满足约束条件的个体的下标
        if len(feasible) > 0:
            feasiblePop = population[feasible]
            bestIndi = feasiblePop[np.argmax(feasiblePop.FitnV)]  # 获取最优个体
            if self.BestIndi.sizes == 0:
                self.BestIndi = bestIndi  # 初始化global best individual
            else:
                delta = (self.BestIndi.ObjV - bestIndi.ObjV) * self.problem.maxormins if \
                    self.problem.maxormins is not None else self.BestIndi.ObjV - bestIndi.ObjV
                # 更新global best individual
                if delta > 0:
                    self.BestIndi = bestIndi
        pop_FitnV, off_FitnV = population.FitnV[:self.NIND, 0], population.FitnV[self.NIND:, 0]
        neighbors = population[slice(self.NIND, self.NIND*2)]
        ind = np.where(off_FitnV < pop_FitnV)[0] # 不能直接成为子代的新解索引, 有概率接受差解
        if ind.size:
            p = np.exp((off_FitnV[ind] - pop_FitnV[ind]) / self.t_now) # 时间越长温度越低, p越小
            ind = ind[np.where(np.random.random(size=p.shape) > p)[0]] # 如果概率小于等于p则接受差解, 大于则不接受
            neighbors[ind] = population[ind]
        return neighbors
        
    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群
        # ==========================初始化配置===========================
        population = self.population
        NIND = self.NIND
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        population.initChrom(NIND)  # 初始化种群染色体矩阵
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        self.call_aimFunc(population)  # 计算种群的目标函数值
        population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
        # ===========================开始进化============================
        while not self.terminated(population):
            for _ in range(self.inner_iter):
                neighbors = self.generate_neighbors(population) # 生成邻域新解
                self.call_aimFunc(neighbors)
                population = population + neighbors
                population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
                population = self.metropolis(population)
            self.t_now *= self.alpha
            # 温度停止条件
            if self.t_now <= self.t_final:
                self.check(population)  # 检查种群对象的关键属性是否有误
                self.stat(population)  # 分析记录当代种群的数据
                self.passTime += time.time() - self.timeSlot  # 更新耗时
                # 调用outFunc()
                if self.outFunc is not None:
                    if type(self.outFunc) != 'function':
                        raise RuntimeError('outFunc must be a function. (如果定义了outFunc，那么它必须是一个函数。)')
                    self.outFunc(self, population)
                self.timeSlot = time.time()  # 更新时间戳
                self.stopMsg = 'The algotirhm stepped because it exceeded the temperature limit.'
                break
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果