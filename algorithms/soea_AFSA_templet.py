import geatpy as ea  # 导入geatpy库
import numpy as np

class soea_AFSA_templet(ea.SoeaAlgorithm):
    """
soea_AFSA_templet : class - AFSA Algorithm(人工鱼群算法类)

注意：仅支持实整数编码

算法描述:
    本算法类实现的是单目标AFSA算法。算法流程如下:
    1) 初始化N条人工鱼。
    2) 若满足停止条件则停止，否则继续执行。
    3) 所有人工鱼进行聚群行为和追尾行为。
    4) 如果聚群行为和追尾行为都没有找到更优解则进行觅食行为。
    5) 更新最优解。
    6) 回到第2步。
    
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
        self.name = 'AFSA'
        self.visuanl = 50 # 视野
        self.step = 1 # 步长
        self.nmax = 500 # 期望在邻域内聚集的最大人工鱼数目
        # self.alpha = 0.01 # 拥挤度因子中的参数 属于(0,1]
        self.try_number = 3 # 觅食行为的尝试次数
        # self.delta = 1 / (self.alpha * self.nmax) if self.problem.maxormins == [-1] else self.alpha * self.nmax # 拥挤度因子
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')
    
    def swarm_and_follow(self, population): # 聚群和追尾行为, 更优解均已经计算过目标函数
        offspring = population.copy()
        Xc = offspring[0] # 存储邻域内中心个体的容器, 也会被临时用作存储聚群行为产生的新个体的容器
        better_i = []
        have_objv_i = [] # 计算了目标函数值的新个体索引, 需要选择时出现此情况, 是better_i的子集, 其余better_i未计算目标函数
        swarm_nums, follow_nums = 0, 0
        for i, Xi in enumerate(population.Chrom):
            d = np.linalg.norm(Xi - population.Chrom, axis=1)
            in_visual_bool = d <= self.visuanl
            in_visual_bool[i] = False # 去除自己(当前遍历的人工鱼)
            nf = in_visual_bool.sum()# 邻域内的伙伴数目
            if nf:
                Xj = None
                # 聚群行为
                Xc.Chrom = population.Chrom[in_visual_bool].mean(axis=0, keepdims=True)
                self.call_aimFunc(Xc)
                temp_pop = Xc + population[i]
                FitnV = ea.scaling(temp_pop.ObjV, temp_pop.CV, self.problem.maxormins)  # 计算适应度
                # if (FitnV[0] > FitnV[1]) and ((temp_pop[0].ObjV / nf) > (self.delta * temp_pop[1].ObjV)):
                if (FitnV[0] > FitnV[1]) and (nf <= self.nmax):
                    Xic = Xc.Chrom - Xi
                    Xj = Xi + Xic / np.linalg.norm(Xic) * self.step * np.random.rand()
                    Xj = ea.boundfix("RI", Xj, offspring.Field) # 边界修复
                    better_i.append(i)
                # 追尾行为
                Xb = population[in_visual_bool][population[in_visual_bool].FitnV.argmax()] # 邻域内的最优个体, 也会被临时用作存储追尾行为产生的新个体的容器
                # if (Xb.FitnV > population[i].FitnV) and ((Xb.ObjV / nf) > (self.delta * population[i].ObjV)):
                if (Xb.FitnV > population[i].FitnV) and (nf <= self.nmax):
                    Xib = Xb.Chrom - Xi
                    Xj_ = Xi + Xib / np.linalg.norm(Xib) * self.step * np.random.rand()
                    Xj_ = ea.boundfix("RI", Xj_, offspring.Field) # 边界修复
                    if Xj is None:
                        offspring.Chrom[i] = Xj_
                        better_i.append(i)
                        follow_nums += 1
                    else: # 选择聚群行为和追尾行为中更优的个体
                        Xc.Chrom, Xb.Chrom = Xj, Xj_
                        temp_pop = Xc + Xb
                        self.call_aimFunc(temp_pop)
                        FitnV = ea.scaling(temp_pop.ObjV, temp_pop.CV, self.problem.maxormins)  # 计算适应度
                        if FitnV[0] > FitnV[1]:
                            offspring.Chrom[i] = Xj
                            offspring.ObjV[i] = temp_pop[0].ObjV
                            swarm_nums += 1
                        else:
                            offspring.Chrom[i] = Xj_
                            offspring.ObjV[i] = temp_pop[1].ObjV
                            follow_nums += 1
                        if offspring.CV is not None:
                            offspring.CV[i] = temp_pop[0].CV if FitnV[0] > FitnV[1] else temp_pop[1].CV
                        have_objv_i.append(i)
                else:
                    if Xj is not None:
                        offspring.Chrom[i] = Xj
                        swarm_nums += 1
        # 没有计算过目标函数的更优解计算目标函数
        need_call_aimfunc_i = list(set(better_i) - set(have_objv_i))
        if need_call_aimfunc_i:
            nca_offspring = offspring[need_call_aimfunc_i]
            self.call_aimFunc(nca_offspring)
            offspring[need_call_aimfunc_i] = nca_offspring
        return offspring, better_i, swarm_nums, follow_nums
    
    def prey(self, population): # 觅食行为, 均已经计算过目标函数
        offspring = population.copy()
        X = offspring[0]
        for _ in range(self.try_number):
            offspring.Chrom = population.Chrom + self.visuanl * np.random.uniform(-1, 1, population.Chrom.shape)
            offspring.Chrom = ea.boundfix("RI", offspring.Chrom, offspring.Field) # 边界修复
            self.call_aimFunc(offspring)
            population = offspring + population
            FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度
            mv_condition = FitnV[:offspring.sizes] > FitnV[offspring.sizes:]
            mv_ind, dmv_ind = np.where(mv_condition)[0], np.where(~mv_condition)[0] # 可以移动和不可以移动的染色体索引
            population = population[np.arange(offspring.sizes, 2*offspring.sizes)]
            if mv_ind.size:
                Xi, population = population[mv_ind], population[dmv_ind]
                Xj, offspring = offspring[mv_ind], offspring[dmv_ind]
                Xij = Xj.Chrom - Xi.Chrom
                Xi.Chrom = Xi.Chrom + Xij / np.linalg.norm(Xij, axis=1, keepdims=True) * self.step * np.random.random(Xi.Chrom.shape)
                Xi.Chrom = ea.boundfix("RI", Xi.Chrom, Xi.Field) # 边界修复
                self.call_aimFunc(Xi)
                X += Xi
            if offspring.sizes == 0:
                return X[slice(1, X.sizes)]
        self.call_aimFunc(offspring)
        X += offspring
        return X[slice(1, X.sizes)]
    
    def udf_log(self, swarm_nums, follow_nums, prey_nums): # 记录每代的各类行为数目
        if self.logTras:
            for log_name, nums in zip(["swarm", "follow", "prey"], 
                                    [swarm_nums, follow_nums, prey_nums]):
                log_value = self.log.get(log_name, [])
                log_value.append(nums)
                self.log[log_name] = log_value
        
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
        # AFSA相关参数的计算 
        # self.delta = 1 / (self.alpha * self.nmax) if self.problem.maxormins == [-1] else self.alpha * self.nmax # 拥挤度因子
        self.ind_set = {i for i in range(NIND)}
        self.udf_log(0, 0, 0)
        # ===========================开始进化============================
        while not self.terminated(population):
            # 聚群和追尾行为
            population, better_i, swarm_nums, follow_nums = self.swarm_and_follow(population)
            # 觅食行为
            # 需要进行觅食行为的人工鱼索引
            need_prey_i = list(self.ind_set - set(better_i))
            prey_nums = len(need_prey_i)
            if prey_nums:
                offspring = self.prey(population[need_prey_i])
                population[need_prey_i] = offspring
            population.FitnV = ea.scaling(population.ObjV, population.CV, self.problem.maxormins)  # 计算适应度, 均以计算过目标函数值
            self.udf_log(swarm_nums, follow_nums, prey_nums)
        return self.finishing(population)  # 调用finishing完成后续工作并返回结果