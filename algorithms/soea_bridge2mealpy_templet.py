# -*- coding: utf-8 -*-
import numpy as np # version 1.24.4
import geatpy as ea  # version 2.7.0
import mealpy # version 3.0.1
from mealpy import Termination
import types, time

# 修复mealpy库整数边界bug
class FixedIntegerVar(mealpy.IntegerVar):
    def correct(self, x):
        x = np.clip(x, self.lb, self.ub)
        x = self.round(x)
        return np.array(x, dtype=int)

# 修复mealpy库时间终止条件的bug, 将该方法替代原方法(实例化后打补丁)
def check_termination(self, mode="start", termination=None, epoch=None):
    if mode == "start":
        self.termination = termination
        if termination is not None:
            if isinstance(termination, Termination):
                self.termination = termination
            elif type(termination) == dict:
                self.termination = Termination(log_to=self.problem.log_to, log_file=self.problem.log_file, **termination)
            else:
                raise ValueError("Termination needs to be a dict or an instance of Termination class.")
            self.nfe_counter = 0
            self.termination.set_start_values(0, self.nfe_counter, time.perf_counter(), 0)
    else:
        finished = False
        if self.termination is not None:
            es = self.history.get_global_repeated_times(self.termination.epsilon)
            finished = self.termination.should_terminate(epoch, self.nfe_counter, time.perf_counter()-self.termination.start_time, es) # 此处mealpy库忘记减去开始时间
            if finished:
                self.logger.warning(self.termination.message)
        return finished

class soea_bridge2mealpy_templet(ea.SoeaAlgorithm):
    """
soea_bridge2mealpy_templet : class - 桥接到mealpy库中单目标算法类

注意: 
    1.不支持决策变量上下界不能取到的情况
    2.不支持约束矩阵CV的写法,请自行使用罚函数的方法处理约束条件
    3.不支持多目标帕累托的方法, 仅支持加权的方法, 因此建议自行在evalVars函数中返回加权后的目标值
    4.evalVars函数请加上single装饰器,即只返回一个ObjV值,因为mealpy是分个体计算的,同时 假的geatpy进化 也需要调用
    5.evalsNum属性是没有真实统计的
    由于未深入了解mealpy,上述事项可能理解有误
"""

    def __init__(self,
                 problem,
                 population,
                 optimizer_name=None,
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
        if population.Encoding not in ["P", "RI", "BG"]:
            raise RuntimeError('编码方式必须为''BG''、''RI''或''P''.')
        optimizer_name = optimizer_name or kwargs.get("optimizer_name", None)
        if optimizer_name is None:
            raise TypeError('optimizer_name 不能为None')
        else:
            self.name = optimizer_name
        # 特定算法的其余参数也可在此设置
        self.model:mealpy.Optimizer = mealpy.get_optimizer_by_name(self.name)(pop_size=population.sizes)
        self.model.check_termination = types.MethodType(check_termination, self.model)
        self.kwargs = kwargs
        self.seed = kwargs.get("seed", None)
        self.problem2dict()
        self.termination2dict()
        self.fake_geatpy = kwargs.get("fake_geatpy", True)
        self.final_pop = kwargs.get("final_pop", True)
        self.clear_geatpy_termination()
    
    def clear_geatpy_termination(self):
        # 因为geatpy是假的过程, 所以在termination2dict后需要将这些清空
        self.MAXTIME = None
        self.MAXEVALS = None
        self.trappedValue = -1
    
    def problem2dict(self):
        self.problem_dict = {"obj_func": self.problem.evalVars}
        if self.population.Encoding == "RI":
            def split_vartypes(arr1D):
                result, sub_ls = [], []
                for i in arr1D:
                    try:
                        if i == sub_ls[0]:
                            sub_ls.append(i)
                        else:
                            result.append([sub_ls[0], len(sub_ls)])
                            sub_ls = [i]
                    except IndexError:
                        sub_ls.append(i)
                result.append([sub_ls[0], len(sub_ls)])
                return result
            vartype_size = split_vartypes(self.problem.varTypes)
            vartype_dict = {0:mealpy.FloatVar, 1:FixedIntegerVar}
            now_ind = 0
            bounds = []
            for vt, s in vartype_size:
                bound = vartype_dict[vt](lb=self.problem.lb[now_ind:now_ind+s], 
                    ub=self.problem.ub[now_ind:now_ind+s])
                now_ind += s
                bounds.append(bound)
        elif self.population.Encoding == "P":
            bounds = mealpy.PermutationVar(valid_set=[i for i in range(self.problem.lb[0], self.problem.ub[0]+1)])
        else:
            bounds = mealpy.BinaryVar(n_vars=self.problem.Dim)
        self.problem_dict["bounds"] = bounds
        self.problem_dict["minmax"] = "max" if self.problem.maxormins[0] == -1 else "min"
        self.problem_dict["log_to"] = self.kwargs.get("log_to", "console") # 此处为mealpy的log默认设置, 输出到控制台, 有需要可以自行更改
        
    def termination2dict(self):
        # geatpy MAXGEN, MAXEVALS, MAXTIME, maxTrappedCount, trappedValue
        # mealpy "max_epoch", "max_fe", "max_time", "max_early_stop" "epsilon"
        term_dict = dict()
        if self.MAXGEN is not None:
            term_dict["max_epoch"] = self.MAXGEN
        if self.MAXEVALS is not None:
            term_dict["max_fe"] = self.MAXEVALS
        if self.MAXTIME is not None:
            term_dict["max_time"] = self.MAXTIME
        if self.maxTrappedCount is not None:
            term_dict["max_early_stop"] = self.maxTrappedCount
        if self.trappedValue != 0:
            term_dict["epsilon"] = self.trappedValue
        self.term_dict = term_dict

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        population = self.population
        NIND = population.sizes
        self.initialization()  # 初始化算法类的一些动态参数
        # ===========================准备进化============================
        population.initChrom(NIND)  # 初始化种群染色体矩阵
        # 插入先验知识
        starting_solutions = None
        # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
        if prophetPop is not None:
            population = (prophetPop + population)[:NIND]  # 插入先知种群
        if prophetPop is not None: # 此处分成两步是为了适配soea_Blossoms_templet
            population.Chorm = ea.boundfix(population.Encoding, population.Chrom, population.Field)
            starting_solutions = population.decoding()
        # mealpy求解
        self.model.solve(problem=self.problem_dict, termination=self.term_dict, 
                         starting_solutions=starting_solutions, seed=self.seed)
        # ===========================开始假的geatpy进化============================
        if self.fake_geatpy:
            if self.model.history.list_current_best:
                for current_best, current_worst in zip(self.model.history.list_current_best, 
                                                       self.model.history.list_current_worst):
                    pop_Chrom = np.array([current_best.solution, current_worst.solution]).reshape((2, -1))
                    if population.Encoding == 'RI' or population.Encoding == 'P':
                        pop = ea.Population(population.Encoding, population.Field, 2, pop_Chrom)
                    else:
                        pop = ea.Population(population.Encoding, population.Field, 2, ea.ri2bs(pop_Chrom, population.Field))
                    pop.ObjV = np.array([[current_best.target.fitness], [current_worst.target.fitness]]).reshape((2, -1))
                    pop.FitnV = np.array([[1], [0]])
                    pop.Phen = pop.decoding()
                    self.evalsNum += NIND
                    self.terminated(pop)
                if self.final_pop: # 为了保留最后一代种群大小
                    if population.Encoding == 'RI' or population.Encoding == 'P':
                        final_pop = ea.Population(population.Encoding, population.Field, NIND, np.tile(np.array(self.model.g_best.solution).reshape((1, -1)), (NIND, 1)))
                    else:
                        final_pop = ea.Population(population.Encoding, population.Field, NIND, ea.ri2bs(np.tile(np.array(self.model.g_best.solution).reshape((1, -1)), (NIND, 1)), population.Field))
                    final_pop.ObjV = np.array([self.model.g_best.target.fitness] * NIND).reshape((-1, 1))
                    final_pop.FitnV = np.ones_like(final_pop.ObjV)
                    final_pop.Phen = final_pop.decoding()
                    pop = final_pop.copy()
                return self.finishing(pop)
        else:
            if self.model.history.list_current_best:
                self.evalsNum = NIND * (len(self.model.history.list_current_best) - 1)
                self.trace = {'f_best': [i.target.fitness for i in self.model.history.list_current_best], 
                              'f_avg': [(i.target.fitness + j.target.fitness) / 2 for i,j in zip(self.model.history.list_current_best,
                                                                                                 self.model.history.list_current_worst)]}
                pop = np.array(self.model.g_best.solution).reshape((1, -1))
                if population.Encoding == 'RI' or population.Encoding == 'P':
                    pop = ea.Population(population.Encoding, population.Field, 1, pop)
                else:
                    pop = ea.Population(population.Encoding, population.Field, 1, ea.ri2bs(pop, population.Field))
                pop.ObjV = np.array([self.model.g_best.target.fitness]).reshape((1, -1))
                pop.Phen = pop.decoding()
                self.BestIndi = pop
                if self.final_pop: # 为了保留最后一代种群大小
                    if population.Encoding == 'RI' or population.Encoding == 'P':
                        final_pop = ea.Population(population.Encoding, population.Field, NIND, np.tile(np.array(self.model.g_best.solution).reshape((1, -1)), (NIND, 1)))
                    else:
                        final_pop = ea.Population(population.Encoding, population.Field, NIND, ea.ri2bs(np.tile(np.array(self.model.g_best.solution).reshape((1, -1)), (NIND, 1)), population.Field))
                    final_pop.ObjV = np.array([self.model.g_best.target.fitness] * NIND).reshape((-1, 1))
                    final_pop.FitnV = np.ones_like(final_pop.ObjV)
                    final_pop.Phen = final_pop.decoding()
                    pop = final_pop.copy()
                return self.finishing(pop)
        # 求解失败 mealpy采用罚函数方法均可视为无约束, 似乎不会求解失败
        return ea.Population(population.Encoding), None
