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

class soea_psy_bridge2mealpy_templet(ea.SoeaAlgorithm):
    """
soea_psy_bridge2mealpy_templet : class - 桥接到mealpy库中单目标算法类多染色体版本

注意: 
    1.不支持决策变量上下界不能取到的情况
    2.不支持约束矩阵CV的写法,请自行使用罚函数的方法处理约束条件
    3.不支持多目标帕累托的方法, 仅支持加权的方法, 因此建议自行在evalVars函数中返回加权后的目标值
    4.evalVars函数请加上single装饰器,即只返回一个ObjV值,因为mealpy是分个体计算的,同时 假的geatpy进化 也需要调用
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
        if population.ChromNum == 1:
            raise RuntimeError('传入的种群对象必须是多染色体的种群类型。')
        for i in range(population.ChromNum):
            if population.Encodings[i] not in ["P", "RI"]:
                raise RuntimeError('编码方式必须为''RI''或''P''.')
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
        self.clear_geatpy_termination()
    
    def clear_geatpy_termination(self):
        # 因为geatpy是假的过程, 所以在termination2dict后需要将这些清空
        self.MAXTIME = None
        self.MAXEVALS = None
        self.trappedValue = -1
    
    def problem2dict(self):
        self.problem_dict = {"obj_func": self.problem.evalVars}
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
        psy_bounds = []
        for i in range(self.population.ChromNum):
            ind_i = self.population.EncoIdxs[i]
            if self.population.Encodings[i] == "RI":
                vartype_size = split_vartypes(self.problem.varTypes[ind_i])
                vartype_dict = {0:mealpy.FloatVar, 1:FixedIntegerVar}
                now_ind = 0
                bounds = []
                for vt, s in vartype_size:
                    bound = vartype_dict[vt](lb=self.problem.lb[ind_i][now_ind:now_ind+s], 
                        ub=self.problem.ub[ind_i][now_ind:now_ind+s])
                    now_ind += s
                    bounds.append(bound)
            elif self.population.Encodings[i] == "P":
                bounds = [mealpy.PermutationVar(valid_set=[i for i in range(self.problem.lb[ind_i][0], self.problem.ub[ind_i][0]+1)])]
            else:
                bounds = [mealpy.BinaryVar(n_vars=len(ind_i))]
            psy_bounds.extend(bounds)
        for i in psy_bounds:
            print(i, i.lb, i.ub)
        self.problem_dict["bounds"] = psy_bounds
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
        # 插入先验知识
        starting_solutions = None
        if prophetPop is not None:
            for i in range(population.ChromNum):
                prophetPop.Chorms[i] = ea.boundfix(population.Encodings[i], prophetPop.Chroms[i], population.Fields[i])
            population.initChrom(NIND)  # 初始化种群染色体矩阵
            population = (prophetPop + population)[:NIND]  # 插入先知种群
            starting_solutions = population.decoding()
        # mealpy求解
        self.model.solve(problem=self.problem_dict, termination=self.term_dict, starting_solutions=starting_solutions, seed=self.seed)
        # ===========================开始假的geatpy进化============================
        if self.fake_geatpy:
            if self.model.history.list_current_best:
                for current_best, current_worst in zip(self.model.history.list_current_best, self.model.history.list_current_worst):
                    pop_Chroms = []
                    now_ind = 0
                    for i in range(population.ChromNum):
                        s = len(self.population.EncoIdxs[i])
                        Chrom_i = np.array([current_best.solution[now_ind:now_ind+s], current_worst.solution[now_ind:now_ind+s]]).reshape((2, -1))
                        if population.Encodings[i] not in ["P", "RI"]:
                            Chrom_i = ea.ri2bs(Chrom_i, population.Fields[i])
                        pop_Chroms.append(Chrom_i)
                        now_ind += s
                    pop = ea.PsyPopulation(population.Encodings, population.Fields, 2, pop_Chroms)
                    pop.ObjV = np.array([[current_best.target.fitness], [current_worst.target.fitness]]).reshape((2, -1))
                    pop.FitnV = np.array([[1], [0]])
                    pop.Phen = pop.decoding()
                    self.terminated(pop)
                return self.finishing(pop)
        else:
            if self.model.history.list_current_best:
                pop_Chroms = []
                now_ind = 0
                pop = np.array(self.model.g_best.solution)
                for i in range(population.ChromNum):
                    Chrom_i = np.array(self.model.g_best.solution[now_ind:now_ind+s]).reshape((1, -1))
                    if population.Encodings[i] not in ["P", "RI"]:
                        Chrom_i = ea.ri2bs(Chrom_i, population.Fields[i])
                    pop_Chroms.append(Chrom_i)
                    now_ind += s
                pop = ea.PsyPopulation(population.Encodings, population.Fields, 1, pop_Chroms)
                pop.ObjV = np.array([self.model.g_best.target.fitness]).reshape((1, -1))
                pop.Phen = pop.decoding()
                self.BestIndi = pop
                return self.finishing(pop)
        # 求解失败 mealpy采用罚函数方法均可视为无约束, 似乎不会求解失败
        return ea.PsyPopulation(population.Encoding), None
