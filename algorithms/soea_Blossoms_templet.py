# -*- coding: utf-8 -*-
import types, time, copy
import numpy as np
import geatpy as ea  # 导入geatpy库
from collections import namedtuple
from geatpy.core.xovbd import xovbd # JADE等算法的调用需要该算子
from scipy.stats import levy # HO算法需要该函数

AlgorithmParam = namedtuple("AlgorithmParam", ["node_name", "algorithm", "algorithm_param"])

def get_new_run_method_code(algorithm):
    """
    传入算法模板, 返回修改的run方法代码字符串
    注意:请注意该方法仅适用于形如下面代码的算法模板
    ```
    # 插入先验知识（注意：这里不会对先知种群prophetPop的合法性进行检查）
    if prophetPop is not None:
        population = (prophetPop + population)[:NIND]  # 插入先知种群
    self.call_aimFunc(population)  # 计算种群的目标函数值
    ```
    """
    import inspect, ast, textwrap
    source_code = inspect.getsource(algorithm.run)
    source_code = textwrap.dedent(source_code)
    tree = ast.parse(source_code)

    new_if_code = \
    """
    # if population.Phen is None:
    #     population.Phen = population.decoding()
    # population = (prophetPop + population)[:NIND]  # 插入先知种群
    # if prophetPop.sizes == population.sizes:
    #     population.ObjV = prophetPop.ObjV
    #     population.CV = prophetPop.CV
    # 如下代码可在缩减种群规模算法下生效
    if prophetPop.sizes <= population.sizes:
        population = prophetPop
        NIND = population.sizes
    else:
        if population.Phen is None:
            population.Phen = population.decoding()
        population = (prophetPop + population)[:NIND]  # 插入先知种群
    """
    new_if_code = textwrap.dedent(new_if_code)
    new_if_body = ast.parse(new_if_code).body

    new_expr_code = \
    """
    if population.ObjV is None:
        self.call_aimFunc(population)  # 计算种群的目标函数值
    """
    new_expr_code = textwrap.dedent(new_expr_code)
    new_expr_body = ast.parse(new_expr_code).body[0]

    def replace_if_body(node):
        if isinstance(node, ast.If):
            if isinstance(node.test, ast.Compare):
                if len(node.test.comparators) == 1 and isinstance(node.test.left, ast.Name) \
                    and node.test.left.id == "prophetPop" and isinstance(node.test.ops[0], ast.IsNot):
                    node.body = new_if_body
                    return node
        return None

    found_if = False
    for ind, node in enumerate(tree.body[0].body):
        if found_if:
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "call_aimFunc":
                tree.body[0].body[ind] = new_expr_body
                break
        else:
            if_node = replace_if_body(node)
            if if_node:
                found_if = True
                continue

    try:
        modify_code = ast.unparse(tree)
    except AttributeError as e: # python旧版本的ast可能没有unparse方法
        import astor
        modify_code = astor.to_source(tree)
    return modify_code

# 算法节点:支持每次仅运行一代; 先验种群或者个体迁移等需要修改self.population属性(带ObjV和CV)
class AlgorithmNode:
    def __init__(self, problem:ea.Problem, node_name:str, algorithm:ea.SoeaAlgorithm, param:dict, prophetPop=None) -> None:
        self.problem = problem
        self.node_name = node_name
        self.algorithm = algorithm
        self.param = param
        self.parseParam(prophetPop)
        self.BestIndi = ea.Population(None, None, 0)  # 初始化BestIndi为空的种群对象
        self.trace = {'f_best': [], 'f_avg': []}  # 重置trace
        self.currentGen = 0  # 初始为第0代
        self.evalsNum = 0 # 初始评价次数
        self.isActive = False # 当前节点是否激活（是否在运行)
        self.isFinished = False # 当前节点是否运行完成了(运行过且结束了)

    def parseParam(self, prophetPop):
        """
        1.提取种群大小、迭代次数参数
        2.初始化种群
        3.获取新的run方法
        """
        self.NIND = self.param.pop("NIND", 50)
        self.MAXGEN = self.param.pop("MAXGEN", None)
        if self.MAXGEN is None:
            self.MAXGEN = int(np.ceil(self.param.pop("MAXEVALS", 5000) / self.NIND)) # 此处没有考虑某些算法每次迭代的评价次数不等于种群大小
        self.init_population = ea.Population(Encoding='RI', NIND=self.NIND) # 用于实例化算法模板的种群
        algorithm = self.algorithm(self.problem, self.init_population, MAXGEN=1, logTras=0, **self.param)
        self.population = algorithm.population.copy() # 当代种群
        self.population.initChrom(self.population.sizes)
        self.population.Phen = self.population.decoding()
        if prophetPop is not None:
            self.population = (prophetPop + self.population)[:self.population.sizes]  # 插入先知种群
        self.run_code = get_new_run_method_code(algorithm)
        exec(self.run_code)
        self.new_run_func = locals().get('run')
    
    def stat(self, pop):
        # 进行进化记录
        feasible = np.where(np.all(pop.CV <= 0, 1))[0] if pop.CV is not None else np.arange(pop.sizes)  # 找到满足约束条件的个体的下标
        if len(feasible) > 0:
            feasiblePop = pop[feasible]
            bestIndi = feasiblePop[np.argmax(feasiblePop.FitnV)]  # 获取最优个体
            if self.BestIndi.sizes == 0:
                self.BestIndi = bestIndi  # 初始化global best individual
            else:
                delta = (self.BestIndi.ObjV - bestIndi.ObjV) * self.problem.maxormins if \
                    self.problem.maxormins is not None else self.BestIndi.ObjV - bestIndi.ObjV
                # 更新global best individual
                if delta > 0:
                    self.BestIndi = bestIndi
            # 更新trace
            self.trace['f_best'].append(bestIndi.ObjV[0][0])
            self.trace['f_avg'].append(np.mean(feasiblePop.ObjV))

    def run_once(self):
        self.isActive = True
        algorithm = self.algorithm(self.problem, self.init_population, MAXGEN=2, logTras=0, drawing=0, **self.param) # 实例化当前节点算法模板
        # 通过猴子补丁修改算法模板实例的run方法
        # 该新run方法可以传入先验种群的ObjV和CV矩阵(当先验种群数目等于种群数目时), 从而避免重复调用aimFunc
        algorithm.run = types.MethodType(self.new_run_func, algorithm)
        [optPop, self.population] = algorithm.run(self.population)
        self.stat(self.population)
        if self.currentGen + 1 >= self.MAXGEN:
            self.isActive = False
            self.isFinished = True
        self.currentGen += 1
        self.evalsNum = algorithm.evalsNum
    
    def run(self, max_gen=None): # 串行可直接运行所有代
        if max_gen is not None: # 实例化当前节点算法模板
            algorithm = self.algorithm(self.problem, self.init_population, MAXGEN=max_gen, logTras=0, drawing=0, **self.param)
        else:
            algorithm = self.algorithm(self.problem, self.init_population, MAXGEN=self.MAXGEN, logTras=0, drawing=0, **self.param)
        [self.BestIndi, self.population] = algorithm.run(self.population)
        self.trace = algorithm.trace
        self.isFinished = True
        self.currentGen = self.MAXGEN
        self.evalsNum = algorithm.evalsNum
    
    def _run_once(self):
        """
        本方法已弃用:该实现无法规避geatpy框架内的重复评价等操作
        """
        self.isActive = True
        algorithm = self.algorithm(self.problem, self.init_population, MAXGEN=2, logTras=0, drawing=0, **self.param) # 实例化当前节点算法模板
        # 这里算法模板run方法内的terminated函数会被重复调用一次, 且先验种群也会重复评价一次
        [optPop, self.population] = algorithm.run(self.population)
        self.stat(self.population)
        if self.currentGen + 1 >= self.MAXGEN:
            self.isActive = False
            self.isFinished = True
        self.currentGen += 1
        self.evalsNum = algorithm.evalsNum

class soea_Blossoms_templet(ea.SoeaAlgorithm):
    """
soea_Blossoms_templet : class - Blossoms Algorithm(繁花协同算法类)

算法类说明:
    该算法类是繁花协同算法，支持不同种群不同算法。

算法描述:
    本算法类实现的是繁花协同算法算法。算法流程如下：
    1) 初始化算法节点。
    2) 若满足进化算法停止条件则停止，否则继续执行。
    3) 循环对各个种群独立进行不同算法进化,迭代一次(按照边数据确定的先后顺序运行节点)。
    4) 如果当前代数需要种群迁移则进行种群迁移。
    5) 判断当前的种群中是否有某个节点已经运行完毕,是否有节点需要加入,运行完毕的节点需退出,加入的节点需初始化并加入。
    6) 回到第2步。
    
"""

    def __init__(self,
                 problem,
                 population,
                 algorithm_param,
                 edge_param=None,
                 mode=None,
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
        self.name = 'Blossoms'
        self.algorithm_param = copy.deepcopy(algorithm_param) # 算法节点参数 包括节点名称、算法模板、算法参数字典(至少包含种群大小和迭代次数) run方法中节点实例化会使得参数字典变成空字典
        self.edge_param = edge_param
        self.mode = mode
        self.insert_best_solution = True # 下一个算法是否插入当前最优解作为先验

        self.migFr = 5  # 发生种群迁移的间隔代数
        self.migOpers = ea.Migrate(MIGR=0.2, Structure=2, Select=1, Replacement=2)  # 生成种群迁移算子对象
        if population.Encoding != 'RI':
            raise RuntimeError('编码方式必须为''RI''.')
        self.parseParam()
    
    def parseParam(self):
        self.tight_front = dict() # 节点的紧前节点
        self.nodes_name = [node.node_name for node in self.algorithm_param] # 所有节点名称
        if self.mode != "chain": # 不是串行时
            if self.edge_param is None: # 均缺省时默认并行
                self.active_node = self.nodes_name.copy() # 并行的初始激活节点是所有节点，串行不需要初始节点(初始节点就是第一个节点)
            else: # 按照边参数执行
                for i, j in self.edge_param:
                    v = self.tight_front.get(j, [])
                    if i not in v and i != j:
                        v.append(i)
                    self.tight_front[j] = v
                self.active_node = list(set(self.nodes_name) - set(self.tight_front.keys())) # 边方式确定的初始激活节点：没有紧前节点
    
    def chainNodeModify(self):
        """
        根据算法设置的代数修改各节点迭代次数参数
        """
        node_maxgen = 0
        isExceed = False
        for name in self.nodes_name:
            node = self.nodes[name]
            temp_maxgen = node_maxgen + node.MAXGEN
            if not isExceed:
                if temp_maxgen > self.MAXGEN: # 代数超出截断
                    isExceed = True
                    node.MAXGEN = self.MAXGEN - node_maxgen
                    self.nodes[name] = node
                elif temp_maxgen == self.MAXGEN: # 代数刚好相等
                    isExceed = True
                else:
                    node_maxgen = temp_maxgen
            else:
                self.nodes.pop(name)
        if not isExceed: # 均遍历完了仍少于设置的MAXGEN, 则延长最后一个节点
            node.MAXGEN = node.MAXGEN + self.MAXGEN - node_maxgen
            self.nodes[name] = node

    def unite(self, population):
        """
        合并种群，生成联合种群。
        """
        unitePop = population[0]
        for i in range(1, self.PopNum):
            unitePop += population[i]
        return unitePop

    def calFitness(self, population):
        """
        计算种群个体适应度，population为种群列表
        该函数直接对输入参数population中的适应度信息进行修改，因此函数不用返回任何参数。
        """
        ObjV = np.vstack(list(pop.ObjV for pop in population))
        CV = np.vstack(list(pop.CV for pop in population)) if population[0].CV is not None else population[0].CV
        FitnV = ea.scaling(ObjV, CV, self.problem.maxormins)  # 统一计算适应度
        # 为各个种群分配适应度
        idx = 0
        for i in range(self.PopNum):
            population[i].FitnV = FitnV[idx: idx + population[i].sizes]
            idx += population[i].sizes

    def EnvSelection(self, population, NUM):  # 环境选择，选择个体保留到下一代
        FitnVs = list(pop.FitnV for pop in population)
        NewChrIxs = ea.mselecting('dup', FitnVs, NUM)  # 采用基于适应度排序的直接复制选择
        for i in range(self.PopNum):
            population[i] = (population[i])[NewChrIxs[i]]
        return population
    
    def chainLoggingAndDisplay(self, node):
        """
        串形(链式)的日志信息记录和打印
        """
        self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算logging的耗时
        self.BestIndi = node.BestIndi
        self.trace['f_best'] += node.trace['f_best']
        self.trace['f_avg'] += node.trace['f_avg']
        if self.logTras != 0:
            if len(self.log['gen']) == 0:  # 初始化log的各个键值
                self.log['f_opt'] = []
                self.log['f_avg'] = []
                self.log["node"] = [node.node_name] * node.MAXGEN
            else:
                self.log["node"].extend([node.node_name] * node.MAXGEN)
            self.log['gen'].extend([i for i in range(self.currentGen, self.currentGen + node.MAXGEN)])
            self.log['eval'].extend(np.linspace(self.evalsNum, 
                                                self.evalsNum + node.evalsNum, node.MAXGEN + 1)[1:].astype(int).tolist())  # 记录评价次数
            self.log['f_opt'] = self.trace['f_best']
            self.log['f_avg'] = self.trace['f_avg']
            # display
            if self.verbose:
                self.passTime += time.time() - self.timeSlot  # 更新用时记录，不计算display()的耗时
                headers = []
                widths = []
                for key in self.log.keys():
                    # 设置单元格宽度
                    if key == 'gen':
                        if self.MAXGEN is None:
                            width = 5
                        else:
                            width = max(3, len(str(self.MAXGEN - 1)))  # 因为字符串'gen'长度为3，所以最小要设置长度为3
                    elif key == 'eval':
                        width = 8  # 因为字符串'eval'长度为4，所以最小要设置长度为4
                    else:
                        width = 13  # 预留13位显示长度，若数值过大，表格将无法对齐，此时若要让表格对齐，需要自定义算法类重写该函数
                    headers.append(key)
                    widths.append(width)
                if len(self.log['gen']) == node.MAXGEN:  # 打印表头
                    header_regex = '|'.join(['{}'] * len(headers))
                    header_str = header_regex.format(*[str(key).center(width) for key, width in zip(headers, widths)])
                    print("=" * len(header_str))
                    print(header_str)
                    print("-" * len(header_str))
                for i in range(self.currentGen, self.currentGen + node.MAXGEN):
                    values = []
                    for key in self.log.keys():
                        value = self.log[key][i] if len(self.log[key]) != 0 else "-"
                        if isinstance(value, float):
                            values.append("%.5E" % value)  # 格式化浮点数，输出时只保留至小数点后5位
                        else:
                            values.append(value)
                    if len(self.log['gen']) != 0:  # 打印表格最后一行
                        value_regex = '|'.join(['{}'] * len(values))
                        value_str = value_regex.format(*[str(value).center(width) for value, width in zip(values, widths)])
                        print(value_str)
                    self.timeSlot = time.time()  # 更新时间戳
        self.evalsNum += node.evalsNum
        self.currentGen += node.MAXGEN
        self.timeSlot = time.time()  # 更新时间戳

    def run(self, prophetPop=None):  # prophetPop为先知种群（即包含先验知识的种群）
        # ==========================初始化配置===========================
        self.initialization()  # 初始化算法类的一些动态参数
        # 节点实例化
        self.nodes = {node.node_name:AlgorithmNode(self.problem, 
            node.node_name, node.algorithm, node.algorithm_param, prophetPop) for node in self.algorithm_param}
        # ===========================开始进化============================
        if self.mode == "chain": # 串行
            self.trace['f_best'] = []
            self.trace['f_avg'] = []
            if self.MAXGEN is None:
                self.MAXGEN = 100 if self.MAXEVALS is None else int(self.MAXEVALS / self.population.sizes)
                print(f"warning: 请注意串行模式下未设置MAXGEN参数, 将使用{self.MAXGEN=}")
            self.chainNodeModify() # 节点的迭代次数参数修改
            algorithm_nodes = list(self.nodes.values())
            # 运行初始节点
            algorithm_node = algorithm_nodes[0]
            algorithm_node.run()
            self.chainLoggingAndDisplay(algorithm_node) # 记录和打印日志
            for algorithm_node in algorithm_nodes[1:]:
                if self.insert_best_solution and (self.BestIndi.sizes != 0):
                    algorithm_node.population[0] = self.BestIndi # 此处仅插入上一个节点的最优解
                algorithm_node.run()
                self.chainLoggingAndDisplay(algorithm_node) # 记录和打印日志
            self.passTime += time.time() - self.timeSlot  # 更新用时记录，因为已经要结束，因此不用再更新时间戳
            self.draw(algorithm_node.population, EndFlag=True)  # 显示最终结果图
            if self.plotter:
                self.plotter.show()
            # 返回最优个体以及最后一代种群
            return [self.BestIndi, algorithm_node.population]
        else: # 并行或按照边参数来执行
            # 初始激活节点计算目标函数
            for name in self.active_node:
                pop = self.nodes[name].population
                self.call_aimFunc(pop)
                self.nodes[name].population = pop.copy()
            populations = [self.nodes[name].population for name in self.active_node]
            self.PopNum = len(populations)
            self.calFitness(populations)  # 统一计算适应度
            unitePop = self.unite(populations)  # 得到联合种群unitePop
            isEnough = True # 节点设置代数是否足够算法设置代数
            if self.log is not None:
                self.log["node"] = [self.active_node]
            while not self.terminated(unitePop):
                # 执行当前激活节点一次
                self.run_node = self.active_node.copy()
                populations = []
                for name in self.run_node:
                    algorithm_node = self.nodes[name]
                    algorithm_node.run_once()
                    self.evalsNum += algorithm_node.evalsNum
                    populations.append(algorithm_node.population)
                    if algorithm_node.isFinished and isEnough:
                        self.active_node.remove(name)
                        last_remove_name = name # 如果算法设置的MAXGEN比节点设置的要多, 则补上最后一个被移除的算法
                # 联合种群迁移等
                self.PopNum = len(populations)
                self.calFitness(populations)  # 统一计算适应度
                if (self.currentGen % self.migFr == 0) and (self.PopNum > 1):
                    before_pops_sizes = [i.sizes for i in populations]
                    populations = self.migOpers.do(populations)  # 进行种群迁移
                    after_pops_sizes = [i.sizes for i in populations]
                    if before_pops_sizes != after_pops_sizes:
                        unitePop = populations[0]
                        for i in populations[1:]:
                            unitePop = unitePop + i
                        populations = []
                        start_ind = 0
                        for i in before_pops_sizes:
                            populations.append(unitePop[np.arange(start_ind,start_ind+i)])
                            start_ind += i
                    for i, name in enumerate(self.run_node): # 遍历替换
                        self.nodes[name].population = populations[i]
                unitePop = self.unite(populations)  # 更新联合种群
                # 紧前节点全部完成则加入对应的新节点
                for k,v in self.tight_front.items():
                    algorithm_node = self.nodes[k]
                    if not algorithm_node.isFinished and not algorithm_node.isActive: # 没有运行过的节点(没有结束也没有激活)
                        active = True
                        for name in v:
                            if not self.nodes[name].isFinished:
                                active = False
                                break
                        if active:
                            if self.insert_best_solution and (self.BestIndi.sizes != 0):
                                self.nodes[k].population[0] = self.BestIndi
                            self.active_node.append(k)
                if len(self.active_node) == 0:
                    self.active_node = [last_remove_name]
                    isEnough = False
                    print(f"warning: 请注意算法设置的MAXGEN比节点设置的要多, enoughGen: {self.currentGen} MAXGEN: {self.MAXGEN}")
                if self.log is not None:
                    self.log["node"] += [self.active_node]
            self.BestIndi.Phen = self.BestIndi.decoding()
            return self.finishing(unitePop)  # 调用finishing完成后续工作并返回结果
