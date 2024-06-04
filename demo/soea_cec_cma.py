import cma
import opfunu
import matplotlib.pyplot as plt

"""
使用cma库运行CEC测试问题的示例
"""

# 通过opfunu构建CEC测试问题
FUN_NAME = "F12022"
ndim = 10

funcs = opfunu.get_functions_by_classname(FUN_NAME)
func = funcs[0](ndim=ndim)

def evalVars(Vars):  # 定义目标函数
    f = func.evaluate(Vars)
    return f

# 算法参数配置 也可以直接传入字典
# opts = cma.CMAOptions()
# opts.set('bounds', [-100, 100])
# opts.set('popsize', 50)
# opts.set('maxiter', 200)

opts = {
    'bounds': [-100, 100],
    'popsize': 50,
    'maxiter': 200,
    'verb_log': 0,
    'verb_disp': 1,
}

# 初始化数据记录列表
iterations = []
f_values = []

# 初始化
es = cma.CMAEvolutionStrategy(ndim * [0], 0.5, opts)

while not es.stop():
    solutions = es.ask()
    es.tell(solutions, [evalVars(x) for x in solutions])
    es.logger.add()  # write data to disc to be plotted
    es.disp()

    # 记录迭代次数、目标函数值和当前最佳解
    iterations.append(es.countiter)
    f_values.append(es.result.fbest)

es.result_pretty()

# cma.plot()  # shortcut for es.logger.plot()
# cma.s.figsave(f'cma_{FUN_NAME}.png')

# 绘制迭代图
plt.plot(iterations, f_values)
plt.title('Optimization Progress')
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.show()