Blossoms可以视为Geatpy的一个扩展，这个名字来源于曾热播的王家卫导演的电视剧《繁花》，写这个代码的时候当时正一边追该剧。

# 功能

简单来说，Blossoms提供了如下功能（这些功能和现有的Geatpy库是完全兼容的）：

1. **algorithms文件夹**内提供了基于Geatpy实现的L-SHADE、JADE、DBO、HO、GWO、PSO等单目标优化算法模板
2. 提供了Blossoms算法模板，通过该模板可以用图的方式自由组合不同的或相同的多个算法（也可以并行或者串行组合）
3. 建立了Geatpy库与Mealpy库的桥接，通过这种方式，现在可以在Geatpy中调用Mealpy库的200+算法
4. 展示了如何通过opfunu库在不同年份的CEC测试函数上评估算法
5. 提供了一些简单的demo和实验代码对Blossoms进行演示，实验代码部分提供了多进程方法和相关的指标计算和绘图代码（参考**demo文件夹**和**experiment文件夹**）
6. 在这个项目中提供了Geatpy库中适用于psy多染色体问题的optimize文件，即**psy_optimize.py**

此外，在使用geatpy、mealpy和opfunu库时发现了几处笔误的bug，需要注意（已经在相关项目上提交了issue），具体如下：
1. ~~geatpy2.7版本的Algorithm.py文件中该判断`if type(self.outFunc) != 'function':`应改成`if not callable(self.outFunc)`,github上源码已修复~~
2. mealpy3.0.1版本中整数决策变量边界修复可能有问题，此外算法的时间终止条件存在问题
+ The Mealpy library exhibits integer boundary and time termination condition errors.

In line 207 of space.py, the recommended method for boundary repair for integer decision variable types employs the `np.clip` function, which may result in inaccurate values for integer decision variables. Therefore, it is recommended to refer to the following changes:

```python
class FixedIntegerVar(mealpy.IntegerVar):
  def correct(self, x)
    x = np.clip(x, self.lb, self.ub)
    x = self.round(x)
    return np.array(x, dtype=int)
```

An error in the timing of the optimizer.py program has been identified on line 197. One possible solution to the aforementioned issue is to modify the code in the following manner:
```python
# original code 
finished = self.termination.should_terminate(epoch, 
self.nfe_counter, time.perf_counter(), es)
# modified code
finished = self.termination.should_terminate(epoch, 
self.nfe_counter, time.perf_counter() - \
self.termination.start_time, es)
```
 This modification ensures that the time calculation is accurate.
 
  3. opfunu1.0.2版本中cec2017测试集部分函数参数设置错误(1.0.3中未改正)
+ An error has been identified in the configuration parameters of Opfunu library.

In cec2017.py, the parameter setting of the test function **F10: Hybrid Function 1** is erroneous. The parameter $p$ should be set to $[0.2, 0.4, 0.4]$.In a similar manner, the parameter $p$ in **F17: Hybrid Function 8** should be set to $[0.2, 0.2, 0.2, 0.2, 0.2, 0.2]$.

As not all test functions in the Opfunu library have been individually verified, it is not possible to ascertain whether there are other errors of a similar nature. Consequently, it is necessary to perform the requisite verification when the library is actually being utilized.
# 如何使用

直接将这个项目下载到本地即可使用（可以不用下载experiment文件夹中的result和fig文件夹内容），在运行过程中缺少什么库安装什么库即可，由于Geatpy目前支持的Python版本为3.10，因此不能在高于3.10的Python中使用

建议在虚拟Python环境中安装如下库：

```shell
pip install geatpy==2.7
pip install mealpy==3.0.1
```

本着不浪费的原则，将上述简单工作整理成了一篇文章，目前正在投递中（我知道很水哈哈别骂了 包括代码也是如此～）

如果你使用了本项目的工作，目前可以参照如下方式引用

```latex
@misc{Blossoms,
	title = {Blossoms: An Extension Package for Python Evolutionary Algorithms based on the Geatpy Library},
	author = {Bardbo},
	year = {2024},
	website = {https://github.com/Bardbo/Blossoms}
}
```

>想交流本项目或者Geatpy相关等可以加V： DaLiu--_-- ，请备注来意以防不通过。

最后感谢本项目中使用到的如下工作，感谢您们的开源。

```latex
@misc{geatpy,
	title = {geatpy: The genetic and evolutionary algorithm toolbox with high performance in python},
	author = {Jazzbin, et.al.},
	year = {2020},
	website = {https://github.com/geatpy-dev/geatpy}
}

@article{van2023mealpy,
  title={MEALPY: An open-source library for latest meta-heuristic algorithms in Python},
  author={Van Thieu, Nguyen and Mirjalili, Seyedali},
  journal={Journal of Systems Architecture},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.sysarc.2023.102871}
}

@article{Van_Thieu_2024_Opfunu,
      author = {Van Thieu, Nguyen},
      title = {Opfunu: An Open-source Python Library for Optimization Benchmark Functions},
      doi = {10.5334/jors.508},
      journal = {Journal of Open Research Software},
      month = {May},
      year = {2024}
  }
```

还有CMA、es.py的作者、scienceplots等不一一列举。