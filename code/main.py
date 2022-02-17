# 单次训练脚本

from Algorithm.Algorithm import Algorithm

algorithm = Algorithm(parallel=False) # 如果使用多显卡并行训练，parallel需要设置为True
algorithm.train() # 100次迭代次数
algorithm.test() # 测试结果