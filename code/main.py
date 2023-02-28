# 单次训练脚本

def run():
    from Algorithm.Algorithm import Algorithm

    algorithm = Algorithm(lead = 1,parallel=False) # 如果使用多显卡并行训练，parallel需要设置为True
    algorithm.train() # 100次迭代次数
    algorithm.test() # 测试结果

def run_crossval():
    from Algorithm.CrossValAlgorithm import Algorithm
    fold = 4

    algorithm = Algorithm(lead = 0,fold = fold,parallel=False) # 如果使用多显卡并行训练，parallel需要设置为True
    algorithm.train() # 100次迭代次数
    algorithm.test() # 测试结果

    algorithm = Algorithm(lead = 1,fold = fold,parallel=False) # 如果使用多显卡并行训练，parallel需要设置为True
    algorithm.train() # 100次迭代次数
    algorithm.test() # 测试结果

if __name__ == '__main__':
    run()
