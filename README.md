# 第四届中国生理信号挑战赛（CPSC 2021）

---

## 简介

CPSC 2021比赛在2021年10月份左右就已经结束了，但是碍于其他的项目，一直没有时间做总结和整理。以至于拖到现在2022年的春节，放假后终于有时间整理代码和一些成果了

本人的最终成绩在Official Entries中排名第11，如果加上Unofficial Entries，则排名在第20。可以从图中看到，我们在隐藏数据集Test_II中仅获得了0.9728分，存在严重的过拟合现象

![rank](pic\readme_pic\pic_1.png)

在比赛过程中，我们一共有12次有效的提交。分数波动如图所示，在1.7到1.8之间

![score](pic\readme_pic\pic_2.png)

**注意**：代码虽然整理完了，但是还没通过测试，这部分工作过段时间再完成

## 最终方案

### 模型

受到CPSC 2018第一名的模型的启发，我们建立了一个CNN及RNN的混合模型。CNN部分一共有15个卷积层，使用shortcut连接，并添加CAMB注意力模块。RNN部分使用一个双向的GRU层获取时序关系。模型如下图所示

![model](pic\readme_pic\pic_3.png)

### 训练方法

使用physionet 2021提供的6个数据库进行预训练，由于CPSC 2021仅提供了2条导联的数据，所以同样使用physionet 2021中对应的I导联和II导联。预训练模型在model目录下的pretrain_model0.pt（导联I）和pretrain_model1.pt（导联II），然后使用预训练模型与CPSC 2021的数据进行训练

将CPSC 2021的数据按照5s一段分割数据，模型用于判断该段数据是否为房颤。最后统计房颤数据段的数量，如果房颤片段超过所有片段数量的95%，则判断该样本为持续性房颤（AFp）。同理，如果正常片段占所有片段数量的95%以上，则该样本判断为正常（N）。剩余的样本则判断为阵发性房颤（AFf）

对于开始及结束位置的判断，按照模型给出的房颤的标签开始及结束位置进行输出。如果房颤标签持续了一段时间，则认为这是一个阵发性方法，否则认为这是一个噪声，将其修改为正常标签

我们提供了2个预训练模型对应两个导联，最后判断的时候会结合两个模型的预测结果，采用置信度高的那个标签，可以看作是一种集成模型

### 一些想法

在比赛中过程中我们还考虑了一些其他的方法，例如使用平稳小波变换将信号变换到时频域上，再进行判别。最终发现该方法耗时过长，超过了时间限制

或者在不同的数据段间使用循环神经网络建立时间关系，我们使用了双向的GRU网络获取该关系，但是提高并不明显，所以最后将其去掉

此外，我们认为我们的模型对于判别正常及持续型房颤的精度比较高，但是对于判断阵发性房颤的精度比较低。因为在隐藏数据集中，阵发性房颤的数量占比较大，所以导致我们在隐藏数据集中的得分较低

集成模型对于整体的提升不算太大，可以考虑去掉

### 其他方法

CPSC会议中优秀队伍提供了部分模型结构图，我们实现了该模型，将其放在others目录下。需要注意的是，由于结构图不完整，所以模型的参数不一定与比赛队伍使用的模型一致，例如卷积核大小等参数我们通过猜测进行合理建模。该模型实现仅用于参考

## 代码目录

* code：用于存放运行代码
    * Algorithm: 用于存放训练算法
    * DataAdapter：用于存放数据读取器代码
    * Model：用于存放模型代码
* model：用于存放已经训练好的模型
* others：在比赛过程中的一些想法及其实现
    * old：未整理前的原始代码
    * pretrain：预训练代码
* out：输出的用于计算分数的json文件
* pic：调试代码过程中可视化结果
* test_record：划分的测试集记录文件
* data：原始数据文件存储目录

## 运行说明

* 运行code目录下的main即可进行单次训练
* 运行code目录下的ensemble_main即可训练集成模型
* 按照官方指引设置模型路径及数据路径，运行entry_2021.py，会在out目录下生成一系列预测结果
* 按照官方指引运行score_2021.py，即可获得评价分数

## 官网说明

### Python example code for the 4th China Physiological Signal Challenge 2021

**What's in this repository?**

We implemented a threshold-based classifier that uses the coefficient of sample entropy (cosEn) of the ECG lead signals as features. This simple example illustrates how to format your Python entry for the Challenge. However, it is not designed to score well (or, more accurately, designed not to do well), so you should not use it as a baseline for your model's performance.

The code uses two main scripts, as described below, to run and test your algorithm for the 2021 Challenge.

**How do I run these scripts?**

You can run this baseline method by installing the requirements

    pip install requirements.txt

and running 

    python entry_2021.py <data_path> <result_save_path>

where <data_path> is the folder path of the test set, <result_save_path> is the folder path of your detection results. 

**How do I run my code and save my results?**

Please edit entry_2021.py to implement your algorithm. You should save the results as ‘.json’ files by record. The format is as {‘predict_endpoints’: [[s0, e0], [s1, e1], …, [sm-1, em-2]] }. The name of the result file should be the same as the corresponding record file.

After obtaining the test results, you can evaluate the scores of your method by running

    python score_2021.py <ans_path> <result_save_path>

where <ans_path> is the folder save the answers, which is the same path as <data_path> while the data and annotations are stored with 'wfdb' format. <result_save_path> is the folder path of your detection results.

**Useful links**

- [MATLAB example code for The China Physiological Signal Challenge (CPSC2021)](https://github.com/CPSC-Committee/cpsc2021-matlab-entry)