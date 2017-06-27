## CAMALAB第一届机器学习暑期讨论班 - 内容

- 时间：7月17日 — 9月15日（9周）

### 机器学习

第一周：热身

- Python编程基础
- 图像表示：
  - 矩阵、颜色空间
  - 图像特征：像素值、颜色直方图、梯度直方图
  - 作业：图像读写、特征提取、颜色空间变换
- 回归：
  - 算法：线性回归，线性回归+正则项（L1, L2）
  - 优化：梯度下降法、牛顿法、拉格朗日
  - 测度：欧式距离、街区距离、范数、Loss
  - 性能评估：PLCC, SRCC, KRCC, RMSE, MAE
  - 作业：PM2.5预测、工资预测

第二周：分类

- 算法：k-NN, Logistic Regression, 决策树
- 作业：
  - MNIST手写数字识别、CIFAR-10图像分类
  - 采用第一周的特征 + 第二周的方法（多种组合，对比结果）

第三周：经典算法

- 支撑向量机 SVM《理解SVM的三层境界》
- 图像特征：LBP, SIFT, Visual BoW,
- 作业：CIFAR-10图像分类

第四周：无监督特征学习（Unsupervised Feature Learning, UFL）

- Andrew Ng论文
- 聚类：K-means, K-means++
- 降维：PCA, ICA, ZCA, LLE, AE
- 作业：UFL + SVM，进行CIFAR-10图像分类
- 推荐阅读：pluskid博客

扩展：

- 经典方法：谱聚类、朴素贝叶斯、EM算法、稀疏编码 Sparse Coding
- 集成学习：Adaboost, Random Forest, gdbt (XGBoost)
- 结构化学习 Structured Learning
- 排序学习 Learning to Rank
- 强化学习 Reinforcement Learning
- 模仿学习 Imitation Learning

### 深度学习

第五周：卷积神经网络 CNN

- 概念：卷积, Pooling, Stride, Padding, Data Augmentation, Learning Rate, Momentum, Softmax, ReLU, BP, SGD, Cross-Entropy Loss
- 网络：LeNet, AlexNet, VGGnet, GoogLeNet, ResNet
- 框架：Keras, Pytorch
- 作业：
  - 自己搭建浅层网络（神经网络+卷积神经网络）
    - 1）自己实现
    - 2）使用框架
    - 两个对比
  - CIFAR-10图像分类 
    - 1）UFL特征+神经网络；
    - 2）卷积神经网络

第六周：递归神经网络

- 网络：RNN, LSTM

- 作业：

  1）由Cosin预测Sin，自己编程实现网络

  2）Image Captioning，使用框架实现

扩展：
- 经典应用论文：Deep Learning 推荐阅读列表（余宙）
- 生成对抗网络：GAN, CGAN, DualGAN, CircleGAN

### 课题/项目

1. 俞俊：图像检索
2. 余宙：视觉内容自动问答（CNN, LSTM）
3. 高飞：视觉质量评价（CNN/GAN）、人脸照片-画像转换（GAN）、视觉质量增强（GAN）
4. 谭敏：图像分类（CNN）
5. 朱素果：视频跟踪