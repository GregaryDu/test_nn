#### test_nn
#### 1) 随机初始化参数比0-初始化参数要好，更快地找到最优状态。
#### 2) 初始化的值在0附近微小的扰动是较好的，如果过大扰动，容易全分为一类。
#### 3) 单纯只有delt(W) += delta(W) 是比较慢地，可以加上moment ，现在搞了个learning-rate=10#
#### Mine-BPNN 能够得到96%左右的准确率; sklear-nn 能够得到98%左右的准确率。

#### notice: delta' = 后层传递过来  (或者直接本层做差)
####         delta  = delta' * h'   (本层导数)
####         DeltaW = delta * h_pre (上层某点)

#### test_rbm
#### 1) 实现k-step CD-approximate Gradient; k-step PCD; Parallel-Tempture;
#### 2) 很奇怪的是，估计的梯度。对参数更新时，用+号求得的mean_error更小！
#### 3) 现在用2000条数据，训练。一直不能很好地收敛，目测是样本过少，无法对联合分布很好地估计。

