#### test_nn
#### 1) 随机初始化参数比0-初始化参数要好，更快地找到最优状态。
#### 2) 初始化的值在0附近微小的扰动是较好的，如果过大扰动，容易全分为一类。
#### 3) 单纯只有delt(W) += delta(W) 是比较慢地，可以加上moment ，现在搞了个learning-rate=10#
#### Mine-BPNN 能够得到96%左右的准确率; sklear-nn 能够得到98%左右的准确率。

#### notice: delta' = 后层传递过来  (或者直接本层做差)
####         delta  = delta' * h'   (本层导数)
####         DeltaW = delta * h_pre (上层某点)

#### some para: hidden=20/40; pix>127,1,0; learning-rate=[0.5,8]; 
