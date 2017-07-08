#### test_nn_single
#### 1) 随机初始化参数比0-初始化参数要好，更快地找到最优状态。
#### 2) 初始化的值在0附近微小的扰动是较好的，如果过大扰动，容易全分为一类。
#### 3) 单纯只有delt(W) += delta(W) 是比较慢地，可以加上moment ，现在搞了个learning-rate=10#
#### Mine-BPNN 能够得到96%左右的准确率; sklear-nn 能够得到98%左右的准确率。

#### notice: delta' = 后层传递过来  (或者直接本层做差)
####         delta  = delta' * h'   (本层导数)
####         DeltaW = delta * h_pre (上层某点)
'''

#### test_bpnn_epoch
#### 1) 使用mini-batch training 
#### 2) 模块化初始化，前向传播过程，后向传播过程，并存储中间结果 
#### 3) 1-hidden-layer 是比较好优化，2层就比较困难了# hidden_nodes_list=[250, 100], batch_size=50, lr==2.0 #
#### 4) 用tensorflow，想借助下其Optimization.AutoGradient，结果不很友好，对Input-Output这种的是很友好地AutoGradient。
####    中间加入层级关系，如何有效使用，暂时还没有发现，估计有只是不知道而已 ##


#### test_rbm
#### 1) 实现k-step CD-approximate Gradient; k-step PCD; Parallel-Tempture;
#### 2) 很奇怪的是，估计的梯度。对参数更新时，用+号求得的mean_error更小！
####    这里理解出现问题了，这里的最大似然估计就没有取负求最小值，而是直接求最大值。当然用+delt了。
#### 3) 现在用2000条数据，训练。一直不能很好地收敛，目测是样本过少，无法对联合分布很好地估计。
####    修改为了全部训练数据，在每个mini-batch上的训练mean-error会很小，但是在测试集上的mean-error就是0.22.
####    为什么,测试集上这么高的mean-error，是因为取1的条件写错了，训练是127，测试集用的是1.我靠。统一为127阈值后，就都非常小了。
#### 4)some para: hidden=20/40; pix>127,1,0; learning-rate=[0.5,8]; batch-size=[100,2000]这个参数设置是对X:Y作为vision-node时的。
#### 5)some para: hidden=40;    pix>127,1,0; learning-rate=0.5;     batch-size=100; 这个参数设置是对仅仅X作为vision-node时的。
####   上述两组参数差距为什么这么大呢，同样是mean-error可以收敛到很小的值。目测是与数据的洗漱有关系，Y部分是密集稀疏，相对于x。
####   这个地方值得好好想一下，这个RBM在什么情况，最容易达到极低的误差???


#### 关于X:Y作为训练数据，然后只给出X部分，采样Y部分的实验，怎么训练都不能得出很好的效果，一直都是瞎猜的准确率0.1。
#### 这就很尴尬了，Asja Fischer等人 在《Training RBM: An Introduction》里，给出了这样的思路。但是自己也没有这么搞个实验结果，贴上。
#### 为啥作者，不弄个类似的实验结果。难道，真的是这么玩不是正确的姿势。
#### 倒是给了个中间feature的样子。 哎 ╮(╯▽╰)╭
#### 后面想想怎么用，是不是可以试下 只用prob做输入，连续化。不再用现在的二元随机状态值。！
