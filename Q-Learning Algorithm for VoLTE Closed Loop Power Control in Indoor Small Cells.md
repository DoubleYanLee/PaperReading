## Q-Learning Algorithm for VoLTE Closed Loop Power Control in Indoor Small Cells

---

### Author: Faris B. Mismar and Brian L. Evans

---

### Publisher: IEEE    

---

F. B. Mismar and B. L. Evans, "Q-Learning Algorithm for VoLTE Closed Loop Power Control in Indoor Small Cells," *2018 52nd Asilomar Conference on Signals, Systems, and Computers*, Pacific Grove, CA, USA, 2018, pp. 1485-1489, doi: 10.1109/ACSSC.2018.8645168.

---

VOLTE全称为Voice over Long-Term Evolution（长期演进语音承载）

### Abstract

*Abstract*—We propose a reinforcement learning (RL) based closed loop power control algorithm for the downlink of the voice over LTE (VoLTE) radio bearer for an indoor environment served by small cells. The main contributions of our paper are to 1) use RL to solve performance tuning problems in an indoor cellular network for voice bearers and 2) show that our derived lower bound loss in effective signal to interference plus noise ratio due to neighboring cell failure is sufficient for VoLTE power control purposes in practical cellular networks. In our simulation, the proposed RL-based power control algorithm significantly improves both voice retainability and mean opinion score compared to current industry standards. The improvement is due to maintaining an effective downlink signal to interference plus noise ratio against adverse network operational issues and faults.

*Index Terms*—reinforcement learning, artificial intelligence, VoLTE, MOS, QoE, optimization, SON.

> 提出：一种基于强化学习的闭环功率控制算法，用于小蜂窝服务的室内环境下的VoLTE承载的下行链路
>
> 贡献：(1) 使用RL来解决语音载体的室内蜂窝网络的性能调优
>
> ​		  (2) 在真实的蜂窝网络中，所推导出的有效信噪比损失下限足够满足控制电压这个目的
>
> 结果：算法在voice retainability和mean opinion score方面都比目前标准优秀。
>
> 结果的原因：保持了有效的下行信号信噪比，以应对不利的网络运行问题和故障

### Introduction

* While cellular data applications are made resilient against wireless impairments through modulation, coding, and retransmissions, delay-sensitive applications such as voice or low latency data transfer may not always benefit from retransmission since it increases delays and risk of data duplication. Therefore, these applications must be made resilient through other means.

> 虽然蜂窝数据应用程序通过调制、编码和重传具有抗无线损害的特性，但语音或低延迟数据传输等对延迟敏感的应用程序可能并不总是受益于重传，因为它增加了延迟和数据复制的风险。

* We devise an RL-based algorithm to improve downlink SINR in an indoor environment for packetized voice using power control (PC) as shown in Fig. 1. 

> 我们设计了一种基于RL的算法，通过功率控制(PC)提高室内环境下分组语音的下行SINR，如图1所示。

* 本文贡献：

1)   Use RL to solve performance tuning problems in an indoor cellular network for voice bearers.

2)   Show that our derived lower bound loss in effective SINR due to neighboring cell failure is sufficient for VoLTE power control in practical cellular networks.

>1)使用RL解决语音承载室内蜂窝网络的性能调优问题。
>2)证明了在实际的蜂窝网络中，由于邻近小区失效而导致的有效SINR的下界损耗对于语音功率控制足够的。



### SYSTEM MODEL

The system comprises two components:

1)  A radio environment where VoLTE capable UEs are served.

2)  A reinforcement learning model using Q-learning to perform closed loop power control to improve effective DL SINR measured at the receiver.

> (2)使用Q-learning来执行闭环功率控制，以在提高有效的深度学习信噪比测量在接收机端

#### Radio Environment

* OFDM（*orthogonal frequency- division multiplexing*）正交频分复用
* frequency division duple 频分双工    multiple access 多址访问
* *homogeneous poisson point process* 齐次泊松点过程
* *intensity parameter* λ 代表了每个小单元所期望的用户数
* *transmit time intervals* 传输时间间隔
* PRB（physical resource blocks ）物理资源块
* point process Φ：根据泊松分布均值 λW = λL^2 sample 出来的静止用户数N(在小区w的服务区域)
* i-th UE 坐标是从iid的连续均匀分布中sample出来的  每个cell中都有N个UE，使这些UE静止来增加通道的连接时间
* 介绍了选择方形的原因。

This cellular cluster can be in a normal state or undergo some fault-generated actions. These faults N cause the channel impairment and are tracked in a special register. We show these faults in Table I.

**We start by writing the signal model in an additive white Gaussian noise channel for our indoor system**

```c++
yi[t]= h_i[t] s_i[t]+ n[t], i=1,2,...N_UE (1)
```

yi[t] is the received signal for the i-th UE

hi[t] is a single-tap flat-fading channel. 信道

n[t] is a Gaussian random process sampled from Norm(0,σn2).

**we compute the received downlink SINR for the i-th UE at TTI t ( γDL, i[t] ) for i = 1,2,...,N_UE as:**

![Screenshot 2020-12-29 at 11.12.31 AM](/Users/yannie/Library/Application Support/typora-user-images/Screenshot 2020-12-29 at 11.12.31 AM.png)

where C is a set of all the cells in the cluster and oj is the coordinates of the j-th base station. Without loss of generality, we assume that o0 is the serving cell placed at the origin, kj ≥ 0 is the proportion of users from the adjacent cells j whose signals are transmitted on the same PRB as the i-th UE at TTI t. Those signals therefore cause *inter-cell interference* (ICI).

**The forward link budget at any TTI t is written as:**

```c++
P_UE,i[t] = P_TX[t] + G_TX − L_m − L_a,i[t] + G_UE 
```

where P_UE,i  is the received power for the allocated *physical resource blocks* (PRB) transmitted at power PTX, GTX is the antenna gain of the transmitter, Lm is a miscellaneous loss (e.g., feeder loss and return loss), La,i is the path loss over the air interface for line of sight indoor propagation for the i-th user, and GUE is the UE antenna gain.

> P_UE是功率P_tx发送的PRB的接收功率。G_tx是传输者的天线增益  L_m是各种各样的损失
>
> L_a,i是路径损失  G_UE是用户天线增益

![Screenshot 2020-12-29 at 11.26.36 AM](/Users/yannie/Library/Application Support/typora-user-images/Screenshot 2020-12-29 at 11.26.36 AM.png)

![Screenshot 2020-12-29 at 11.26.54 AM](/Users/yannie/Library/Application Support/typora-user-images/Screenshot 2020-12-29 at 11.26.54 AM.png)

该试验中需要提高的就是这个信噪比

#### Reinforcement Learning

该读这里的时候，老师说这篇文章的SINR定义不正确，不建议follow。。。。















