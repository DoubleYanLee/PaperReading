# Applications of Deep Reinforcement Learning in Communications and Networking: A Survey

## NETWORK ACCESS AND RATE CONTROL

​		物联网等现代网络在本质上变得更加去中心化和ad-hoc。在这样的网络中，传感器和移动用户等实体需要做出独立的决策，信道和基站的选择，以实现自己的目标，如吞吐量最大化。

​		但网络状态具有动态性和不确定性。

* *Dynamic spectrum access:* 动态频谱访问允许用户在本地选择信道，以最大限度地提高其吞吐量。但用户可能没有对系统的完整观察，so use DQL
* *Joint user association and spectrum access:*  User association is implemented to determine which user to be assigned to which Base Station.  (joint user association and spectrum access problems in[42][43]) 这是个非凸优化组合问题，so use DQL（提供分布式的解决方案）

> 凸函数的局部最优解就是全局最优解，在数学中的一个非凸的最优化问题也就意味着局部最优解并不是全局最优解,所以非凸函数的寻优是最难的  
>
> 因为非凸，所以要对全局都要有一个了解 。即需要接近完全和准确的网络信息来获得最优策略

* *Adaptive rate control:*  HTTP上的动态自适应流(DASH)系统，其允许客户端或用户独立选择不同比特率的视频片段下载。目标就是最大化其体验质量(QoE)。so use DQL

>不用动态规划的原因：动态规划的复杂性高，且需要完整的网络信息。

### *Network Access* 网络接入(spectrum access & user association)

>i.e. 拉丁语的id est  意为"that is"即   e.g. 拉丁语的exempligratia 举例

#### channel access

* 传感器选择M条通道来传输网络包，根据传输后的反馈，好链路 reward "+1“  不好的链路 reward"-1". 

  * 目的：找到一个最优的策略来最大化sensor’s expected accumulated discounted reward

  * 物体之前是选择短视(myopic)策略这个方案  但myopic策略需要知道system transition matrix
  * 现在用 DQN 的经验重放(experience replay)策略  
  * DQN 输入state(action & reward)  输出Q-values(action相关的Q-values)   adopt  ε-greedy policy
  * 结果：该方案的平均奖励值为4.4， 接近于myopic策略的4.5。

DQL keeps following the learned policy over time slots and stops learning a suitable policy. But IoT environments are dynamic,  the DQN in the DQL needs to be re-trained

* adaptive DQL scheme is proposed  该方案评估当前策略每一时期的累积奖励，当reward低于给定的阀值时，DQN会被重新训练去找到一个new good policy。

#### 上面的都是one sensor，现在考虑multi-sensor的场景

> joint channel selection and packet forwarding  联合通道选择和包转发

![Screenshot 2021-01-05 at 2.25.14 PM](/Users/yannie/Library/Application Support/typora-user-images/Screenshot 2021-01-05 at 2.25.14 PM.png)一个传感器作为中继(relay)，将从相邻传感器接收到的数据包转发到接收器(sink)。Relay（中继节点）有一个buffer，来存储接收到的数据包。在每个时间序列中，这个sensor选择一组通道(能最大化 (发送数据包的数量 : 发送功率) )来转发数据包。

* the sensor’s problem can be formulated as an MDP
  * action: 选择一组通道  通道上传输的数据包数量  和 调制模式
  * state： 结合buffer state 和 channel state
  * 输入是 state  输出是要选择的action
  * 传感器的效用函数是有界的，所以算法被证明是收敛的。
  * 结果：与random action selection scheme相比，该方案显著提高了系统的效用。
  * 不足：随着数据包到达率的增加，由于传感器需要消耗更多的功率来传输所有数据包，因此该方案的系统效用会降低。

#### The channel access problem in the energy harvesting-enabled IoT system

![Screenshot 2021-01-05 at 3.59.56 PM](/Users/yannie/Library/Application Support/typora-user-images/Screenshot 2021-01-05 at 3.59.56 PM.png)

	* BS作为控制器为传感器分配信道。
	* 然而，由于传感器能量可用性的不确定性，就可能使信道分配效率低下。比如：给那些能量不多的传感器分配信道是不划算的，因为他们很快就不能用了

* BS的问题是：预测传感器的🔋状态，并为channel access选择传感器，以使total rate最大化

  	* 过去：使用上行资源分配方案    缺点：该方案要求BS对所有随机过程都有 perfect non-causal knowledge。
  	* 但是传感器随机分布在一个地理区域内，所以可能无法获得 perfect non-causal knowledge。 so use DQL
  	* DQL使用由两个基于LSTM的神经网络层组成的DQN。第一层来预测传感器的电池状态，第二层利用预测的状态和通道状态信息(CSI)来确定通道访问策略。
  	* state集合包括：(1)通道访问的分配历史;  (2) 预测的电量信息历史;  (3)真实的电量信息历史;(4)传感器当前(Channel State Information)CSI。
  	* action集合包含：被选择过的传感器集合
  	* reward是：总速率和预测误差之间的差值。
  	* 结果：该方案在总速率上接近最优方法[52]，优于myopic策略[45]。此外，该方案获得的电池电量预测误差接近于零。

  #### 以上的方法都关注优化rate maximization 但在V2V系统中，延迟也要考虑

  * 每个V2V transmitter/receiver面临的问题：在约束延迟时间的情况下选择信道和发射功率，使其容量最大化。
  * DQN中each V2V transmitter的action：选择信道和选择发送功率
  * reward是：有关V2V transmitter容量和延迟的函数
  * state包括：(1)对应V2V链路的瞬时CSI  (2)前一个时隙中 V2V链路的干扰  (3)在前一时间段内，V2V发射机的邻居所选择的信道  (4)满足延迟约束的剩余时间。
  * 输入：state  action    输出：该action所得到的Q-values
  * 结果：在车辆链路有可能违反延迟约束时，来动态调整功率和信道选择。该方案对比随机信道分配方案，满足延迟约束的车辆发射机数量更多了。

  

  为了降低频谱成本，上述loT系统通常使用未授权(unlicensed)的信道。但这可能对现有的网络产生干扰。

  >什么叫 unlicensed channel？

  ####  (这个应用不太懂)利用DQN将动态信道接入和干扰管理问题一起都解决：

  ![Screenshot 2021-01-05 at 5.53.46 PM](/Users/yannie/Library/Application Support/typora-user-images/Screenshot 2021-01-05 at 5.53.46 PM.png)

  > SBS(Small Base Station).  LTE network: Long-Term Evolution(一个标准) network

  在每个时隙，SBS选择一个通道来传输其数据包。但是，所选通道上可能有WLAN通信，因此SBS概率性地访问所选通道。

  * SBS的action：信道选择和概率性地访问信道
  * SBS的问题是：确定一个action vector，以便在所有通道和时间段内最大限度地提高其总吞吐量，即最大化效用函数。 
  * **（这里为什么又谈及资源分配）**资源分配问题可以表述为一个非合作博弈（non-cooperative game），利用基于LSTM的DQN可以求解该博弈。
  * DQN输入：该通道上SBSs和WLAN的历史流量    输出： SBSs的预测action vector

  The utility function of each SBS is proved to be convex, and thus the DQN-based algorithm converges to a Nash equilibrium of the game. 

  >证明了每个SBS的效用函数是凸的，因此基于DQN的算法收敛于博弈的纳什均衡。

  * 结果：与标准Q-learning相比，该方案的平均吞吐量提高了28%。

  ##### (这个也不是很懂)多用户共享K个信道的动态频谱访问问题

  在某个时隙，用户以一定的尝试概率选择信道或选择根本不传输数据包。

  1)  state: 用户历史的action和当前的obeservation

  2)  用户的策略是: mapping from the history to an attempt probability. 

  3) 问题：找到一个策略向量，也就是policy, 从而maximize its expected accumulated discounted data rate of the user

  以上的问题训练一个DQN来解决

  * 输入：past actions 和 the corresponding observations.    输出：estimated Q-values of the actions
  * 为了避免Q-learning的过高估计，我们使用DDQN(dueling DQN)来解决这个问题。

  the multichannel random access is modeled as a non-cooperative game, the game has a subgame perfect equilibrium.

  Note that some users can keep increasing their attempt probability to increase their rates. This makes the equilibrium point inefficient, and thus the strategy space of the users is restricted to avoid the situation.

  用户的这种策略空间(不断增加尝试的可能性，以提高其成功率)是被限制的

  * 结果：该方案的信道吞吐量是slotted-Aloha [56]的两倍。原因是，在该方案中，每个用户仅从其局部观察中学习，没有在线协调或载波感知

  

  **在上述模型中，用户数量在所有时间段都是固定的，不考虑新用户的到来。**

  该系统的问题是找到一种信道分配决策，使新UT(User Terminals)在时间段内的总服务阻塞概率最小，同时又不会对当前UT造成干扰。

  The system’s problem can be viewed as a temporal correlated sequential decision-making optimization problem.

  * Agent: satellite system
  * Action： is an index indicating which channel is allocated to the new arrived UT.
  * state集合:  current UTs, the current channel allocation matrix, and the new arrived UT.(由于同信道干扰，状态具有空间相关特征,所以可以用image tensor来表示。因此，DQN采用CNN来提取状态的有用特征)
  * reward: is positive when the new service is satisfied and is negative when the service is blocked
  * 结果：通过将可用信道分配给新到达的UTs，与固定信道分配方案相比，该方案可以将系统流量提高24.4%。
  * 不足：随着UTs数目的增加，可用通道数目很低，甚至为零。此时，所提方案的动态信道分配决策变得毫无意义，两种方案之间的性能差异变得不显著。在未来的工作中，可以研究一种基于DQL的信道和功率联合分配算法。(a joint channel and power allocation algorithm based on the DQL can be investigated.)

  ### *Joint User Association and Spectrum Access*

  * Joint user association and spectrum access problems 是典型的非凸优化问题

  >以前采用了线性规划等传统方法来获得最优解。但这些方法几乎需要知道完整并且准确的网络信息，而这通常无法达到的。
  >
  >所以使用Q-learning算法。然而，由于joint optimization problem存在较大的state空间和action空间，因此获得最优解具有很大的挑战性
  >
  >所以用DQN

  ![Screenshot 2021-01-05 at 6.58.41 PM](/Users/yannie/Library/Application Support/typora-user-images/Screenshot 2021-01-05 at 6.58.41 PM.png)

  * 每个用户的问题是，在保证用户的信噪比(SINR)高于最低服务质量(QoS)要求的同时，选择一个BS和一个channel，使其数据速率最大化。

  

  

  

  

  

  

  

  

  

  

  

  

  

















