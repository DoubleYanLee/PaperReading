# A Deep-Learning-Based Radio Resource Assignment Technique for 5G Ultra Dense Networks

#### ABSTRACT 摘要

​		深度学习在网络流量控制中的应用仍然不成熟，因为很难将网络流量特征作为一个合适的输入和输出数据集来独特地描述到学习结构中。

​        替代简单的传统F/TDD进行有效的无线电资源控制至关重要

>
>
>LTE通常分为**FDD LTE**和**TDD LTE**。LTE(Long Term Evolution 长期演进 是3G和4G之间过渡的技术)
>
>**TDD，时分双工(Time Division Duplexing)**
>
>**FDD，频分双工(Frequency Division Duplexing)**

​		因为现有的动态TDD方法通常用于波束形成和基于大规模MIMO的UDNs(ultra dense networks超密集网络)的资源分配，改变了传统的上下行链路配置，容易出现重复拥塞。

​		在本文中，我们解决了这个问题，并讨论了如何利用深度LSTM(Long Short Term Memory)学习技术对UDN基站(即eNB)的交通负荷进行本地化预测。该算法在局部预测的基础上，预先执行适当的行动策略，以智能的方式避免/缓解拥塞。仿真结果表明，我们的方案在丢包率、吞吐量和MOS方面优于传统方法。

#### Introduction 引言

​		如果5G网络运营商继续使用传统的流量控制政策和方法，可能无法应对庞大和多样化的流量。		传统技术如协调多点(CoMP)传输可以用来避免/减轻UDN中可能出现的拥塞。尽管CoMP可以提高小区边缘终端的吞吐量和连接稳定性，但不同小区之间的eNB(**Evolved Node B** 演进节点  通常作为 base stations )需要根据特定的终端交换信息和调整信号。然而，有关neighboring cells的所有情况的信息并不总是可用的。要解决这个问题，在本文的proposal中，每个eNB只需要学习它自己的单元的traffic conditions or patterns

​	时分双工(TDD)传输，被认为更适合于超密集的小蜂窝网络。

​	这种基于TDD的动态常规流量控制策略的一个关键缺点: 它只考虑了当前的网络情况。这意味着，尽管在未来出现类似的交通状况时，网络可能会发生拥塞，但传统的交通控制策略无法采取任何适当的措施来预测和/或采取规避措施来避免或缓解拥塞事件。由于eNB无法智能地向终端分配无线电资源，UDN中的传统流量控制策略可能会导致重复拥塞.

​	从5G UDN的角度出发，设计智能流量控制策略至关重要。

​	对于本文的5G UDN流量管理(即无线资源控制)，我们考虑了长短期记忆(long -term memory, LSTM)结构，它是一种更复杂、更有组织的RNN结构，能够有效地预测输入数据集的上下文。之所以使用LSTM，是因为它不仅使用当前数据，而且使用过去的数据来提供输出(例如，在我们的例子中，eNB发送缓冲区中的预估包数)。因此，我们针对5G UDN提出了一种基于LSTM的新型资源控制算法。通过利用LSTM，我们提出的算法能够对来自过去和当前数据集的未来交通特征进行本地化预测。

Our proposal functions in the following steps. First, the LSTM structure is initialized. Then the appropriate dataset is prepared and formatted. This is followed by the training of the LSTM structure. Upon training, the localized prediction is made whereby the output of the LSTM struc- ture reports whether congestion will occur in the future or not. Preliminary simulation results are presented to demonstrate that the UDN resource action policy based on our proposal significantly improves the radio resource control in terms of packet loss rate, throughput, and mean opinion score (MOS).

The remainder of this article is organized as follows. The following section surveys the rele- vant research works. Then our considered system model is presented. The formal problem state- ment is also formulated. Next, our proposed deep LSTM-based intelligent traffic control policy for 5G UDN is presented. A performance compari- son of the conventional method and the proposal is provided. The article is concluded in the final section.



## system model and PRoblem FoRmulation

常规的递归神经网络并不能很好地解决长时依赖，LSTM可以很好地解决这个问题。

本文中，使用LSTM的动机为，其可以更有效地考虑UDN输入流量特性之间的联系和上下文含义。

UL/DL(upLink-downLink)

