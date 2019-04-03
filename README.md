# SimpleRNN
COMP7606 deep learning assignment 1

05 Adam罪状二：可能错过全局最优解

深度神经网络往往包含大量的参数，在这样一个维度极高的空间内，非凸的目标函数往往起起伏伏，拥有无数个高地和洼地。有的是高峰，通过引入动量可能很容易越过；但有些是高原，可能探索很多次都出不来，于是停止了训练。

之前，Arxiv上的两篇文章谈到这个问题。

第一篇就是前文提到的吐槽Adam最狠的UC Berkeley的文章《The Marginal Value of Adaptive Gradient Methods in Machine Learning》。文中说到，同样的一个优化问题，不同的优化算法可能会找到不同的答案，但自适应学习率的算法往往找到非常差的答案（very poor solution）。他们设计了一个特定的数据例子，自适应学习率算法可能会对前期出现的特征过拟合，后期才出现的特征很难纠正前期的拟合效果。但这个文章给的例子很极端，在实际情况中未必会出现。

另外一篇是《Improving Generalization Performance by Switching from Adam to SGD》，进行了实验验证。他们CIFAR-10数据集上进行测试，Adam的收敛速度比SGD要快，但最终收敛的结果并没有SGD好。他们进一步实验发现，主要是后期Adam的学习率太低，影响了有效的收敛。他们试着对Adam的学习率的下界进行控制，发现效果好了很多。

于是他们提出了一个用来改进Adam的方法：前期用Adam，享受Adam快速收敛的优势；后期切换到SGD，慢慢寻找最优解。这一方法以前也被研究者们用到，不过主要是根据经验来选择切换的时机和切换后的学习率。这篇文章把这一切换过程傻瓜化，给出了切换SGD的时机选择方法，以及学习率的计算方法，效果看起来也不错。

这个算法挺有趣，下一篇我们可以来谈谈，这里先贴个算法框架图：

06 到底该用Adam还是SGD？

所以，谈到现在，到底Adam好还是SGD好？这可能是很难一句话说清楚的事情。去看学术会议中的各种paper，用SGD的很多，Adam的也不少，还有很多偏爱AdaGrad或者AdaDelta。可能研究员把每个算法都试了一遍，哪个出来的效果好就用哪个了。毕竟paper的重点是突出自己某方面的贡献，其他方面当然是无所不用其极，怎么能输在细节上呢？

而从这几篇怒怼Adam的paper来看，多数都构造了一些比较极端的例子来演示了Adam失效的可能性。这些例子一般过于极端，实际情况中可能未必会这样，但这提醒了我们，理解数据对于设计算法的必要性。优化算法的演变历史，都是基于对数据的某种假设而进行的优化，那么某种算法是否有效，就要看你的数据是否符合该算法的胃口了。

算法固然美好，数据才是根本。

另一方面，Adam之流虽然说已经简化了调参，但是并没有一劳永逸地解决问题，默认的参数虽然好，但也不是放之四海而皆准。因此，在充分理解数据的基础上，依然需要根据数据特性、算法特性进行充分的调参实验。

少年，好好炼丹吧。


