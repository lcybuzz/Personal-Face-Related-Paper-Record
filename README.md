# Face parsing论文记录
# Under construction！
## Table of Contents
- [Deep Learning Methods](#deep-learning-methods)
- [Classical Methods](#classical-methods)
- [Datasets](#datasets)
- [Segments](#segments)
- [Others](#others)


# Deep Learning Methods
## *Hierarchical face parsing via deep learning*
**[Paper]**  Hierarchical face parsing via deep learning<Br>
**[Year]** CVPR 2012<Br>
**[Author]**   		[Ping Luo](http://personal.ie.cuhk.edu.hk/~pluo/),	[Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/),	[Xiaoou Tang](https://www.ie.cuhk.edu.hk/people/xotang.shtml)  <Br>
**[Pages]** <Br>
**[Description]** <Br>

### Guided by Detected★
**[Paper]** A CNN Cascade for Landmark Guided Semantic Part Segmentation <Br>
**[Year]** ECCV 2016 <Br>
**[Author]**   	[Aaron S. Jackson](http://aaronsplace.co.uk/), [Michel Valstar](http://www.cs.nott.ac.uk/~pszmv/), 	[Georgios Tzimiropoulos](http://www.cs.nott.ac.uk/~pszyt/) <Br>
**[Pages]** http://aaronsplace.co.uk/papers/jackson2016guided/index.html <Br>
**[Description]** <Br>
1) 提出一种用landmarks引导part segmentation的方法, 用pose-specific信息辅助分割, 分为landmark检测和分割两步;
2) landmark detection: 先用一个FCN预测68个高斯状的landmarks(68个输出channel,每个channel对应1个2D Gaussian)
3) segmentation: 将detection得到的68个channel加到输入图像上, 再用1个FCN完成分割. 这个的一个key aspect是验证集上的landmark localization error加到landmark真值上去生成2D Gaussian (没看懂他的理由???)
4) 实验部分用IoU评价, 但是没与其它方法对比, 说服力略显不足; 数据是自行从landmark数据集中生成的分割图.

### *Face Parsing via Recurrent Propagation*
**[Paper]**  Parsing via Recurrent Propagation<Br>
**[Year]** BMVC 2017<Br>
**[Author]**   	[Sifei Liu](https://www.sifeiliu.net/publication), [Jianping Shi](http://shijianping.me/), Ji Liang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/) <Br>
**[Pages]** <Br>
**[Description]** <Br>

### FC-CNN ★☆
**[Paper]** Face Parsing via a Fully-Convolutional Continuous CRF Neural Network<Br>
**[Year]** arXiv 1708 <Br>
**[Author]**   Lei Zhou, Zhi Liu, [Xiangjian He](https://www.uts.edu.au/staff/xiangjian.he) <Br>
**[Pages]** <Br>
**[Description]** <Br>
1) 粗读. 将CRF与CNN结合起来, 模型包括unary, pairwise和continuous CRF(C-CRF)三个子网络;
2) C-CRF网络首先用superpixel pooling layer将unary和pairwise网络的pixel-level feaature转化为region-level feature. 目的是保留边界信息和保证同区域标注的一致性(?)
3) 网络基于Caffe, 可以端到端训练. 未开源,可保持关注.

# Classical Methods

### *Exemplar-Based* ★
**[Paper]** Exemplar-Based Face Parsing <Br>
**[Year]** CVPR 2013 <Br>
**[Author]**   [Brandon M. Smith](http://pages.cs.wisc.edu/~bmsmith/#), [Li Zhang](http://pages.cs.wisc.edu/~lizhang/), [Jonathan Brandt](https://research.adobe.com/person/jonathan-brandt/), [Zhe Lin](https://research.adobe.com/person/zhe-lin/), [Jianchao Yang](http://www.ifp.illinois.edu/~jyang29/)	 <Br>
**[Pages]** http://pages.cs.wisc.edu/~lizhang/projects/face-parsing/ <Br>
**[Description]**  <Br>
	1) 粗读, 基于exemplar的人脸解析. 提供了一个基于Helen的人脸解析数据集

# Datasets
## Segments
http://pages.cs.wisc.edu/~lizhang/projects/face-parsing/
http://vis-www.cs.umass.edu/lfw/part_labels/
### Others
http://vis-www.cs.umass.edu/lfw/
http://www.ifp.illinois.edu/~vuongle2/helen/
https://neerajkumar.org/databases/base/databases/lfpw/
https://blog.csdn.net/chenriwei2/article/details/50631212
## Tutorials
