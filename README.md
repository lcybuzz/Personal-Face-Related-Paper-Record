# Face parsing Paper Record
# Under construction！
# Table of Contents
- [Deep Learning Methods](#deep-learning-methods)
	- [Face Parsing](#face-parsing)
	- [Face Detection](#face-detection)
- [Classical Methods](#classical-methods)
- [Datasets](#datasets)
- [Librarys](#librarys)
- [Resources-Lists](#resources-lists)

# Deep Learning Methods

## Face Parsing

### *Hierarchical face parsing via deep learning*
**[Paper]**  Hierarchical face parsing via deep learning<Br>
**[Year]** CVPR 2012<Br>
**[Author]**   		[Ping Luo](http://personal.ie.cuhk.edu.hk/~pluo/),	[Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/),	[Xiaoou Tang](https://www.ie.cuhk.edu.hk/people/xotang.shtml)  <Br>
**[Pages]** <Br>
**[Description]** <Br>

### MO-GC ★
**[Paper]**  Multi-Objective Convolutional Learning for Face Labeling<Br>
**[Year]** CVPR 2015<Br>
**[Author]** [Sifei Liu](https://www.sifeiliu.net/), [Jimei Yang](https://eng.ucmerced.edu/people/jyang44), Chang Huang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/)  <Br>
**[Pages]** https://www.sifeiliu.net/face-parsing <Br>
**[Description]** <Br>
1) 模拟CRF, 提出一种用多个目标函数优化一个CNN的人脸解析方法. 一个loss针对unary label likehood, 一个loss针对pairwise label dependency;
2) 提出一种nonparametric prior作为global regularization. 首先在脸部key point真值图像块上基于PCA建立一形状子空间, 测试时搜索与测试图像最相似的若干真值图像, 根据key point将真值图像与测试图像align，将几张aligned后的mask取平均作为prior;
3) 在LFW和Helen上实验, 多目标函数的策略对精度有微小提升, nonparametric prior效果提升明显：

### Guided by Detected ★★
**[Paper]** A CNN Cascade for Landmark Guided Semantic Part Segmentation <Br>
**[Year]** ECCV 2016 <Br>
**[Author]**   	[Aaron S. Jackson](http://aaronsplace.co.uk/), [Michel Valstar](http://www.cs.nott.ac.uk/~pszmv/), 	[Georgios Tzimiropoulos](http://www.cs.nott.ac.uk/~pszyt/) <Br>
**[Pages]** http://aaronsplace.co.uk/papers/jackson2016guided/index.html <Br>
**[Description]** <Br>
1) 提出一种用landmarks引导part segmentation的方法, 用pose-specific信息辅助分割, 分为landmark检测和分割两步;
2) landmark detection: 先用一个FCN预测68个高斯状的landmarks(68个输出channel,每个channel对应1个2D Gaussian)
3) segmentation: 将detection得到的68个channel加到输入图像上, 再用1个FCN完成分割. 这个的一个key aspect是验证集上的landmark localization error加到landmark真值上去生成2D Gaussian (没看懂他的理由???)
4) 实验部分用IoU评价, 但是没与其它方法对比, 说服力略显不足; 数据是自行从landmark数据集中生成的分割图.

### CnnRnnGan ★
**[Paper]**  End-to-end semantic face segmentation with conditional random fields as convolutional, recurrent and adversarial networks<Br>
**[Year]** arXiv 1703<Br>
**[Author]** [Umut Güçlü](http://guc.lu/), Yagmur Güçlütürk, Meysam Madadi, Sergio Escalera, Xavier Baró, Jordi González, Rob van Lier, Marcel van Gerven <Br>
**[Pages]** https://github.com/umuguc (还没开源)<Br>
**[Description]** <Br>
1) 大致浏览. 本文提出了一个大杂烩, 将dilation, CRFasRNN, adversarial training整合到一end to end的框架中. 不过, 首先要检测landmark, 将landmark连接生成初始分割图, 再用landmark将输入图像和分割图与模板对齐.<Br>
2) 效果较好, 但暂时未开源.<Br>
3) 有一个问题没细看: 在Helen上实验时, 是分别训练了5个网络解析不同类别吗??<Br>

### RNN-G ★☆
**[Paper]**  Parsing via Recurrent Propagation<Br>
**[Year]** BMVC 2017<Br>
**[Author]**   	[Sifei Liu](https://www.sifeiliu.net/), [Jianping Shi](http://shijianping.me/), Ji Liang, [Ming-Hsuan Yang](http://faculty.ucmerced.edu/mhyang/) <Br>
**[Pages]** <Br>
**[Description]** <Br>
1) 提出一种coarse to fine的人脸解析方法, 第一阶段解析出皮肤, 头发和背景, 第二部解析facial components. CNN和RNN参数都较少, 因此速度很快. <Br>
2) 第一阶段将CNN的hierarchical representation和RNN的label propagation结合起来. CNN有两个输出, 一个是feature map, 另一个是edge map. RNN考虑上下左右四个方向, 以feature map为输入, 并用edge map作为gate, 即边缘处两个node的联系应该小, 相同类别区域两个node联系应该大.<Br>
3) 第二个阶段设计了eye/eyebrow, nose和mouth三个子网络, 根据landmark将五官crop成patch, 送入相应的网络进行解析. <Br>
4) 本文也需要额外的landmark检测, 检测出的landmard用于将脸转正和crop五官. <Br>

### FC-CNN ★☆
**[Paper]** Face Parsing via a Fully-Convolutional Continuous CRF Neural Network<Br>
**[Year]** arXiv 1708 <Br>
**[Author]**   Lei Zhou, Zhi Liu, [Xiangjian He](https://www.uts.edu.au/staff/xiangjian.he) <Br>
**[Pages]** <Br>
**[Description]** <Br>
1) 将CRF与CNN结合起来, 模型包括unary, pairwise和continuous CRF(C-CRF)三个子网络; 网络基于Caffe, 可以端到端训练. 未开源,性能较好.<Br>
2) Unary net采用类似SegNet的结构. pairwise net将相邻像素的feature连接起来并用1*2和2*1的卷积得到其水平和垂直方向的相似的, 最后得到相似度矩阵.<Br>
3) C-CRF网络首先用superpixel pooling layer将unary和pairwise网络的pixel-level feaature转化为region-level feature. 目的是保留边界信息和保证同区域标注的一致性(?). 再使用unary和pairwise的超像素特征构成目标能量函数.<Br>
4) 介绍了一种端到端训练C-CRF的方法, 没细看.<Br>
3) 貌似应该需要额外的方法得到超像素.<Br>

## Face Detection

### MTCNN ★★
**[Paper]**  Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks<Br>
**[Year]** SPL 2016 <Br>
**[Author]**   	[Kaipeng Zhang](http://kpzhang93.github.io/), Zhanpeng Zhang, Zhifeng Li, [Yu Qiao](http://mmlab.siat.ac.cn/yuqiao/) <Br>
**[Pages]**  https://kpzhang93.github.io/MTCNN_face_detection_alignment/ <Br>
**[Description]** <Br>
1) 以3个CNN级联的方式，完成coarse到fine的人脸检测和对齐;
2) 三个网络分别为Prposal(P)-Net, Refine(R)-Net和Output(O)-Net, 三个网络都是结构相似的小型CNN，总体速度较快;
3) 网络的训练包括三个task: 人脸分类(是否是人脸的二分类问题), bounding box回归, landmark定位. 三个任务是分别取样本和训练的;
4) 提出online hard sampling mining, 在一个mini-batch中对每个sample的loss排序, 只取loss由大到小前70%的sample参与back propagation

### SSH ★★
**[Paper]**  WIDER FACE: A Face Detection Benchmark<Br>
**[Year]** ICCV 2017 <Br>
**[Author]** [Mahyar Najibi](http://legacydirs.umiacs.umd.edu/~najibi/), [Pouya Samangouei](https://po0ya.github.io/), [Rama Chellappa](http://legacydirs.umiacs.umd.edu/~rama/), [Larry S. Davis](http://legacydirs.umiacs.umd.edu/~lsd/) <Br>
**[Pages]** https://github.com/mahyarnajibi/SSH <Br>
**[Description]** <Br>
1) Single stage, no head of classification network<Br>
2) Scale-invariant by design, detect faces from various depths<Br>

## Landmark Detection

### CNN_FacePoint
**[Paper]**  Deep Convolutional Network Cascade for Facial Point Detection <Br>
**[Year]** CVPR 2013 <Br>
**[Author]** Yi Sun, [Xiaogang Wang](http://www.ee.cuhk.edu.hk/~xgwang/), [Xiaoou Tang](https://www.ie.cuhk.edu.hk/people/xotang.shtml)<Br>
**[Pages]** http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm <Br>
**[Description]** <Br>
	
### TCDCN
**[Paper]**  Facial Landmark Detection by Deep Multi-task Learning <Br>
**[Year]** ECCV 2014 <Br>
**[Author]** [Zhanpeng Zhang](https://zhzhanp.github.io/), [Ping Luo](http://personal.ie.cuhk.edu.hk/~pluo/), [Chen Change Loy](http://personal.ie.cuhk.edu.hk/~ccloy/), [Xiaoou Tang](https://www.ie.cuhk.edu.hk/people/xotang.shtml)<Br>
**[Pages]** http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html <Br>
**[Description]** <Br>

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
Helen	http://pages.cs.wisc.edu/~lizhang/projects/face-parsing/ <Br>
LFW	http://vis-www.cs.umass.edu/lfw/part_labels/ <Br>
	
## Others
LFW	http://vis-www.cs.umass.edu/lfw/ <Br>
Helen	http://www.ifp.illinois.edu/~vuongle2/helen/ <Br>
LFPW	https://neerajkumar.org/databases/base/databases/lfpw/ <Br>
WIDER	http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/ <Br>
http://www.face-rec.org/databases/ <Br>
	
# Librarys	
libfacedetection	https://github.com/ShiqiYu/libfacedetection
	
# Resources-Lists	
https://blog.csdn.net/chenriwei2/article/details/50631212 <Br>


