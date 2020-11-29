># CH01 邂逅openCV

>>## 1.1 OpenCV 周边概念认知
####  图像 处理、计算机视觉与OpenCV
图像处理( Image Processing)是用计算机对图像进行分析，以达到所需结果的技术，又称影像处理。图像处理技术一般包括图像压缩，增强和复原，匹配、描述和识别3个部分。图像处理一般指 数字图像处理( Digital Image Processing)。其中，数字图像是指用工业相机、摄像机、扫描仪等设备经过拍摄得到的一个大的二维数组。该数组的元素称为像素，其值称为灰度值。而数字图像处理是通过计算机对图像进行去除噪声、增强、复原、分割、提取特征等处理的方法和技术。

计算机视觉(Computer Vision) 是一门研究如何使机器“看”的科学，具体地说，就是是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一做图形处理，用电脑处理使之成为更适合人眼观察或传送给仪器检测的图像的一门学科。作为一门科学学科，计算机视觉研究相关的理论和技术，试图建立能够从图像或者多维数据中获取“信息”的人工智能系统。因为感知可以看做是从感官信号中提取信息，所以计算机视觉也可以看做是研究如何使人工系统从图像或多维数据中“感知”的科学。

####图像处理和计算机视觉的区别

- 图像处理侧重于“处理”图像一如增强，还原，去噪，分割，等等;
- 而计算机视觉重点在于使用计算机( 也许是可移动式的)来模拟人的视觉，因此模拟才是计算机视觉领域的最终目标。
- 而OpenCV, 是一个基于开源发行的跨平台计算机视觉库，它实现了图像处理和计算机视觉方面的很多通用算法。

>>## 1.2 OpenCV基本架构分析
#### OpenCV的所有模块及功能
1.[calib3d]- Calibration (校准)和3D这两个词的组合缩写。这个模块主要是相机校准和三维重建相关的内容，包括基本的多视角几何算法、单个立体摄像头标定、物体姿态估计、立体相似性算法、3D信息的重建等。

2.[contrb]- Contributed/Experimental Stuf 的缩写。该模块包含了一些最近添加的不太稳定的可选功能，不用去多管。新增了新型人脸识别、立体匹配、人工视网膜模型等技术。

3.[core]- 核心功能模块， 包含如下内容:

- OpenCV 基本数据结构 
-  动态数据结构
-  绘图函数 
-  数组操作相关函数 
-   辅助功能与系统函数和宏 
-    与OpenGL的互操作

4.[imgproc]- lmage 和Process这两个单词的缩写组合，图像处理模块。包含如下内容:
- 线性和非线性的图像滤波 
- 图像的几何变换 
- 其他(Miscellaneous) 图像转换 
- 直方图相关 
- 结构分析和形状描述 
- 运动分析和对象跟踪 
- 特征检测 
- 目标检测等内容

5.[features2d]- 也就是Features2D, 即2D功能框架，包含如下内容:
- 特征检测和描述 
- 特征检测器(Feature Detectors) 通用接口 
- 描述符提取器(Descriptor Extractors)通用接口 
- 描述符匹配器(Descriptor Matchers)通用接口 
- 通用描述符(Generic Descriptor)匹配器通用接口 
- 关键点绘制函数和匹配功能绘制函数 

6.[flann]- Fast Library for Approximate Nearest Neighbors,高维的近似近邻快速搜索算法库，包含以下两个部分:

- 快速近似最近邻搜索 
- 聚类

7.[gpu]- 运用 GPU加速的计算机视觉模块。

8.[highgui]- 高层 GUI图形用户界面，包含媒体的输入输出、视频捕捉、图像和视频的编码解码、图形交互界面的接口等内容。

9.[legacy]- 一些已经废弃的代码库，保留下来作为向下兼容，包含如下内容:

- 运动分析 
- 期望最大化 
- 直方图 
- 平面细分(C API) 
- 特征检测和描述( Feature Detection and Description) 
- 描述符提取器(Descriptor Extractors)的通用接口 
- 通用描述符(Generic Descriptor Matchers)的常用接口 
- 匹配器

10.[ml]一Machine Learning，机器学习模块，基本上是统计模型和分类 算法，包含如下内容: 
- 统计模型(Statistical Models) 
- 一般贝叶斯分类器(Normal Bayes Classifier) 
- K-近邻(K-Nearest Neighbors) 
- 支持向量机( Support Vector Machines) 
- 决策树(Decision Trees) 
- 提升(Booting) 
- 梯度提高树(Gradient Boosted Trees) 
- 随机树(Random Trees ) 
- 超随机树(Extremely randomized trees ) 
- 期望最大化( Expectation Maximization) 
- 神经网络(Neural Networks) 
- MLData

11.[nonfree]- 一些具有专利的算法模块，包含特征检测和GPU相关的内容。最好不要商用。

12.[objdetect]- 目标检测模块，包含Cascade lassification (级联分类)和Latent SVM这两个部分。

13.[ocl]- OenCL-acelerated Computer Vision,运用OpenCL加速的计算机视觉组件模块。

14.[photo]- Computational Photography,包含图像修复和图像去噪两部分。

15.[stitching]- images stitching,图像拼接模块，包含如下部分:

- 拼接流水线 
- 特点寻找和匹配图像 
- 估计旋转 
- 自动校准 
- 图片歪斜 
- 接缝估测 
- 曝光补偿 
- 图片混合

16.[superres]- SuperResolution, 超分辨率技术的相关功能模块。

17.[ts]- OpenCV 测试相关代码，不用去管。

18.[video]- 视频分 析组件，该模块包括运动估计、背景分离、对象跟踪等视频处理相关内容。

19.[Videostab]- Video stabilization,视频稳定相关的组件。

>>## 1.3 OpenCV带来了什么



>>## 1.4 OpenCV的下载，安装与配置

在官网上下载社区版本的visual studio 2019，然后根据教程进行配置
配置完成后进行代码测试，安装好后打开界面如下
[![DdSHxJ.png](https://s3.ax1x.com/2020/11/25/DdSHxJ.png)](https://imgchr.com/i/DdSHxJ)
例如下代码，运行出图像说明安装配置成功，就可以进行更多的实验啦
[![Dd986H.png](https://s3.ax1x.com/2020/11/25/Dd986H.png)](https://imgchr.com/i/Dd986H)
[![DdClEq.png](https://s3.ax1x.com/2020/11/25/DdClEq.png)](https://imgchr.com/i/DdClEq)
>>## 1.5 快速上手OpenCV图像处理
#### 图像显示
上面在安装配置好OpenCV后已经实验过图像显示，这里就不再重复了
#### 图像腐蚀
代码如下
[![Dd96ns.png](https://s3.ax1x.com/2020/11/25/Dd96ns.png)](https://imgchr.com/i/Dd96ns)
图像原图和腐蚀对比
[![DdCJ8U.png](https://s3.ax1x.com/2020/11/25/DdCJ8U.png)](https://imgchr.com/i/DdCJ8U)
[![DdCD56.png](https://s3.ax1x.com/2020/11/25/DdCD56.png)](https://imgchr.com/i/DdCD56)
#### 图像模糊
[![DdCxI0.png](https://s3.ax1x.com/2020/11/25/DdCxI0.png)](https://imgchr.com/i/DdCxI0)
#### canny边缘检测
<a href="https://imgchr.com/i/DdPVd1"><img src="https://s3.ax1x.com/2020/11/25/DdPVd1.md.png" alt="DdPVd1.png" border="0" /></a>

>>## 1.6 OpenCV视频操作基础
#### 播放视频
[![DdPwQg.png](https://s3.ax1x.com/2020/11/25/DdPwQg.png)](https://imgchr.com/i/DdPwQg)
#### 调用摄像头
[![DdPowR.png](https://s3.ax1x.com/2020/11/25/DdPowR.png)](https://imgchr.com/i/DdPowR)
># CH02 启程前的认知准备
#### 彩色目标跟踪
[![DdFVC6.png](https://s3.ax1x.com/2020/11/25/DdFVC6.png)](https://imgchr.com/i/DdFVC6)
#### 点追踪
[![DdQRsI.jpg](https://s3.ax1x.com/2020/11/25/DdQRsI.jpg)](https://imgchr.com/i/DdQRsI)
#### 人脸识别
[![DdQoFS.png](https://s3.ax1x.com/2020/11/25/DdQoFS.png)](https://imgchr.com/i/DdQoFS)
># CH03 HighGUI图形用户界面初步
#### 利用imwrite生成透明png图像
[![DdQqQs.jpg](https://s3.ax1x.com/2020/11/25/DdQqQs.jpg)](https://imgchr.com/i/DdQqQs)
#### OpenCV的命名空间
OpenCV中的C++类和函数都是定义在命名空间cv之内的，有两种方法可以访问:第一种，是在代码开头的适当位置加上 usingnamespace cv;这句代码，规定程序位于此命名空间之内;另外一种，是在使用OpenCV的每一个类和函数时，都加入cv::命名空间。不过这种情况会很繁琐，每用一个OpenCV的类或者函数，都要多敲四下键盘写出cv::。比如在写简单的OpenCV程序的时候，以下三句可以作为标配:
 #include <opencv2/core/core.hpp>
 #include<opencv2 / highgui /highgui.hpp>using namespace cv; 

#### Mat类简析
Mat类是用于保存图像以及其他矩阵数据的数据结构，默认情0。我们也可以指定其初始尺寸，比如定义一个Mat类对象，就要写(320,640,cv::Scalar( 100));
Mat类型作为OpenCV2、OpenCV3新纪元的重要代表，在稍后者会花长篇幅详细讲解它，现在我们只要理解它是对应于OpenCIplImage，主要用来存放图像的数据结构就行了。对于本节，我们寻Mat的其实就简单的这样一句代码:
Mat srcImage= imread ( "dota.jpg" );

#### Mat类简析
Mat类是用于保存图像以及其他矩阵数据的数据结构，默认情况下其尺寸为0。我们也可以指定其初始尺寸，比如定义一个Mat类对象，就要写cv::Mat pic(320,640,cv::Scalar(100));
Mat类型作为OpenCV2、OpenCV3新纪元的重要代表，在稍后的章节中，笔者会花长篇幅详细讲解它，现在我们只要理解它是对应于OpenCV1.0时代的Ipllmage，主要用来存放图像的数据结构就行了。对于本节，我们需要用到关于Mat的其实就简单的这样一句代码:
Mat srcImage= imread ( "dota.jpg" );
#### 图像的载人与显示概述
在新版本的OpenCV2中，最简单的图像载入和显示只需要两句代码，非常便捷。这两句代码分别对应了两个函数，它们分别是imread()以及imshow()。
#### 图像的载入，显示和输出
学习过以往版本OpenCV的读者应该都清楚,对于OpenCV1.0时代的基于C语言接口而建的图像存储格式: IplImage*，如果在退出前忘记 release掉的话，会造成内存泄露，而且用起来十分繁琐。我们在 debug 程序的时候，往往很大一部分时间会去纠结手动释放内存相关的问题。虽然对于小型的程序来说，手动管理内存不是什么难题，但一旦开发的项目日益庞大，代码量达到一定的规模，我们便会开始越来越多地纠缠于内存管理的问题，而不能把全部精力用于解决核心开发目标。因为不合适的图像存储数据结构而疲于维护日益庞大的项目，就有些舍本逐末的感觉了。
自踏入2.0版本的时代以来，OpenCV采用了Mat类作为数据结构进行图像存取。这一改进使OpenCV变得和几乎零门槛入门的Matlab一样，很容易上手和用于实际开发。新版OpenCV中甚至有些函数名称都和 Matlab中的一样，比如大家所熟知的 imread、imwrite、imshow等函数。这对于广大图像处理和计算机视觉领域的研究者们来说，的确是一件可喜可贺的事情。而这一小节中，我们主要来详细讲解SOpenCV2、OpenCV3入门最基本的问题，即图像的载入、显示和输出。
#### 基本鼠标操作
[![DdQOLq.png](https://s3.ax1x.com/2020/11/25/DdQOLq.png)](https://imgchr.com/i/DdQOLq)

># 章节小结
通过123章对OpenCV的认识学习，安装与配置，对OpenCV有了初步的学习和体验，总体来说是非常的神奇有趣。只是实验中有些问题还是需要进行搜索攻克。
