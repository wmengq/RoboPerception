##安装Opencv
下载Opencv
![avatar](\picture\1.png)
解压
![avatar](\picture\2.png)
配置环境
![avatar](\picture\3.png)
##部署OpenCv
 添加包含目录
![avatar](\picture\4.png)
添加库目录
![avatar](\picture\5.png)
添加附加依赖项
![avatar](\picture\6.png)
####安装完成
#第1章 邂逅OpenCV
##1.1OpenCV周边概念认知
###1.1.1图像处理、计算机视觉与OpenCv
图像处理(Image Processing)是用计算机对图像进行分析，以达到所需结果的技术，又称影像处理。图像处理技术一般包括图像压缩，增强和复原，匹配、描述和识别3个部分。图像处理一般指数字图像处理(Digital Image Processing)。其中，数字图像是指用工业相机、摄像机、扫描仪等设备经过拍摄得到的一个大的二维数组。该数组的元素称为像素，其值称为灰度值。而数字图像处理是通过计算机对图像进行去除噪声、增强、复原、分割、提取特征等处理的方法和技术。
计算机视觉（Computer Vision）是一门研究如何使机器“看”的科学，具体地说，就是是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图形处理，用电脑处理使之成为更适合人眼观察或传送给仪器检测的图像的一门学科。作为一门科学学科，计算机视觉研究相关的理论和技术，试图建立能够从图像或者多维数据中获取“信息”的人工智能系统。因为感知可以看做是从感官信号中提取信息，所以计算机视觉也可以看做是研究如何使人工系统从图像或多维数据中“感知”的科学。
##1.1.2 OpenCV概述
OpenCV的全称是Open Source Computer Vision Library，直译就是“开源计算机视觉库”。取代表开源的单词“Open”、"Computer”的首字母“C”以及“Vision”的首字母“V”，组合命名为“OpenCV”。OpenCV 于 1999年由Intel建立，如今由 Willow Garage提供支持。它是一个基于开源发行的跨平台计算机视觉库，可以运行在 Linux、Windows、Mac OS、Android、iOS、Maemo、FreeBSD、OpenBSD 等操作系统上。OpenCV由一系列C函数和C++类构成，轻量且高效。强大的OpenCV除了用C/C++语言进行开发和使用之外，还支持使用C#、Ch、Ruby等编程语言，同时提供了对Python、Ruby、MATLAB等语言的接口，实现了图像处理和计算机视觉方面的很多通用算法。
##1.1.3起源及发展
OpenCV项目最早由Intel公司于1999年启动，旨在促进CPU密集型应用。为了达到这一目的，Intel启动了多个项目，包括实时光线追踪和三维显示墙。在 Intel的性能库团队的帮助下，OpenCV 实现了一些核心代码和算法,并发给Intel俄罗斯的库团队。因此，OpenCV的诞生，是在与软件性能库团队的合作下，发源于Intel的研究中心，并在俄罗斯得到实现和优化。在开始时，OpenCV有以下三大目标，这三大目标也说明了OpenCV的初衷。为基本的视觉应用提供开放且优化的源代码，以促进视觉研究的发展，从而有效地避免“闭门造车”。通过提供一个通用的架构来传播视觉知识，开发者可以在这个架构上继续开展工作，所以代码应该是非常易读且可改写的。OpenCV库采用的协议不要求商业产品继续开放代码，这使得可移植的、性能被优化的代码可以自由获取，可以促进基于视觉的商业应用的发展。而如今看来，OpenCV出现的目的是提供一个可普遍适用的计算机视觉库。OpenCV的第一个预览版本于2000年在IEEE Conference on Computer Vision andPattern Recognition公开，并且后续提供了5个测试版本。1.0版本于2006年发布。OpenCV的第二个主要版本是2009年10月的OpenCV 2.0。该版本的主要更新包括C++接口，更容易、更安全的模式,新的函数，以及对现有实现代码的优化（特别是多核心方面)。现在，约定每6个月就会有一个官方OpenCV版本发布性能被优化的代码可以自由获取，可以促进基于视觉的商业应用的发展。而如今看来，OpenCV出现的目的是提供一个可普遍适用的计算机视觉库。OpenCV的第一个预览版本于2000年在IEEE Conference on Computer Vision andPattern Recognition公开，并且后续提供了5个测试版本。1.0版本于2006年发布。OpenCV的第二个主要版本是2009年10月的OpenCV 2.0。该版本的主要更新包括C++接口，更容易、更安全的模式,新的函数，以及对现有实现代码的优化（特别是多核心方面)。现在，约定每6个月就会有一个官方OpenCV版本发布并且是由某商业公司赞助的独立小组进行开发。在2012年8月，对 OpenCV的支持由一个非营利性组织(OpenCV.org）来提供，并保留了一个开发者网站和用户网站。值得一提的是，现在它也集成了对CUDA的支持。
2014年8月21日，随着OpenCV3时代的第一个版本OpenCV3.0 Alpha在官网首页现身并提供下载，这个强大的计算机视觉库正式迎来了全新的纪元。

##1.1.4应用概述
自从 OpenCV在1999年1月发布 beta版本开始，它就被广泛应用于许多领域、产品和研究成果，具体包括卫星地图和电子地图的拼接、扫描图像对齐、医学图像去噪（消噪或滤波)、图像中的物体分析、安全和入侵检测系统、自动监视和安全系统，以及制造业中的产品质量检测系统、摄像机标定、军事应用、无人飞行器、无人汽车和无人水下机器人。此外，还可以将视觉识别技术用在声谱图上，用OpenCV进行声音和音乐识别。
OpenCV提供的视觉处理算法非常丰富。由于它部分以高效的C语言编写，加上其开源的特性，处理得当，不需要添加新的外部支持也可以完整地编译链接生成执行程序，所以很多研究者用它来做算法的移植。OpenCV的代码经过适当改写可以正常地运行在 DSP系统和单片机系统中，这种移植在高等院校中经常作为相关专业本科生的毕业设计或者研究生、博士生的课题选题。
OpenCV可用于解决如下领域的问题:
·人机交互
·物体识别图像分区
·人脸识别动作识别
·运动
跟踪机器人
##1.2OpenCV基本架构分析
初学OpenCV时，先了解一下 OpenCV的整体模块架构，再重点学习和突破自己感兴趣的部分，就会有得心应手，一览众山小的学习体验。基于这个理念，笔者决定将此节的内容放在本书的首章，让大家高屋建瓴，在学习之初就能把握住新版OpenCV的脉络。初学OpenCV时，先了解一下 OpenCV的整体模块架构，再重点学习和突破自己感兴趣的部分，就会有得心应手，一览众山小的学习体验。基于这个理念，笔者决定将此节的内容放在本书的首章，让大家高屋建瓴，在学习之初就能把握住新版OpenCV的脉络。在此需要提示一下，若想和笔者一起在计算机中进入到OpenCV的文件夹目录，可以先跳到1.3节，先学习如何进行OpenCV的安装和配置，然后再反过来看本节的内容。至于 OpenCV组件结构的研究方法，我们不妨管中窥豹，通过OpenCV安装路径下include目录里面头文件的分类存放，来一窥OpenCV这些年迅猛发展起来的庞杂组件架构。进入到...lopencvlbuildNinclude目录，可以看到有opencv和 openev2这两个文件夹。显然，opencv这个文件夹里面包含着旧版的头文件，而 opencv2这个文件夹里面包含着具有时代意义的新版OpenCV2系列的头文件。
在opencv这个文件夹里面，也就是...lopencvbuildlincludelopencv目录下，可以看到如下各种头文件。这里面大概就是OpenCV 1.0最核心的，而且保留下来的内容的头文件，如图1.2所示。我们可以把它们整体理解为一个大的组件。
(1)【calib3d】-——Calibration（校准）和 3D这两个词的组合缩写。这个模块主要是相机校准和三维重建相关的内容，包括基本的多视角几何算法、单个立体摄像头标定、物体姿态估计、立体相似性算法、3D信息的重建等。
(2)【contrib 】——Contributed/Experimental Stuf的缩写。该模块包含了一些最近添加的不太稳定的可选功能，不用去多管。新增了新型人脸识别、立体匹配、人工视网膜模型等技术。
( 3 )【core】——核心功能模块，包含如下内容:· OpenCV基本数据结构
·动态数据结构
·绘图函数
数组操作相关函数
辅助功能与系统函数和宏●与 OpenGL的互操作
(4)【imgproc】——Image和 Process这两个单词的缩写组合
图像处理模块。
包含如下内容:
·线性和非线性的图像滤波·图像的几何变换
·其他（Miscellaneous）图像转换
直方图相关
结构分析和形状描述·运动分析和对象跟踪·特征检测
·目标检测等内容
(5)【features2d】——也就是Features2D，即 2D功能框架，包含如下内容:·特征检测和描述
●特征检测器（Feature Detectors）通用接口
·描述符提取器（Descriptor Extractors）通用接口描述符匹配器（Descriptor Matchers）通用接口通用描述符(Generic Descriptor）匹配器通用接口·关键点绘制函数和匹配功能绘制函数
(6)【flann】——Fast Library for Approximate Nearest Neighbors，高维的近似近邻快速搜索算法库，包含以下两个部分:
·快速近似最近邻搜索
·聚类
(7)【gpu】——运用GPU加速的计算机视觉模块。
( 8)【highgui】——高层GUI图形用户界面，包含媒体的输入输出、视频捕捉、图像和视频的编码解码、图形交互界面的接口等内容。
(9)【legacy 】——一些已经废弃的代码库，保留下来作为向下兼容，包含如下内容:
运动分析
期望最大化
直方图
·平面细分（CAPI)
·特征检测和描述（Feature Detection and Description)
描述符提取器（Descriptor Extractors）的通用接口通用描述符（Generic Descriptor Matchers）的常用接口匹配器
(10)【ml】——Machine Learning，机器学习模块，基本上是统计模型和分类算法，包含如下内容:
·统计模型( Statistical Models）
·一般贝叶斯分类器（Normal Bayes Classifier)· K-近邻（K-Nearest Neighbors)
·支持向量机（Support Vector Machines)·决策树(Decision Trees)
提升（Boosting）
·梯度提高树(Gradient Boosted Trees)随机树(Random Trees)
超随机树(Extremely randomized trees)期望最大化（Expectation Maximization)神经网络（Neural Networks)
MLData
(11)【nonfree】——一些具有专利的算法模块，包含特征检测和GPU相关的内容。最好不要商用。
(12)【objdetect】——目标检测模块，包含Cascade Classification（级联分类)和 Latent SVM这两个部分。
(13)【ocl】—OpenCL-accelerated Computer Vision，运用OpenCL加速的计算机视觉组件模块。
(14)【photo】——Computational Photography，包含图像修复和图像去噪两部分
(15)【stitching】——images stitching，图像拼接模块，包含如下部分:
拼接流水线
特点寻找和匹配图像
·估计旋转
·自动校准
·图片歪斜
·接缝估测
●曝光补偿
图片混合
(16)【superres】 SuperResolution，超分辨率技术的相关功能模块。(17)【ts】—OpenCV测试相关代码，不用去管。
(18)【video】——视频分析组件，该模块包括运动估计、背景分离、对象跟踪等视频处理相关内容。
(19)【Videostab】——Video stabilization，视频稳定相关的组件，官方文档中没有多做介绍，不用管它。
##1.3OpenCV3带来了什么
2009年10月01日，OpenCV 2.0发布，这标志着革命性的 OpenCV2时代的来临。OpenCV2带来了全新的C++接口，将 OpenCV的能力无限放
人和oPabot
时代,OpenCV增加了新的平台支持,包括iOS和 Andriod,通过CUDA
实现了GPU加速，为 Python和 Java用户提供了接口，基于Github 和 Buildbot构建了充满艺术感的持续集成系统，所以才有了被全世界的很多公司和学校所采用的稳定易用的 OpenCV 2.4.x。
2014年8月21日，OpenCV 3.0 Alpha发布，宣告着OpenCV3时代的登场。官方更新日志中提到，OpenCV在3.0中改变了项目架构的方式，所以3.0时代不会有像2.0时代一样激进的尝试，只会有足够稳定的改进。且不说更多眼花缭乱的新特性，项目架构的改变是OpenCV3最为重大的革新之处。让我们在1.3.1节中一起进行了解。
###1.3.1项目架构的改变
通过1.2节《OpenCV基本架构分析》的介绍，我们可以发现 OpenCV是一个相对于整体的项目，各个模块都是以整体的形式构建然后组合在一起的。然而，随着功能的增加，OpenCV主体集成了各种各样的功能模块，
变得越来越臃
肿。
而3.0的出现，就是为了给日益发福的OpenCV减肥，因为OpenCV3决定像其他大项目一样，抛弃整体架构，使用内核+插件的架构形式。
在 GitHub 中，除了存放着正式版OpenCV 的主仓库和新增加的"opencv_extra”仓库以外，OpenCV 3.0中还添加了一个名为opencv_contrib的全
新仓库，这个新仓库中有R多充满艺术感的图像修复、深度地图处理、新的及文本识别、新的边缘检测器、充满艺术感的图像修复、深度地图处理、新的
光流和追踪算法等。
#第一个程序:图像显示

include <opencv2/ opencv .hpp>
using namespace cv;
int main()
{
// 【1】读入一张图片
Mat img=imread("1.jpg"');l 
【2】在窗口中显示载入的图片
imshow("【载入的图片】 " , img ) ;
// 【3】等待6000 ms后窗口自动关闭
waitKey (6000) ;
}
![avatar](\picture\7.png)
![avatar](\picture\8.png)
include <opencv2/opencv.hpp>
using namespace cv;//包含cv命名空间
void main ()
{
Mat srcImage = imread( "1 .jpg"); 
//载入图像
imshow("【原始图】", srcImage);
//显示图像
waitKey(0);
//等待任意按键按下
}
![avatar](\picture\9.png)
#第二个程序:图像腐蚀

include <opencv2/highgui/highgui.hpp>
// opencv highgui模块头文件
include <opencv2/ imgproc/imgproc.hpp>
/ / 0pencV图像处理头文件
using namespace cv;
//包含cv命名空间
int main()
//控制台应用程序的入口函数，我们的程序从这里开始
//载入原图
Mat srcImage = imread ("1.jpg" ) ;
//显示原图
imshow ("【原图】腐蚀操作", srcImage ) ;
//进行腐蚀操作
Mat element = getStructuringElement (MORPH_RECT，Size(15，15));
Mat dstImage;
erode (srcImage, dstImage, element) ;
//显示效果图
imshow("【效果图】腐蚀操作",dstImage);
waitKey (0) ;
return 0;
![avatar](\picture\10.png)
#第三个程序:图像模糊
include "opencv2/highgui/highgui.hpp"
include "opencv2 /imgproc /imgproc.hpp"using namespace cv;
// 
//描述:控制台应用程序的入口函数，我们的程序从这里开始// 
int main ()
//【1】载入原始图
Mat srcImage=imread ("1.jpg");
//[2】显示原始图
imshow( "均值滤波【原图】", srcImage );
//【3】进行均值滤波操作
Mat dstImage;
blur ( srcImage, dstImage, size (7,7) );
//[4】显示效果图
imshow("均值滤波【效果图】" , dstImage );
waitKey(0 );
)
![avatar](\picture\11.png)
##第四个程序: canny边缘检测
include <opencv2 / opencv.hpp>
include<opencv2 /imgproc/imgproc.hpp>
using namespace cv;
int main ()
//【0】载入原始图
Mat srcImage = imread ("l.jpg");
//工程目录下应该有一张名为1.jpg的素材
imshow("【原始图Canny边缘检测",srcImage);
/显示原
Mat dst Image, edge , gray Image;1/参数定义
【1】创建与src同类型和大小的矩阵(dst)
dstImage.create( srcImage.size(),srcImage.type());
//【2】将原图像转换为灰度图像
//此句代码的 OpenCV2版为:
// cvtcolor( srcImage,grayImage,CV_BGR2GRAY );
//此句代码的OpencV3版为:
cvtColor( srcImage,grayImage,COLOR_BGR2GRAY );
//【3】先使用3x3内核来降噪
blur( grayImage, edge,Size(3, 3));
//【4】运行canny算子
Canny( edge,edge,3，9,3 );
//【5】显示效果图
imshow("【效果图】Canny边缘检测", edge) ;
waitKey(0);
return 0;
}
![avatar](\picture\12.png)
![avatar](\picture\13.png)
#读取并播放视频
#include <opencv2 opencv. hpp>
using namespace cv;
int main ( )
//【 1】读入视频
videoCapture capture ( "1.avi" );
//[2】循环显示每一帧
while(1)
Mat frame;
//定义一个Mat变量，用于存储每一帧的图像capture>>frame;
//读取当前帧
imshow("读取视频",frame) ;
//显示当前帧
waitKey (30);
//延时30ms
}
return0;
}
![avatar](\picture\14.png)
##调用摄像头采集图像
#include <opencv2/ opencv.hpp>
using namespace cv;
int main ()
/ /【1】从摄像头读入视频
VideoCapture capture (0) ;
//【2】循环显示每一帧
while(1)
Mat frame;
//定义一个Mat变量，用于存储每一帧的图
capture>>frame;
//读取当前帧
imshow("读取视频"，frame) ;
//显示当前帧
waitKey(30);
//延时30ms
)
return0;
}
![avatar](\picture\15.png)
#include "opencv2/ opencv. hpp"
using namespace cv ;
int main ()
//从摄像头读入视频
VideoCapture capture (0);
Mat edges;
//循环显示每一帧while (1)
{
//【1】读入图像
Mat frame ;
//定义一个 Mat变量，用于存储每一帧的图像
capture >> frame ; //读取当前帧
//【2】将原图像转换为灰度图像
cvtcolor (frame，edges,cv_BGR2GRAY); //转化BGR彩色图为灰度图
//[ 3】使用3x3内核来降噪（2x3+1=7)
blur(edges,edges,Size ( 7,7));//进行模糊
//[4】进行canny边缘检测并显示
Canny (edges, edges,0，30，3);
imshow("被canny后的视频", edges);
//显示经过处理后的当前帧
if (waitKey ( 30) >= 0) 
break; /
/延时30ms
return 0 ;
}
![avatar](\picture\16.png)