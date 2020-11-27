# OpenCV编程入门

## 一、OpenCV基础

### 1.1 图像处理、计算机视觉与OpenCV

- 图像处理 
  
  用计算机对图像进行分析，以达到所需结果的技术，又称影像处理。图像处理技术一般包括图像压缩，增强和复原，匹配、描述和识别3个部分。图像处理一般指数字图像处理(Digital Image Processing)。其中，数字图像是指用工业相机、摄像机、扫描仪等设备经过拍摄得到的一个大的二维数组。该数组的元素称为像素，其值称为灰度值。而数字图像处理是通过计算机对图像进行去除噪声、增强、复原、分割、提取特征等处理的方法和技术。

- 计算机视觉
  
  一门研究如何使机器“看”的科学，具体地说，就是是指用摄影机和电脑代替人眼对目标进行识别、跟踪和测量等机器视觉，并进一步做图形处理，用电脑处理使之成为更适合人眼观察或传送给仪器检测的图像的一门学科。作为一门科学学科，计算机视觉研究相关的理论和技术，试图建立能够从图像或者多维数据中获取“信息”的人工智能系统。因为感知可以看做是从感官信号中提取信息，所以计算机视觉也可以看做是研究如何使人工系统从图像或多维数据中“感知”的科学。

- 图像处理和计算机视觉区别
  
  图像处理侧重于“处理”图像——如增强，还原，去噪，分割，等等;而计算机视觉重点在于使用计算机（也许是可移动式的）来模拟人的视觉，因此模拟才是计算机视觉领域的最终目标。

### 1.2 OpenCV

#### 概述
  一个基于开源发行的跨平台计算机视觉库，它实现了图像处理和计算机视觉方面的很多通用算法，已经成为了计算机视觉领域最有力的研究工具之一。

#### 设计
  执行速度尽量快，主要关注实时应用。它采用优化的C/C++代码编写，能够充分利用多核处理器的优势，其主要目标是构建一个简单易用的计算机视觉框架，以帮助开发人员更便捷地设计更复杂的计算机视觉相关应用程序。

#### 应用
  OpenCV 覆盖了计算机视觉的许多应用领域，如工厂产品检测、医学成像、信息安全、用户界面、摄像机标定、立体视觉和机器人等。因为计算机视觉和机器学习密切相关，所以OpenCV还提供 MLL(Machine Learning Library〉机器学习库。该机器学习库主要用于统计方面的模式识别和聚类（clustering)。MLL除了用在视觉相关的任务中，还可以方便地应用于其他机器学习场合。
  
  OpenCV 可用于解决如下领域的问题:
  - 人机交互
  - 物体识别
  - 图像分区
  - 人脸识别
  - 动作识别
  - 运动跟踪
  - 机器人
  
#### 目标
  - 为基本的视觉应用提供开放且优化的源代码，以促进视觉研究的发展，从而有效地避免“闭门造车”。
  - 通过提供一个通用的架构来传播视觉知识，开发者可以在这个架构上继续开展工作，所以代码应该是非常易读且可改写的。
  - OpenCV 库采用的协议不要求商业产品继续开放代码，这使得可移植的、性能被优化的代码可以自由获取，可以促进基于视觉的商业应用的发展。

#### 模块
  - 【calib3d】————Calibration（校准）和3D这两个词的组合缩写。这个模块主要是相机校准和三维重建相关的内容，包括基本的多视角几何算法、单个立体摄像头标定、物体姿态估计、立体相似性算法、3D信息的重建等。
  - 【contrib】————Contributed/Experimental Stuf的缩写。该模块包含了一些最近添加的不太稳定的可选功能，不用去多管。新增了新型人脸识别、立体匹配、人工视网膜模型等技术。
  - 【core】————核心功能模块。
  - 【imgproc】————Image和 Process这两个单词的缩写组合，图像处理模块。
  - 【features2d】————也就是Features2D，即2D功能框架。
  - 【flann】————Fast Library for Approximate Nearest Neighbors，高维的近似近邻快速搜索算法库。
  - 【gpu】————运用GPU加速的计算机视觉模块。
  - 【highgui】————高层GUI 图形用户界面，包含媒体的输入输出、视频捕捉、图像和视频的编码解码、图形交互界面的接口等内容。
  - 【legacy 】————一些已经废弃的代码库，保留下来作为向下兼容。
  - 【ml】————Machine Learning，机器学习模块，基本上是统计模型和分类算法。
  - 【nonfree】————些具有专利的算法模块，包含特征检测和GPU相关的内容。最好不要商用。
  - 【objdetect】————目标检测模块，包含Cascade Classification(级联分类)和 Latent SVM这两个部分。
  - 【ocl】————OpenCL-accelerated Computer Vision，运用OpenCL加速的计算机视觉组件模块。
  - 【photo】————Computational Photography，包含图像修复和图像去噪两部分。
  - 【stitching】————images stitching，图像拼接模块。
  - 【superres】————SuperResolution，超分辨率技术的相关功能模块。
  - 【ts】————OpenCV测试相关代码，不用去管。
  - 【video】————视频分析组件，该模块包括运动估计、背景分离、对象跟踪等视频处理相关内容。
  - 【Videostab】————Video stabilization，视频稳定相关的组件，官方文档中没有多做介绍，不用管它。

### 1.3 OpenCV下载、安装与配置

  百度按照csdn以及编程入门上的步骤一步步来就能完成按照和环境配置。

#### 下载解压好opencv4.5
![avatar](\picture\1.png)

#### 在此电脑中添加环境变量
![avatar](\picture\2.png)

#### 打开vs2019
![avatar](\picture\3.png)

#### 新建C++项目配置opencv
![avatar](\picture\4.png)

#### 右键目录，在设置中配置VC++目录
![avatar](\picture\5.png)

要配置 包含目录 和 库目录
![avatar](\picture\6.png)

#### 配置连接器中的输入
![avatar](\picture\7.png)

配置完后点击确定退出。
之后在资源文件中新建一个 C++ 文档即可开始跑代码。

### 1.4 OpenCV图像处理
根据编程入门依次完成相关程序

#### 1）图像显示
代码：
```C++
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;

int main()
{
    Mat image = imread("E:\\opencv4.5\\opencv\\opencv\\fack.jpg");  //存放自己图像的路径 
    imshow("显示图像", image);
    waitKey(0);
    return 0;
}
```

实验结果：
![avatar](\picture\1.图像显示结果.png)

#### 2）图像腐蚀
代码：
```C++
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

int main()
{
	//载入原图
	Mat srcImage = imread("fack.jpg");
	//显示原图
	imshow("【原图】腐蚀操作", srcImage);
	//进行腐蚀操作
	Mat element = getStructuringElement(MORPH_RECT, Size(15, 15));
	Mat dstImage;
	erode(srcImage, dstImage, element);
	//显示效果图
	imshow("【效果图】腐蚀操作", dstImage);
	waitKey(0);

	return 0;
}
```

实验结果：
![avatar](\picture\2.图像腐蚀结果.png)

#### 3）图像模糊
代码：
```C++
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

int main()
{
	// [1]载入原始图
	Mat srcImage = imread("fack.jpg");

	// [2]显示原始图
	imshow("均值滤波[原图] ", srcImage);

	// [3]进行均值滤波操作
	Mat dstImage;
	blur( srcImage, dstImage, Size(7, 7));

	// [4]显示效果图
	imshow( "均值滤波[效果图]",dstImage );

	waitKey(0);
}
```

实验结果：
![avatar](\picture\3.图像模糊结果.png)

#### 4）canny边缘检测
代码：
```C++
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

int main()
{
	//【0】载入原始图
	Mat srcImage = imread("fack.jpg"); 

	//【1】创建与src同类型和大小的矩阵（dst）
	imshow("【原始图】Canny边缘检测", srcImage); //显示原图
	Mat edge, grayImage;//参数定义

	//【2】将原图转为灰度图像
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);

	//【3】先试用3*3内核降噪
	blur(grayImage, edge, Size(3, 3));

	//【4】运行canny算子
	Canny(edge, edge, 3, 9, 3);

	//【5】显示效果图
	imshow("【效果图】Canny边缘检测", edge);

	waitKey(0); //等待按键按下

	return 0;
```

实验结果：
![avatar](\picture\4.边缘检测结果.png)

### 1.5 OpenCV视频操作
学习利用 OpenCV 中的 VudeoCapture 类，来对视频进行读取显示以及调用摄像头。

#### 1）读取并播放视频
代码：
```C++
#include <opencv2\opencv.hpp>
using namespace cv;

int main()
{
	//[1]读入视频
	VideoCapture capture("qq.avi");

	//[2]循环显示每一帧
	while (1)
	{
		Mat frame;  // 定义一个Mat变量，用于存储每一帧的图像
		capture >> frame; // 读取当前帧
		imshow("读取视频", frame); //显示当前桢
		waitKey(30); //延时30ms
	}
	return 0;
}
```

实验结果：
![avatar](\picture\5.视频播放结果.png)

#### 2）调用摄像头采集图像
代码：
```C++
#include <opencv2\opencv.hpp>
using namespace cv;

int main()
{
	//[1]从摄像头读入视频
	VideoCapture capture(0);

	//[2]循环显示每一帧
	while (1)
	{
		Mat frame;  // 定义一个Mat变量，用于存储每一帧的图像
		capture >> frame; // 读取当前帧
		imshow("读取视频", frame); //显示当前桢
		waitKey(30); //延时30ms
	}
	return 0;
}
```

实验结果：
![avatar](\picture\6.摄像头视频结果.png)

#### 3）摄像头调用配合canny边缘检测
代码：
```C++
#include <opencv2\opencv.hpp>
using namespace cv;

int main()
{
	//从摄像头读入视频
	VideoCapture capture(0);
	Mat edges;

	//循环显示每一帧
	while (1)
	{
		//【1】读入图像
		Mat frame;  // 定义一个Mat变量，用于存储每一帧的图像
		capture >> frame; // 读取当前帧

		//【2】将原图像转换为灰度图像
		cvtColor(frame, edges, COLOR_BGR2GRAY);//opencv2用的是CV_BGR2GRAY

		//【3】使用 3x3内核来降噪（2x3 + 1 = 7 )
		blur(edges, edges, Size(7, 7)); //进行模糊

		//【4】进行canny边缘检测并显示
		Canny(edges, edges, 0, 30, 3);
		imshow("被canny后的视频", edges); //显示经过处理后的当前帧
		if (waitKey(30) >= 0) break; //延时30ms
	}
	return 0;
}
```

实验结果：
![avatar](\picture\7.摄像头边缘结果.png)


## 二、OpenCV认知准备

### 2.1 官方例程
- 彩色目标跟踪：Camshift
- 光流：optical flow
- 点追踪：lkdemo
- 人脸识别：objectDetection
- 支持向量机引导

### 2.2 CMake
  一个跨平台的安装（编译）工具，可以用简单的语句来描述所有平台的安装(编译过程)。他能够输出各种各样的 makefile 或者 project 文件，能测试编译器所支持的 C++ 特性,类似 UNIX 下的 automake。只是 CMake 的组态档取名为 CMakeLists.txt。

  Cmake 并不直接建构出最终的软件，而是产生标准的建构档（如 Unix 的 Makefile 或 Windows Visual C++ 的 projects/workspaces），然后再依一般的建构方式使用。这使得熟悉某个集成开发环境（IDE）的开发者可以用标准的方式建构他的软件，这种可以使用各平台的原生建构系统的能力是 CMake 和 SCons 等其他类似系统的区别之处。

### 2.3 “opencv.hpp”头文件认知
opencv.hpp 中已经包含了OpenCV各模块的头文件，如高层GUI图形用户界面模块头文件 “highgui.hpp”、图像处理模块头文件 “imgproc.hpp”、2D特征模块头文件 “features2d.hpp”等。

在编写 core、objdetect、imgproc、photo、video、features2d、objdetect、calib3d、ml、highgui、contrib模块的应用程序时，原则上仅写上一句 “#include<opencv2/opencv.hpp>” 即可，这样可以精简优化代码。

### 2.4 argc和argv
- int argc 表示命令行字串的个数
- char *argv[] 表示命令行参数的字符串

  使用的开发环境为 Visual Studio，往后若在 OpenCV 相关代码中遇到这两个参数，可以在代码中将其用路径字符串替换，或者在项目属性页中给其赋值,而对程序整体影响不大的部分就将其注释掉。

### 2.5 printf()简析
printf 函数是我们经常会用到的格式输出函数，其关键字最末一个字母f即为“格式” (format）之意。其功能是按用户指定的格式，把指定的数据显示到窗口中。

printf 函数的一个比较有特殊的用法是“格式字符串”，其用于指定输出格式，可由格式字符串和非格式字符串两种组成。格式字符串是以%开头的字符串，在%后面跟有各种格式字符，以说明输出数据的类型、形式、长度、小数位数等。

## 三、HighGUI图形用户界面初步
### 3.1图像的载入、显示和输出
#### 图像的载入：imread()函数
用于读取文件中的图片到 OpenCV 中。可以在 OpenCV 官方文档中查到它的原型，如下：
Mat imread (const string & filename,intflags=1 )

- 第一个参数，const string& 类型的 filename，填我们需要载入的图片路径名。在 Windows 操作系统下，OpenCV的imread函数支持如下类型的图像载入。
- 第二个参数，int 类型的 flags，为载入标识，它指定一个加载图像的颜色类型。可以看到它自带默认值1，所以有时候这个参数在调用时可以忽略。如果在调用时忽略这个参数，就表示载入三通道的彩色图像。
- flags 是 int 型的变量，若我们不在这个枚举体中取固定的值，可以这样进行:
  - flags > 0 返回一个3通道的彩色图像;
  - flags = 0 返回灰度图像;
  - flags < 0 返回包含Alpha通道的加载图像。

#### 图像的显示：imshow()函数
imshow() 函数用于在指定的窗口中显示一幅图像，函数原型如下：
void imshow (const string& winname,InputArray mat);

- 第一个参数: const string& 类型的 winname，填需要显示的窗口标识名称。
- 第二个参数:InputArray 类型的 mat，填需要显示的图像。

#### 创建窗口：namedWindow()函数
namedWindow 函数用于创建一个窗口。若是简单地进行图片显示，可以略去 namedWindow 函数的调用，即先调用 imread 读入图片，然后用 imshow 直接指定出窗口名进行显示即可。但需要在显示窗口之前就用到窗口名时，要指定滑动条依附到某个窗口上，就需要 namedWindow 函数先创建出窗口，显式地规定窗口名称了。
namedWindow的函数原型如下:
void namedwindow(const string& winname,int flags=wINDOw_AUTOSIZE );

- 第一个参数，const string& 型的 name，填写被用作窗口的标识符的窗口名称。
- 第二个参数，int 类型的 flags ，窗口的标识。

#### 输出图像到文件：imwrite()函数
在 OpenCV 中，输出图像到文件一般采用 imwrite 函数，它的声明如下：
bool imwrite (const string& filename,InputArray img，const vector<int>& params=vector<int>();

- 第一个参数，const string& 类型的 filename，填需要写入的文件名。
- 第二个参数，InputArray 类型的 img，一般填一个 Mat 类型的图像数据。
- 第三个参数，const vector<int>& 类型的 params，表示为特定格式保存的参数编码。它有默认值 vector<int>()，所以一般情况下不需要填写。

### 程序：OpenCV生成一幅png图片
代码:
```C++
#include <opencv2\opencv.hpp>
#include <vector>

using namespace cv;
using namespace std;

void createAlphaMat(Mat& mat)
{
	for (int i = 0; i < mat.rows; ++i) {
		for (int j = 0; j < mat.cols; ++j) {
			Vec4b& rgba = mat.at<Vec4b>(i, j);
			rgba[0] = UCHAR_MAX;
			rgba[1] = saturate_cast<uchar>((float(mat.cols - j)) / ((float)mat.cols) * UCHAR_MAX);
			rgba[2] = saturate_cast<uchar>((float(mat.rows - i)) / ((float)mat.rows) * UCHAR_MAX);
			rgba[3] = saturate_cast<uchar>(0.5 * (rgba[1] + rgba[2]));
		}
	}
}

int main()
{
	//创建带Alpha通道的Mat
	Mat mat(480, 640, CV_8UC4);
	createAlphaMat(mat);

	vector<int>compression_params;

	// 此句代码的OpenCV2版为:
	// compression_params.push_back(CV_IMWRITE_PNG_COMPRESSION);
	//此句代码的openCV3版为:
	compression_params.push_back(IMWRITE_PNG_COMPRESSION);
	compression_params.push_back(9);

	try {
		imwrite("dota1.png", mat, compression_params);
		imshow("生成的PNG图", mat);
		fprintf(stdout, "PNG困片丈件的alpha数据保存完半~\n可以在工程目录查看由imwrite函数生成的图片\n");
		waitKey(0);
	}
	catch (runtime_error& ex) {
		fprintf(stderr, "图像转换成PNG格式发生錯误: %s\n", ex.what());
		return 1;
	}

	return 0;
}
```

实验结果：
![avatar](\picture\8.imwrite结果.png)

### 程序：图像的载入、显示和输出
代码:
```C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

int main()
{
	Mat girl = imread("girl.jpg"); //载入 图像到Mat
	namedWindow("[1]动漫图"); //创建一个名为 ”[1]动漫图"的窗口
	imshow("[1]动漫图", girl);//显示名为"[1]动漫图"的窗口

	//载入图片
	Mat image = imread("dota5.jpg");
	Mat logo = imread("dota_logo.jpg");
	//载入后先显示
	namedWindow(" [2] 原画图"); imshow(" [2] 原画图", image);
	namedWindow(" [3] logo 图"); imshow("[3] logo 图", logo);

	//定义一个Mat类型，用于存放，图像的ROI
	Mat imageROI;
	//方法一
	imageROI = image(Rect(800, 350, logo.cols, logo.rows));
	//方法二
	// imageROI =
	image(Range(350, 350 + logo.rows), Range(800, 800 + logo.cols));

	// 将logo加到原图上
	addWeighted(imageROI, 0.5, logo, 0.3, 0., imageROI);

	//显示结果
	namedWindow(" [4] 原画+logo 图"); imshow("[4] 原画+logo 图", image);

	//输出一张jpg图片到工程目录下
	imwrite("由imwrite生成的图片.jpg", image);

	waitKey();

	return 0;
}
```

实验结果：
![avatar](\picture\9.图像混合结果.png)


### 3.2 滑动条的创建和使用

#### 创建滑动条：createTrackbar()函数
createTrackbar() 函数用于创建一个可以调整数值的滑动条（常常也被称作轨迹条)，并将滑动条附加到指定的窗口上，使用起来很方便。需要记住，它往往会和一个回调函数配合起来使用。先看下它的函数原型，如下：
C++: int createTrackbar(conststring& trackbarname，conststring& winname, int* value,int count,TrackbarCallback onChange=0, void* userdata=0);

- 第一个参数，const string& 类型的 trackbarname，轨迹条的名字，用来代表我们创建的轨迹条。
- 第二个参数，const string& 类型的 winname，窗口的名字，表示这个轨迹条会依附到哪个窗口上，即对应 namedWindow() 创建窗口时填的某一个窗口名。
- 第三个参数，int* 类型的 value，一个指向整型的指针，表示滑块的位置。在创建时，滑块的初始位置就是该变量当前的值。
- 第四个参数，int 类型的 count，表示滑块可以达到的最大位置的值。滑块最小位置的值始终为0。
- 第五个参数，TrackbarCallback 类型的 onChange 它有默认值0。这是一个指向回调函数的指针，每次滑块位置改变时，这个函数都会进行回调。
- 第六个参数，void* 类型的 userdata，也有默认值 0。这个参数是用户传给回调函数的数据，用来处理轨迹条事件。如果使用的第三个参数 value 实参是全局变量的话，完全可以不去管这个 userdata 参数。

### 程序：用轨迹条来控制两幅图像的Alpha混合
代码：
```C++
#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
using namespace cv;

#define _CRT_SECURE_NO_WARNINGS
#define WINDOW_NAME "【混合示例】"

const int g_nMaxAlphaValue = 100; //AIpha 值的最大值
int g_nAlphaValueSlider;//滑动条对应的变量
double g_dAlphaValue;
double g_dBetaValue;

//声明存储图像的变量
Mat g_srcImage1;
Mat g_srcImage2;
Mat g_dstImage;

void on_Trackbar(int, void*)
{
	//求出当前alpha值相对于最大值的比例
	g_dAlphaValue = (double)g_nAlphaValueSlider / g_nMaxAlphaValue;
	//则beta值为1减去alpha值
	g_dBetaValue = (1.0 - g_dAlphaValue);

	//根据alpha和beta值进行线性混合
	addWeighted(g_srcImage1, g_dAlphaValue, g_srcImage2, g_dBetaValue, 0.0, g_dstImage);

	//显示效果图
	imshow(WINDOW_NAME, g_dstImage);
}

int main(int argc, char** argv)
{
	//加载图像(两图像的尺寸需相同)
	g_srcImage1 = imread("1.jpg");
	g_srcImage2 = imread("2.jpg");
	if (!g_srcImage1.data) {
		printf("读取第一幅图片错误，请确定目录下是否有imread函数指定图片存在~! \n");
		return -1;
	}
	if (!g_srcImage2.data) {
		printf("读取第二幅图片错误，请确定目录下是否有imread函数指定图片存在~! \n");
		return -1;
	}

	//设置滑动条初值为70
	g_nAlphaValueSlider = 70;

	//创建窗体
	namedWindow(WINDOW_NAME, 1);

	//在创建的窗体中创建一个滑动条控件
	char TrackbarName[50];
	sprintf_s( TrackbarName, "透明值 &d", g_nMaxAlphaValue);

	createTrackbar(TrackbarName, WINDOW_NAME, &g_nAlphaValueSlider, g_nMaxAlphaValue, on_Trackbar);

	//结果在回调函数中显示
	on_Trackbar(g_nAlphaValueSlider, 0);

	//按任意键退出
	waitKey(0);

	return 0;
}
```

实验结果：
![avatar](\picture\10.滑动条结果.png)


### 鼠标操作
OpenCV中的鼠标操作和滑动条的消息映射方式很类似，都是通过一个中介函数配合一个回调函数来实现的。创建和指定滑动条回调函数的函数为createTrackbar，而指定鼠标操作消息回调函数的函数为SetMouseCallback。

SetMousecallback函数的作用是为指定的窗口设置鼠标回调函数，原型如下：
C++ : void setMouseCallback(conststring& winname,MouseCallback onMouse,void*userdata=0 )

- 第一个参数，const string& 类型的 winname，窗口的名字。
- 第二个参数，MouseCallback 类型的 onMouse，指定窗口里每次鼠标时间发生的时候，被调用的函数指针。

### 程序：鼠标操作
代码：
```C++
#include <opencv2/opencv.hpp>
using namespace cv;
#define WINDOW_NAME "[ 程序窗口 ] " //为窗口标题定义的宏

void on_MouseHandle(int event, int x, int y, int flags, void* param);
void DrawRectangle(cv::Mat& img, cv::Rect box);
void ShowHelpText();

Rect g_rectangle;
bool g_bDrawingBox = false;//是否进行绘制
RNG g_rng(12345);

int main(int argc, char** argv)
{
	// [1]准备参数
	g_rectangle = Rect(-1, -1, 0, 0);
	Mat srcImage(600, 800, CV_8UC3), tempImage;
	srcImage.copyTo(tempImage);
	g_rectangle = Rect(-1, -1, 0, 0);
	srcImage = Scalar::all(0);

	// [2]设置鼠标操作回调函数
	namedWindow(WINDOW_NAME);
	setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&srcImage);

	// [3]程序主循环，当进行绘制的标识符为真时，进行绘制
	while (1)
	{
		srcImage.copyTo(tempImage);//复制源图到临时变量
		if (g_bDrawingBox) DrawRectangle(tempImage, g_rectangle);//当进行绘制的标识符为真，则进行绘制
		imshow(WINDOW_NAME, tempImage);
		if (waitKey(10) == 27) break;//按下 ESC键，程序退出
	}
	return 0;
}

void on_MouseHandle(int event, int x, int y, int flags, void* param)
{
	Mat&image = *(cv::Mat*) param; 
	switch ( event) 
	{
		//鼠标移动消息.
		case EVENT_MOUSEMOVE :
		{
			if (g_bDrawingBox) //如果是否进行绘制的标识符为真，则记录下长和宽到RECT型变量中
			{
				g_rectangle.width = x-g_rectangle.x;
				g_rectangle.height = y - g_rectangle.y;
			}
		}
		break;

		//左键按下消息
		case EVENT_LBUTTONDOWN :
		{
			g_bDrawingBox = true;
			g_rectangle = Rect(x, y, 0, 0);//记录起始点
		}
			break;

		//左键拾起消息
		case EVENT_LBUTTONUP:
		{
			g_bDrawingBox = false;//置标识符为 false
			//对宽和高小于0的处理
			if (g_rectangle.width < 0)
			{
				g_rectangle.x += g_rectangle.width;
				g_rectangle.width *= -1;
			}

			if (g_rectangle.height < 0)
			{
				g_rectangle.y += g_rectangle.height;
				g_rectangle.height *= -1;
			}
			//调用函数进行绘制
			DrawRectangle(image, g_rectangle);
		}
		break;
	}
}

void DrawRectangle(cv::Mat& img, cv::Rect box)
{
	rectangle(img, box.tl(), box.br(), Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));//随机颜色
}
```

实验结果：
![avatar](\picture\11.鼠标操作结果.png)
