# OpenCV编程入门

## 四、OpenCV数据结构与基本绘图

### 4.1 基础图像容器Mat
#### 数字图像储存
通过各种各样的方法从现实世界获取到数字图像，通常由显示屏上看到的都是真实而漂亮的图像，但是这些图像在转化到我们的数字设备中时，记录的却是图像中的每个点的数值。

矩阵就是图像在数码设备中的表现形式。OpenCV 作为一个计算机视觉库，其主要的工作是处理和操作并进-一步 了解这些形式和信息。因此，理解OpenCV是如何存储和处理图像是非常有必要的。那么，就让我们来进一一步了解和学习。

#### Mat结构使用
从OpenCV踏入2.0时代，使用Mat类数据结构作为主打之后，OpenCV变得越发像需要很少编程涵养的Matlab那样，上手很方便。甚至有些函数名称都和Matlab一样，比如大家所熟知的 imread、imwrite、imshow等函数。

关于Mat类，首先我们要知道的是:
1) 不必再手动为其开辟空间
2) 不必再在不需要时立即将空间释放。

这里指的是手动开辟空间并非必须，但它依旧是存在的——大多数OpenCv函数仍会手动地为输出数据开辟空间。当传递一个已经存在的Mat对象时，开辟好的矩阵空间会被重用。也就是说，我们每次都使用大小正好的内存来完成任务。

总而言之，Mat是一个类，由两个数据部分组成:矩阵头（包含矩阵尺寸、存储方法、存储地址等信息）和一个指向存储所有像素值的矩阵（根据所选存储方法的不同，矩阵可以是不同的维数）的指针。矩阵头的尺寸是常数值，但矩阵本身的尺寸会依图像的不同而不同，通常比矩阵头的尺寸大数个数量级。因此，当在程序中传递图像并创建副本时，大的开销是由矩阵造成的，而不是信息头。

OpenCV是一个图像处理库，囊括了大量的图像处理函数，为了解决问题通常要使用库中的多个函数，因此在函数中传递图像是常有的事。同时不要忘了我们正在讨论的是计算量很大的图像处理算法，因此，除非万不得已，不应该进行大图像的复制，因为这会降低程序的运行速度。

为了解决此问题，OpenCV使用了引用计数机制。其思路是让每个Mat对象有自己的信息头，但共享同一个矩阵。这通过让矩阵指针指向同一地址而实现。而拷贝构造函数则只复制信息头和矩阵指针，而不复制矩阵。

#### 显式创建Mat对象的方法

1) 【方法一】使用Mat()构造函数

   最常用的方法是直接使用Mat()构造函数,这种方法简单明了,示范代码如下。
   ```C++
    Mat M(2,2，cv_8UC3,Scalar(0,0,255));
    cout<<"M = " << endl<<" "<<M<<endl << endl;
   ```

2) 【方法二】在C\C++中通过构造函数进行初始化
   
   这种方法为在C\C++中通过构造函数进行初始化，示范代码如下。
   ```C++
   int sz[ 3] = {2,2,2} ;
   Mat L(3,sz,cv_8uc,Scalar : :all(0) ) ;
   ```

3) 【方法三】为已存在的 lpllmage指针创建信息头
   
   方法三是为已存在的Ipllmage指针创建信息头，示范代码如下。
   ```C++
   IplImage* img = cvLoadImage ( "1.jpg"，1);
   Mat mtx (img) ; //转换IplImage*-> Mat
   ```

4) 【方法四】利用Create()函数
   
   方法四是利用Mat类中的Create()成员函数进行Mat类的初始化操作，示范代码如下。
   ```C++
   M.create ( 4 ,4,cv_8UC(2));
   cout <<"M= "<< endl <<" "<<M<<endl << endl;
   ```

5) 【方法五】采用Matlab式的初始化方式
   
   方法五采用Matlab形式的初始化方式: zeros()，ones()，eyes()。使用以下方式指定尺寸和数据类型:
   ```C++
   Mat E= Mat::eye (4,4, cV_64F);
   cout << "E=" << endl << " " << E << endl << endl;
   Mat O = Mat::ones (2,2, cV_32F);
   cout << "O=" << endl << " " << O << endl << endl;
   Mat Z = Mat::zeros (3,3,cv_8UC1);
   cout << "Z=" << endl << " " << Z << endl << endl;
   ```

6) 【方法六】对小矩阵使用逗号分隔式初始化函数
   方法六为对小矩阵使用逗号分隔式初始化函数，示范代码如下。
   ```C++
   Mat C = (Mat_<double>(3,3)<< 0，-1，0，-1，5,-1,0，-1,0);
   cout << "C = " << endl << " "<< C << endl << endl;
   ```

7) 【方法七】为已存在的对象创建新信息头
   
   方法七为使用成员函数clone()或者copyTo()为一个已存在的Mat对象创建一个新的信息头，示范代码如下。
   ```C++
   Mat RowClone = c.row (1).clone () ;
   cout <<"RowClone = " << endl << " " << RowClone << endl << endl;
   ```

### 程序：创建Mat
代码：
```C++
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

int main()
{
    Mat M(3, 2, CV_8UC3, Scalar(0, 0, 255));
    /*
       3 行 2 列
       CV_[位数][带符号与否][类型前缀]C[通道数]
       CV_8UC3 : 表示使用8位的unsigned char型，每个像素由三个元素组成三通道
       Scalar : short类型的向量
     */
    cout << "M = " << endl << " " << M << endl << endl;
    return 0;
}
```

实验结果：
![avatar](\picture\12.Mat类使用.png)

### 格式化输出方法

1) 【风格一】OpenCV默认风格
   
   风格一为OpenCV 默认风格的输出方法，如下。
   ```C++
   cout << "r (0pencv默认风格)="<<r <<";"<< endl << endl;
   ```

### 示例程序：基础图像容器Mat类的使用

### 基本图形的绘制
代码：
```C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
using namespace cv;

#define WINDOW_NAME1 "[绘制图 1]" //为窗口标题定义的宏
#define WINDOW_NAME2 "[绘制图 2]" //为窗口标题定义的宏
#define WINDOW_WIDTH 600//定义窗口大小的宏

void DrawEllipse(Mat img, double angle)
{
	int thickness = 2;
	int lineType = 8;

	ellipse(img,
		Point(WINDOW_WIDTH / 2.0, WINDOW_WIDTH / 2.0),
		Size(WINDOW_WIDTH / 4.0, WINDOW_WIDTH / 16.0),
		angle,
		0,
		360,
		Scalar(255, 0, 0),
		thickness,
		lineType);
}

void DrawFilledCircle(Mat img, Point center)
{
	int thickness = -1;
	int lineType = 8;

	circle(img,
		center,
		WINDOW_WIDTH / 32.0,
		Scalar(0, 0, 255),
		thickness,
		lineType);
}

void DrawPolygon(Mat img)
{
	int lineType = 8;

	/** 创建一些点 */
	Point rook_points[1][20];
	rook_points[0][0] = Point(WINDOW_WIDTH / 4.0, 7 * WINDOW_WIDTH / 8.0);
	rook_points[0][1] = Point(3 * WINDOW_WIDTH / 4.0, 7 * WINDOW_WIDTH / 8.0);
	rook_points[0][2] = Point(3 * WINDOW_WIDTH / 4.0, 13 * WINDOW_WIDTH / 16.0);
	rook_points[0][3] = Point(11 * WINDOW_WIDTH / 16.0, 13 * WINDOW_WIDTH / 16.0);
	rook_points[0][4] = Point(19 * WINDOW_WIDTH / 32.0, 3 * WINDOW_WIDTH / 8.0);
	rook_points[0][5] = Point(3 * WINDOW_WIDTH / 4.0, 3 * WINDOW_WIDTH / 8.0);
	rook_points[0][6] = Point(3 * WINDOW_WIDTH / 4.0, WINDOW_WIDTH / 8.0);
	rook_points[0][7] = Point(26 * WINDOW_WIDTH / 40.0, WINDOW_WIDTH / 8.0);
	rook_points[0][8] = Point(26 * WINDOW_WIDTH / 40.0, WINDOW_WIDTH / 4.0);
	rook_points[0][9] = Point(22 * WINDOW_WIDTH / 40.0, WINDOW_WIDTH / 4.0);
	rook_points[0][10] = Point(22 * WINDOW_WIDTH / 40.0, WINDOW_WIDTH / 8.0);
	rook_points[0][11] = Point(18 * WINDOW_WIDTH / 40.0, WINDOW_WIDTH / 8.0);
	rook_points[0][12] = Point(18 * WINDOW_WIDTH / 40.0, WINDOW_WIDTH / 4.0);
	rook_points[0][13] = Point(14 * WINDOW_WIDTH / 40.0, WINDOW_WIDTH / 4.0);
	rook_points[0][14] = Point(14 * WINDOW_WIDTH / 40.0, WINDOW_WIDTH / 8.0);
	rook_points[0][15] = Point(WINDOW_WIDTH / 4.0, WINDOW_WIDTH / 8.0);
	rook_points[0][16] = Point(WINDOW_WIDTH / 4.0, 3 * WINDOW_WIDTH / 8.0);
	rook_points[0][17] = Point(13 * WINDOW_WIDTH / 32.0, 3 * WINDOW_WIDTH / 8.0);
	rook_points[0][18] = Point(5 * WINDOW_WIDTH / 16.0, 13 * WINDOW_WIDTH / 16.0);
	rook_points[0][19] = Point(WINDOW_WIDTH / 4.0, 13 * WINDOW_WIDTH / 16.0);

	const Point* ppt[1] = { rook_points[0] };
	int npt[] = { 20 };

	fillPoly(img,
		ppt,
		npt,
		1,
		Scalar(255, 255, 255),
		lineType);
}

void DrawLine(Mat img, Point start, Point end)
{
	int thickness = 2;
	int lineType = 8;
	line(img,
		start,
		end,
		Scalar(0, 0, 0),
		thickness,
		lineType);
}

int main(void)
{
	//创建空白的Mat图像
	Mat atomImage = Mat::zeros(WINDOW_WIDTH, WINDOW_WIDTH, CV_8UC3); 
	Mat rookImage = Mat::zeros(WINDOW_WIDTH, WINDOW_WIDTH, CV_8UC3);

	// [1.1]先绘制出椭圆
	DrawEllipse(atomImage, 90); 
	DrawEllipse(atomImage, 0); 
	DrawEllipse(atomImage, 45); 
	DrawEllipse(atomImage, -45);

	// [1.2]再绘制圆心
	DrawFilledCircle(atomImage, Point(WINDOW_WIDTH / 2, WINDOW_WIDTH / 2));

	// [2.1]先绘制出椭圆
	DrawPolygon(rookImage);

	// [2.2]绘制矩形
	rectangle ( rookImage,
		Point(0, 7 * WINDOW_WIDTH / 8),
		Point(WINDOW_WIDTH, WINDOW_WIDTH), 
		Scalar( 0, 255, 255 ),
		-1,
		8);

	// [2.3]绘制一些线段
	DrawLine(rookImage, Point(0, 15 * WINDOW_WIDTH / 16), Point(WINDOW_WIDTH, 15 * WINDOW_WIDTH / 16));

	DrawLine(rookImage, Point(WINDOW_WIDTH / 4, 7 * WINDOW_WIDTH / 8), Point(WINDOW_WIDTH / 4, WINDOW_WIDTH));

	DrawLine(rookImage, Point(WINDOW_WIDTH / 2, 7 * WINDOW_WIDTH / 8), Point(WINDOW_WIDTH / 2, WINDOW_WIDTH));

	DrawLine(rookImage, Point(3 * WINDOW_WIDTH / 4, 7 * WINDOW_WIDTH / 8), Point(3 * WINDOW_WIDTH / 4, WINDOW_WIDTH));

	imshow( WINDOW_NAME1, atomImage );

	moveWindow(WINDOW_NAME1, 0, 200);

	imshow( WINDOW_NAME2, rookImage );

	moveWindow(WINDOW_NAME2, WINDOW_WIDTH, 200);

	waitKey(0); 
	return(0);
}
```

实验结果：
![avatar](\picture\13.基本图形绘制.png)

## 五、core组件进阶

### 访问图像中的像素

#### 图像在内存之中的存储方式

OpenCV 中子列的通道顺序是反过来的——BGR而不是RGB。很多情况下，因为内存足够大，可实现连续存储，因此，图像中的各行就能一行一行地连接起来，形成一个长行。连续存储有助于提升图像扫描速度，我们可以使用 isContinuous() 来判断矩阵是否是连续存储的。

#### 颜色空间缩减

颜色空间缩减(color space reduction)便可以派上用场了，它在很多应用中可以大大降低运算复杂度。颜色空间缩减的做法是:将现有颜色空间值除以某个输入值，以获得较少的颜色数。也就是“做减法”，比如颜色值0到9可取为新值0，10到19可取为10，以此类推。

uchar类型的三通道图像,每个通道取值可以是0~255，于是就有256×256个不同的值。我们可以定义:
- 0~9范围的像素值为0;
- 10~19范围的像素值为10;
- 20～29范围的像素值为20。

uchar (无符号字符，即О到255之间取值的数) 类型的值除以int值，结果仍是 char。因为结果是 char 类型的，所以求出来小数也要向下取整。

#### 访问图像中像素的三类方法

任何图像处理算法,都是从操作每个像素开始的。即使我们不会使用 OpenCV 提供的各种图像处理函数，只要了解了图像处理算法的基本原理，也可以写出具有相同功能的程序。在 OpenCV 中，提供了三种访问每个像素的方法。

### 程序：访问图像像素
代码：
```C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

void colorReduce(Mat& inputImage, Mat& outputImage, int div)
{
	......
}

int main()
{
	// [1]创建原始图并显示
	Mat srcImage = imread("1.jpg"); 
	imshow("原始图像", srcImage);

	// [2]按原始图的参数规格来创建创建效果图
	Mat dstImage;
	dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());//效果图的大小、类型与原图片相同

	// [3]记录起始时间
	double time0 = static_cast<double>(getTickCount());

	// [4]调用颜色空间縮减函數
	colorReduce(srcImage, dstImage, 32);

	// [5]计算运行时间并输出
	time0 = ((double)getTickCount() - time0) / getTickFrequency(); cout << "此方法运行时间为: " << time0 << "秒" << endl; //输 出运行时间.

	// [6]显示效果图
	imshow("效果图", dstImage); 
	waitKey(0);
}
```

- 方法一 指针访问：C操作符[];
  ```C++
  void colorReduce(Mat& inputImage, Mat& outputImage, int div)
  {
	//参数准备
	outputImage = inputImage.clone(); //复制实参到临时变量
	int rowNumber = outputImage.rows; //行数
	int colNumber = outputImage.cols * outputImage.channels(); //列数x通道数=每一行元素的个数

	//双重循环，遍历所有的像素值
	for (int i = 0; i < rowNumber; i++) //行循环
	{
		uchar* data = outputImage.ptr<uchar>(i); //获取第i行的首地址
		for (int j = 0; j < colNumber; j++) //列循环
		{
			data[j] = data[j] / div * div + div / 2;
		} //行处理结束
	}
  }
  ```

- 方法二 迭代器 iterator;
  ```C++
  void colorReduce(Mat& inputImage, Mat& outputImage, int div)
  {
	//参数准备
	outputImage = inputImage.clone(); //复制实参到临时变量
	//获取迭代器
	Mat_<Vec3b>::iterator it = outputImage.begin<Vec3b>(); //初始位置
	Mat_<Vec3b>::iterator itend = outputImage.end<Vec3b>(); //终止位置

	//存取彩色图像像素
	for (; it != itend; ++it) 
	{
		(*it)[0] = (*it)[0] / div * div + div / 2;
		(*it)[1] = (*it)[1] / div * div + div / 2;
		(*it)[2] = (*it)[2] / div * div + div / 2;
	}
  }
  ```

- 方法三 动态地址计算。
  ```C++
  void colorReduce(Mat& inputImage, Mat& outputImage, int div)
  {
	//参数准备
	outputImage = inputImage.clone(); //复制实参到临时变量
	int rowNumber = outputImage.rows; //行数
	int colNumber = outputImage.cols; //列数

	//存取彩色图像像素
	for (int i = 0; i < rowNumber; i++) 
	{
		for (int j = 0; j < colNumber; j++) 
		{
			outputImage.at<Vec3b>(i, j)[0] = outputImage.at<Vec3b>(i, j)[0] / div * div + div / 2; //蓝色通道
			outputImage.at<Vec3b>(i, j)[1] = outputImage.at<Vec3b>(i, j)[1] / div * div + div / 2; //绿色通道
			outputImage.at<Vec3b>(i, j)[2] = outputImage.at<Vec3b>(i, j)[2] / div * div + div / 2; //红色通道
		} //行处理结束
	}
  }
  ```

实验结果：
![avatar](\picture\14.指针访问像素.png)

这三种方法在访问速度上略有差异。debug 模式下，这种差异非常明显，不过在 release 模式下，这种差异就不太明显了。我们通过一组程序来说明这几种方法。程序的目的是减少图像中颜色的数量，比如原来的图像是是256种颜色，我们希望将它变成64种颜色，那只需要将原来的颜色除以4（整除）以后再乘以4就可以了。

### ROI区域图像叠加&图像混合

#### 感兴趣区域：ROI
感兴趣区域(ROI，region of interest)，来专注或者简化工作过程。也就是从图像中选择的一个图像区域，这个区域是图像分析所关注的重点。我们圈定这个区域，以便进行进一步处理。而且，使用ROI指定想读入的目标，可以减少处理时间，增加精度，给图像处理来带不小的便利。

定义ROI区域有两种方法:第一种是使用表示矩形区域的Rect。它指定矩形的左上角坐标(构造函数的前两个参数)和矩形的长宽（构造函数的后两个参数)以定义一个矩形区域。

//方法一 imageROI=image (Rect (500,250,logo.cols, logo .rows ) ) ;

另一种定义ROI的方式是指定感兴趣行或列的范围(Range)。Range是指从起始索引到终止索引(不包括终止索引）的一连段连续序列。cRange可以用来定义Range。

//方法二 imageROI=image (Range(250,250+logoImage.rows) ,Range (200,200+logoImage.cols));

#### 计算数加权和：addWeighted()函数

这个函数的作用是计算两个数组（图像阵列）的加权和。原型如下:
```C++
void (InputArray src1,double alpha,InputArray src2，double beta,doublegamma,OutputArray dst, int dtype=-1);
```

- 第一个参数，InputArray 类型的 src1，表示需要加权的第一个数组，常常填一个Mat;
- 第二个参数，double 类型的 alpha，表示第一个数组的权重;
- 第三个参数，InputArray 类型的 src2，表示第二个数组，它需要和第一个数组拥有相同的尺寸和通道数;
- 第四个参数，double 类型的 beta，表示第二个数组的权重值;
- 第五个参数，double 类型的 gamma，一个加到权重总和上的标量值。其含义通过接下来列出的式子自然会理解;
- 第六个参数，OutputArray 类型的 dst，输出的数组，它和输入的两个数组拥有相同的尺寸和通道数;
- 第七个参数，int 类型的 dtype，输出阵列的可选深度，有默认值-1。当两个输入数组具有相同的深度时，这个参数设置为-1（默认值)，即等同于 src1.depth()。

### 程序：初级图像混合
代码：
```C++
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
using namespace cv;
using namespace std;

bool ROI_AddImage(); 
bool LinearBlending();
bool ROI_LinearBlending();

int main()
{
	system("color 5E");
	if (ROI_AddImage() && LinearBlending() && ROI_LinearBlending())
	{
		cout << endl << "运行成功，得出了你需要的图像~! : )";
	}
	waitKey(0); 
	return 0;
}

bool ROI_AddImage()
{
	// [1]读入图像
	Mat srcImagel = imread("dota5.jpg");
	Mat logoImage = imread("dota_logo.jpg");

	if (!srcImagel.data) { printf("读取srcImagel错误~! \n"); return false; }

	if (!logoImage.data) { printf("读取logoImage错误~! \n"); return false; }

	//[2]定义一个Mat类型并给其设定ROI区域
	Mat imageROI = srcImagel(Rect(200, 250, logoImage.cols, logoImage.rows));

	// [3]加载掩模(必须是灰度图)
	Mat mask = imread("dota_logo.jpg", 0);

	// [4]将掩膜复制到ROI
	logoImage.copyTo(imageROI, mask);

	// [5]显示结果
	namedWindow("<1>利用ROI实现图像叠加示例窗口");
	imshow("<1>利用ROI实现图像叠加示例窗口", srcImagel);

	return true;
}

bool LinearBlending()
{
	//[0]定义一些局部变量
	double alphaValue = 0.5;
	double betaValue;
	Mat srcImage2, srcImage3, dstImage;

	// [1]读取图像(两幅图片需为同样的类型和尺寸)
	srcImage2 = imread("1.jpg");
	srcImage3 = imread("2.jpg");

	if (!srcImage2.data) { printf("读取srcImage2错误~! \n"); return false; }

	if (!srcImage3.data) { printf("读取srcImage3错误~! \n"); return false; }

	// [2]进行图像混合加权操作
	betaValue = (1.0 - alphaValue);
	addWeighted(srcImage2, alphaValue, srcImage3, betaValue, 0.0, dstImage);

	// [3]创建并显示原图窗口
	namedWindow("<2>线性混合示例窗口[原图]", 1);
	imshow("<2>线性混合示例窗口[原图]", srcImage2);

	namedWindow("<3>线性混合示例窗口[ 效果图] ", 1); 
	imshow("<3>线性混合示例窗口[效果图] ", dstImage);

	return true;
}

bool ROI_LinearBlending()
{
	//[1]读取图像.

	Mat srcImage4 = imread("dota5.jpg", 1); 
	Mat logoImage = imread("dota_logo. jpg");

	if (!srcImage4.data) { printf("读取srcImage4错误~! \n"); return false; }

	if (!logoImage.data ) { printf("读取logoImage错误~! \n"); return false; }
	//[2]定义一个Mat类型并給其設定ROI区域
	Mat imageROI;
	
	//方法一
	imageROI = srcImage4(Rect(200, 250, logoImage.cols, logoImage.rows));
	//方法二
	//imageROI = srcImage4 (Range (250 , 250+logoImage . rows) , Range (200, 200+logoImage.cols));
	
	//[3]將logo加到原困上
	addWeighted(imageROI, 0.5, logoImage, 0.3, 0., imageROI);
	//[4]显示結果
	namedWindow("<4>区域线性图像混合示例窗口");
	imshow("<4>区域线性图像混合示例窗口", srcImage4);
	
	return true;
}
```

实验结果：
![avatar](\picture\17.初级图像混合.png)

### 分离颜色通道、多通道图像混合

#### 通道分离：split()函数
split() 函数用于将一个多通道数组分离成几个单通道数组。
#### 通道合并：merge()函数
merge() 函数是 split() 函数的逆向操作——将多个数组合并成一个多通道的数组。它通过组合一些给定的单通道数组，将这些孤立的单通道数组合并成一个多通道的数组，从而创建出一个由多个单通道阵列组成的多通道阵列。

### 程序：多通道图像混合
代码：
```C++
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace std;

bool ROI_AddImage(); 
bool LinearBlending();
bool ROI_LinearBlending();

int main()
{
	system("color 5E");

	if (ROI_AddImage() && LinearBlending() && ROI_LinearBlending())
	{
		cout << endl << "运行成功，得出了你需要的图像~! : )";
	}

	waitKey(0); 
	return 0;
}

bool ROI_AddImage()
{
	// [1]读入图像
	Mat srcImagel = imread("dota5.jpg");
	Mat logoImage = imread("dota_logo.jpg");

	if (!srcImagel.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~! \n"); return false; }

	if (!logoImage.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~! \n"); return false; }

	//[2]定义一个Mat类型并给其设定ROI区域
	Mat imageROI = srcImagel(Rect(200, 250, logoImage.cols, logoImage.rows));

	// [3]加载掩模(必须是灰度图)
	Mat mask = imread("dota_logo.jpg", 0);

	// [4]将掩膜复制到ROI
	logoImage.copyTo(imageROI, mask);

	// [5]显示结果
	namedWindow("<1>利用ROI实现图像叠加示例窗口");
	imshow("<1>利用ROI实现图像叠加示例窗口", srcImagel);

	return true;
}

bool LinearBlending()
{
	//[0]定义一些局部变量
	double alphaValue = 0.5;
	double betaValue;
	Mat srcImage2, srcImage3, dstImage;

	// [1]读取图像(两幅图片需为同样的类型和尺寸)
	srcImage2 = imread("1.jpg");
	srcImage3 = imread("2.jpg");

	if (!srcImage2.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~! \n"); return false; }

	if (!srcImage3.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~! \n"); return false; }

	// [2]进行图像混合加权操作
	betaValue = (1.0 - alphaValue);
	addWeighted(srcImage2, alphaValue, srcImage3, betaValue, 0.0, dstImage);

	// [3]创建并显示原图窗口
	namedWindow("<2>线性混合示例窗口[原图]", 1);
	imshow("<2>线性混合示例窗口[原图]", srcImage2);

	namedWindow("<3>线性混合示例窗口[ 效果图] ", 1); 
	imshow("<3>线性混合示例窗口[效果图] ", dstImage);

	return true;
}

bool ROI_LinearBlending()
{
	//[1]读取图像.

	Mat srcImage4 = imread("dota5.jpg", 1); 
	Mat logoImage = imread("dota_logo. jpg");

	if (!srcImage4.data) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~! \n"); return false; }

	if (!logoImage.data ) { printf("读取图片错误，请确定目录下是否有imread函数指定图片存在~! \n"); return false; }
	//[2]定义一个Mat类型并給其設定ROI区域
	Mat imageROI;
	
	//方法一
	imageROI = srcImage4(Rect(200, 250, logoImage.cols, logoImage.rows));
	//方法二
	//imageROI = srcImage4 (Range (250 , 250+logoImage . rows) , Range (200, 200+logoImage.cols));
	
	//[3]將logo加到原困上
	addWeighted(imageROI, 0.5, logoImage, 0.3, 0., imageROI);
	//[4]显示結果
	namedWindow("<4>区域线性图像混合示例窗口");
	imshow("<4>区域线性图像混合示例窗口", srcImage4);
	
	return true;
}
```

实验结果：
![avatar](\picture\18.多通道图像混合.png)

### 图像对比度、亮度调整

#### 理论
一般的图像处理算子都是一个函数，它接受一个或多个输入图像，并产生输出图像。

图像亮度和对比度的调整操作，其实属于图像处理变换中比较简单的一种——点操作( pointoperators)。点操作有一个特点:仅仅根据输入像素值(有时可加上某些全局信息或参数)，来计算相应的输出像素值。这类算子包括亮度（brightness）和对比度( contrast）调整、颜色校正(colorcorrection)和变换( transformations）。

两种最常用的点操作(或者说点算子)是乘上一个常数(对应对比度的调节)以及加上一个常数（对应亮度值的调节)。

### 程序：图像调整
代码：
```C++
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
using namespace std;
using namespace cv;

static void on_ContrastAndBright(int, void*);
static void ShowHelpText();


int g_nContrastValue; //对比度值
int g_nBrightValue; //亮度值
Mat g_srcImage, g_dstImage;

int main()
{
	//【1】读取输入图像
	g_srcImage = imread("1.jpg");

	if (!g_srcImage.data) { printf("读取图片错误, 请确定目录下是否有imread函数指定图片存在~!"); return false; }

	g_dstImage = Mat::zeros(g_srcImage.size(), g_srcImage.type());

	//【2】设定对比度和亮度的初值
	g_nContrastValue = 80;
	g_nBrightValue = 80;

	//【3】创建效果窗口
	namedWindow(" [ 效果窗口 ] ", 1);

	//【4】进行回调函数初始化
	createTrackbar("对比度: ", " [效果窗口图] ", &g_nContrastValue, 300, on_ContrastAndBright);

	createTrackbar("亮 度: ", " [效果窗口图] ", &g_nBrightValue, 200, on_ContrastAndBright);

	//【5】进行回调函数初始化
	on_ContrastAndBright(g_nContrastValue, 0); 
	on_ContrastAndBright(g_nBrightValue, 0);

	//【6】按下“q”键时，程序退出
	while (char(waitKey(1)) != 'q') {}
	return 0;
}
static void on_ContrastAndBright(int, void*)
{
	//创建窗口
	namedWindow("[原始图窗口]", 1);

	// 三个for循环，执行运算 g_dstImage(i,j) = a*g_srcImage(i,j) + b
	for (int y = 0; y < g_srcImage.rows; y++) 
	{
		for (int x = 0; x < g_srcImage.cols; x++) 
		{
			for (int c = 0; c < 3; c++) 
			{
				g_dstImage.at<Vec3b>(y, x)[c] = saturate_cast<uchar>((g_nContrastValue * 0.01) * (g_srcImage.at<Vec3b>(y, x)[c]) + g_nBrightValue);
			}
		}
	}

	//显示图像
	imshow("[原始窗口图]", g_srcImage);
	imshow("[原始窗口图]", g_dstImage);
}
```

实验结果：
![avatar](\picture\19.图像调整.png)

### 离散傅里叶变换

#### 原理
简单来说，对一张图像使用傅里叶变换就是将它分解成正弦和余弦两部分，也就是将图像从空间域(spatial domain）转换到频域( frequency domain)。

这一转换的理论基础为:任一函数都可以表示成无数个正弦和余弦函数的和的形式。傅里叶变换就是一个用来将函数分解的工具。

#### dft()函数
dft函数的作用是对一维或二维浮点数数组进行正向或反向离散傅里叶变换。
```C++
void dft(InputArray src,OutputArray dst, int flags=0，intnonzeroRows=0 )
```
- 第一个参数，InputArray 类型的 src。输入矩阵，可以为实数或者虚数。
- 第二个参数，OutputArray 类型的 dst。函数调用后的运算结果存在这里，其尺寸和类型取决于标识符，也就是第三个参数 flags。
- 第三个参数，int 类型的 flags。转换的标识符，有默认值0，取值可以为标识符的结合。

#### 扩充图像边界：copyMakeBorderO函数
copyMakeBorder 函数的作用是扩充图像边界。
```C++
void copyMakeBorder(InputArray src，outputArray dst,int top, int bottom，int left,int right,int borderType,const scalar& value=Scalar ( ) )
```
- 第一个参数，InputArray 类型的 src，输入图像，即源图像，填 Mat 类的对象即可。
- 第二个参数，OutputArray 类型的 dst，函数调用后的运算结果存在这里，即这个参数用于存放函数调用后的输出结果，需和源图片有一样的尺寸和类型，且 size应该为 Size ( src.cols+left+right, src.rows+top+bottom)。
- 接下来的4个参数分别为 int 类型的 top、bottom、left、right，分别表示在源图像的四个方向上扩充多少像素，例如 top=2，bottom=2，left=2，right=2 就意味着在源图像的上下左右各扩充两个像素宽度的边界。
- 第七个参数，borderType 类型的，边界类型，常见取值为 BORDER_CONSTANT，可参考 borderInterpolate() 得到更多的细节。
- 第八个参数，const Scalar& 类型的 value，有默认值 Scalar()，可以理解为默认值为0。当 borderType 取值为 BORDER_CONSTANT 时，这个参数表示边界值。

### 程序：傅里叶变换
代码：
```C++
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
using namespace cv;

int main()
{
	// [1]以灰度模式读取原始图像并显示
	Mat srcImage = imread("1.jpg", 0);

	if (!srcImage.data) { printf("读取图片 错误，请确定目录下是否有imread函数指定图片存在~! \n"); return false; }

	imshow("原始图像", srcImage);

	int ShowHelpText();

	// [2]将输入图像延扩到最佳的尺寸，边界用0补充
	int m = getOptimalDFTSize(srcImage.rows);
	int n = getOptimalDFTSize(srcImage.cols);

	//将添加的像素初始化为0.
	Mat padded;
	copyMakeBorder(srcImage, padded, 0, m - srcImage.rows, 0, n - srcImage.cols, BORDER_CONSTANT, Scalar::all(0));

	// [3]为傅里叶变换的结果(实部和虚部)分配存储空间。
	//将planes数组组合合并成一个多通道的数组complexI
	Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI; 
	merge(planes, 2, complexI);
	
	//[4]进行就地离散傅里叶变换
	dft(complexI, complexI);
	//[5]将复数转换为幅值，即=> 1og(1 + sqrt (Re(DFT(I))^2+ Im(DFT(I))^2))
	split (complexI, planes); // 将多通道数组complexI分离成几个单通道数组，planes[0] = Re(DFT(I)， planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magnitudeImage = planes[0];

	//[6] 进行对数尺度(logarithmic scale)缩放
	magnitudeImage += Scalar::all(1);
	log(magnitudeImage, magnitudeImage); //求自然对数

	//[7]剪切和重分布幅度图象限
	//若有奇数行或奇数列，进行频谱裁剪
	magnitudeImage = magnitudeImage(Rect(0, 0, magnitudeImage.cols & -2, magnitudeImage.rows & - 2));

	//重新排列傅里叶图像中的象限，使得原点位于图像中心
	int cx = magnitudeImage.cols / 2;
	int cy = magnitudeImage.rows / 2;
	Mat q0(magnitudeImage, Rect(0, 0, cx, cy)); // ROI 区域的左上
	Mat q1(magnitudeImage, Rect(cx, 0, cx, cy)); // ROI 区域的右下
	Mat q2(magnitudeImage, Rect(0, cy, cx, cy)); // ROI 区域的左下
	Mat q3(magnitudeImage, Rect(cx, cy, cx, cy)); // ROI 区域的右下
	// 交换象限（左上与右下进行交换）
	Mat tmp;
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	// 交换象限（右上与左下进行交换）

	q1.copyTo(tmp);
	q2.copyTo(q1);
	tmp.copyTo(q2);

	//[8]归一化，用0到1之间的浮点值将矩阵变换为可视的图像格式

	//此句代码的openCV2版为: .

	//normalize (magnitudeImage, magnitudeImage, 0, l, CV_MINMAX);
	//此句代码的openCV3版为:
	normalize(magnitudeImage, magnitudeImage, 0, 1, NORM_MINMAX);

	// [9]显示效果图
	imshow("频谱幅值", magnitudeImage); 
	waitKey();

	return 0;
}
```

实验结果：
![avatar](\picture\20.傅里叶函数.png)

### 输入输出 XML 和 YAML 文件

#### 简介
- XML，即 eXtensible Markup Language，翻译成中文为“可扩展标识语言”。首先，XML是一种元标记语言。所谓“元标记”，就是开发者可以根据自身需要定义自己的标记，比如可以定义标记<book>、<name>。任何满足XML命名规则的名称都可以标记，这就向不同的应用程序打开了的大门。此外，XML是一种语义/结构化语言，它描述了文档的结构和语义。
- YAML是“YAML Ain't a Markup Language”（译为“YAML不是一种置标语言”)的递回缩写。在开发的这种语言时，YAML的原意是:“Yet Another MarkupLanguage”(仍是一种置标语言)，但为了强调这种语言以数据为中心，而不是以置标语言为重点，而用返璞词进行重新命名。YAML是一个可读性高，用来表达资料序列的格式。它参考了其他多种语言，包括:XML、C语言、Python、Perl,以及电子邮件格式RFC2822。

### 程序：XML 和 YAML 文件写入
代码：
```C++
#include "opencv2/opencv.hpp"  
#include <time.h>  
using namespace cv;

int main()
{
	//改变console字体颜色
	system("color 5F");


	//初始化
	FileStorage fs("test.yaml", FileStorage::WRITE);

	//开始文件写入
	fs << "frameCount" << 5;
	time_t rawtime; time(&rawtime);
	fs << "calibrationDate" << asctime(localtime(&rawtime));
	Mat cameraMatrix = (Mat_<double>(3, 3) << 1000, 0, 320, 0, 1000, 240, 0, 0, 1);
	Mat distCoeffs = (Mat_<double>(5, 1) << 0.1, 0.01, -0.001, 0, 0);
	fs << "cameraMatrix" << cameraMatrix << "distCoeffs" << distCoeffs;
	fs << "features" << "[";
	for (int i = 0; i < 3; i++)
	{
		int x = rand() % 640;
		int y = rand() % 480;
		uchar lbp = rand() % 256;

		fs << "{:" << "x" << x << "y" << y << "lbp" << "[:";
		for (int j = 0; j < 8; j++)
			fs << ((lbp >> j) & 1);
		fs << "]" << "}";
	}
	fs << "]";
	fs.release();

	printf("\n文件读写完毕，请在工程目录下查看生成的文件~");
	getchar();

	return 0;
}
```

实验结果：
![avatar](\picture\21.文件写入.png)

### 程序：XML 和 YAML 文件读取
代码：
```C++
#include "opencv2/opencv.hpp"  
#include <time.h>  
using namespace cv;
using namespace std;

int main()
{
	//改变console字体颜色
	system("color 6F");

	//初始化
	FileStorage fs2("test.yaml", FileStorage::READ);

	// 第一种方法，对FileNode操作
	int frameCount = (int)fs2["frameCount"];

	std::string date;
	// 第二种方法，使用FileNode运算符> > 
	fs2["calibrationDate"] >> date;

	Mat cameraMatrix2, distCoeffs2;
	fs2["cameraMatrix"] >> cameraMatrix2;
	fs2["distCoeffs"] >> distCoeffs2;

	cout << "frameCount: " << frameCount << endl
		<< "calibration date: " << date << endl
		<< "camera matrix: " << cameraMatrix2 << endl
		<< "distortion coeffs: " << distCoeffs2 << endl;

	FileNode features = fs2["features"];
	FileNodeIterator it = features.begin(), it_end = features.end();
	int idx = 0;
	std::vector<uchar> lbpval;

	//使用FileNodeIterator遍历序列
	for (; it != it_end; ++it, idx++)
	{
		cout << "feature #" << idx << ": ";
		cout << "x=" << (int)(*it)["x"] << ", y=" << (int)(*it)["y"] << ", lbp: (";
		// 我们也可以使用使用filenode > > std::vector操作符很容易的读数值阵列
		(*it)["lbp"] >> lbpval;
		for (int i = 0; i < (int)lbpval.size(); i++)
			cout << " " << (int)lbpval[i];
		cout << ")" << endl;
	}
	fs2.release();

	//程序结束，输出一些帮助文字
	printf("\n文件读取完毕，请输入任意键结束程序~");
	getchar();

	return 0;
}
```

实验结果：
![avatar](\picture\22.文件读取.png)
