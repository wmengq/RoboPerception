#4.1基础图像容器Mat
##4.1.1 数字图像存储概述
我们可以通过各种各样的方法从现实世界获取到数字图像，如借助相机、扫描仪、计算机摄像头或磁共振成像等。通常由显示屏上看到的都是真实而漂亮的图像，但是这些图像在转化到我们的数字设备中时，记录的却是图像中的每个点的数值。
##4.1.2Mat结构的使用
关于Mat类，首先我们要知道的是:
(1)不必再手动为其开辟空间。
(2)不必再在不需要时立即将空间释放。
总而言之，Mat是一个类，由两个数据部分组成:矩阵头（包含矩阵尺寸、存储方法、存储地址等信息）和一指向存储所有像素值的矩阵（根据所选存储方法不同，矩阵可以是不同的维数）的指针。矩阵头的尺寸是常数值，但矩阵本身的尺寸会依据图像的不同而不同，通常比矩阵头的尺寸大数个数量级。
##4.1.3像素值的存储方法
存储像素值需要指定颜色空间和数据类型。其中，颜色空间是指针对一个给定的颜色，如何组合颜色元素以对其编码。最简单的颜色空间要属灰度级空间，只处理黑色和白色，对它们进行组合便可以产生不同程度的灰色。
##4.1.4显式创建Mat对象的七种方法
代码：
include<opencv2\opencv.hpp>
include<iostream>
using namespace std;
using namespace cv;

int main() {

	// 指定存储元素的数据类型以及每个矩阵点的通道数：
	//		CV_[位数][是否带符号][类型前缀]C[通道数]

	// 方法1：使用Mat()构造函数
	Mat M(2, 2, CV_8UC3, Scalar(0, 0, 255));
	cout << "M = " << endl << " " << M << endl << endl;
	cout << "---------------------------------------" << endl;

	// 方法2：利用create()函数
	M.create(4, 4, CV_8UC(3));
	cout << "M = " << endl << " " << M << endl << endl;
	cout << "---------------------------------------" << endl;

	
	// 方法3：采用Matlab式的初始化方式
	Mat E = Mat::eye(4, 4, CV_64F);
	cout << "E = " << endl << " " << E << endl << endl;
	cout << "---------------------------------------" << endl;

	Mat O = Mat::ones(2, 2, CV_32F);
	cout << "O = " << endl << " " << O << endl << endl;
	cout << "---------------------------------------" << endl;

	Mat Z = Mat::zeros(3, 3, CV_8UC1);
	cout << "Z = " << endl << " " << Z << endl << endl;
	cout << "---------------------------------------" << endl;

	// 方法4：对小矩阵使用逗号分隔式初始化函数
	//						分隔式顺序按列从左到右
	Mat C = (Mat_<double>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	cout << "C = " << endl << " " << C << endl << endl;
	cout << "---------------------------------------" << endl;

	// 方法5：为已存在的对象创建新信息头
	Mat RowClone = C.row(1).clone();
	cout << "RowClone = " << endl << " " << RowClone << endl << endl;
	cout << "---------------------------------------" << endl;

	// 使用randu()方法填充矩阵
	Mat r = Mat(10, 3, CV_8UC3);
	randu(r, Scalar::all(0), Scalar::all(255));
	cout << r<<endl << endl;
	cout << "------------------------------" << endl;
	
	system("pause");
}
代码结果：
![avatar](/picture/1.png)
![avatar](/picture/2.png)
##4.1.5 
代码：
#include<iostream>
#include<opencv2\opencv.hpp>
 
using namespace std;
using namespace cv;
 
int main()
{
	Mat r = Mat(10, 3, CV_8UC3);
	randu(r, Scalar::all(0), Scalar::all(255));//randu()函数产生的随机值来填充矩阵，需要给定一个上限和下限
	cout << "r(opencv默认风格)=" << r << ";" << endl << endl;//【1】opencv默认风格
	cout << "r(python风格)=" <<format(r, Formatter::FMT_PYTHON)<< ";" << endl << endl;//【2】python风格（opencv3版本）
	//cout<<"r(python风格)="<<format(r,"python")<<";"<<endl<<endl;//opencv2版本
	cout << "r(逗号分割风格)=" << format(r, Formatter::FMT_CSV) << ";" << endl << endl;//【3】逗号分割风格（CSV）
	//cout<<"r(逗号分割风格)="<<format(r,"csv")<<";"<<endl<<endl;//opencv2版本
	cout << "r(numpy风格)=" << format(r, Formatter::FMT_NUMPY) << ":" << endl << endl;//【4】numpy风格
	//cout<<"r(numpy风格)="<<format(r,"numpy")<<":"<<endl<<endl;//opencv2版本
	cout << "r(c语言风格)=" << format(r, Formatter::FMT_C) << ":" << endl << endl;//【5】c语言风格
	//cout<<"r(c语言风格)="<<format(r,"c")<<":"<<endl<<endl;//opencv2版本
 
 
}
代码结果：
![avatar](/picture/3.png)
![avatar](/picture/4.png)
##4.1.6输出其他常用数据结构
代码：
#include<iostream>
#include<opencv2\opencv.hpp>
 
using namespace std;
using namespace cv;
 
int main()
{
	Point2f p(6, 2);
	cout << "【二维点】p=" << p << ";\n" << endl;//【1】定义和输出二维点
	Point3f p3f(8, 2, 0);
	cout << "【三维点】p3f=" << p3f << ";\n" << endl;//【2】定义和输出三维点
	vector<float>v;
	v.push_back(3);
	v.push_back(5);
	v.push_back(7);
	cout << "【基于Mat的Vec】shortvec=" << Mat(v) << ";\n" << endl;//【3】定义和输出基于Mat的std::vector
	vector<Point2f>points(20);
	for (size_t i = 0; i < points.size(); ++i)
		points[i] = Point2f((float)(i * 5), (float)(i % 7));
	cout << "【二维点向量】points=" << points << ";";//【4】定义和输出Std::vector点
 
 
 
}
代码结果：
![avatar](/picture/5.png)
#4.2常用数据结构和函数
![avatar](/picture/6.png)
![avatar](/picture/7.png)
![avatar](/picture/8.png)
![avatar](/picture/9.png)
![avatar](/picture/10.png)
代码：
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
using namespace cv;
void main( )
{
    //【1】载入图片
    Mat srcImage=imread ( "1 .jpg" , 1), dstImage;
    //【2】转换颜色空间
    cvtcolor(srcImage , dst Image,COLOR_BGR2Lab);
//【3】显示效果图
imshow( "效果图" , dstImage);
//【4】保持窗口显示
waitKey ();
}
代码结果：
![avatar](/picture/11.png)
##4.3基本图形的绘制
·用于绘制直线的 line函数;
·用于绘制椭圆的ellipse函数;
·用于绘制矩形的rectangle函数;
·用于绘制圆的circle函数;
·用于绘制填充的多边形的fillPoly函数。
![avatar](/picture/12.png)
![avatar](/picture/13.png)
![avatar](/picture/14.png)
![avatar](/picture/15.png)
![avatar](/picture/16.png)
![avatar](/picture/17.png)
![avatar](/picture/18.png)
![avatar](/picture/19.png)
代码：
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
using namespace cv;

//此程序对于OpenCV3版需要额外包含头文件：
#include <opencv2/imgproc/imgproc.hpp>



//-----------------------------------【宏定义部分】-------------------------------------------- 
//		描述：定义一些辅助宏 
//------------------------------------------------------------------------------------------------ 
#define WINDOW_NAME1 "【绘制图1】"        //为窗口标题定义的宏 
#define WINDOW_NAME2 "【绘制图2】"        //为窗口标题定义的宏 
#define WINDOW_WIDTH 600//定义窗口大小的宏



//--------------------------------【全局函数声明部分】-------------------------------------
//		描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void DrawEllipse( Mat img, double angle );//绘制椭圆
void DrawFilledCircle( Mat img, Point center );//绘制圆
void DrawPolygon( Mat img );//绘制多边形
void DrawLine( Mat img, Point start, Point end );//绘制线段



//-----------------------------------【ShowHelpText( )函数】----------------------------------
//          描述：输出一些帮助信息
//----------------------------------------------------------------------------------------------
void ShowHelpText()
{
	//输出欢迎信息和OpenCV版本
	printf("\n\n\t\t\t非常感谢购买《OpenCV3编程入门》一书！\n");
	printf("\n\n\t\t\t此为本书OpenCV3版的第20个配套示例程序\n");
	printf("\n\n\t\t\t   当前使用的OpenCV版本为：" CV_VERSION );
	printf("\n\n  ----------------------------------------------------------------------------\n");
}




//---------------------------------------【main( )函数】--------------------------------------
//		描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main( void )
{

	// 创建空白的Mat图像
	Mat atomImage = Mat::zeros( WINDOW_WIDTH, WINDOW_WIDTH, CV_8UC3 );
	Mat rookImage = Mat::zeros( WINDOW_WIDTH, WINDOW_WIDTH, CV_8UC3 );

	ShowHelpText();
	// ---------------------<1>绘制化学中的原子示例图------------------------

	//【1.1】先绘制出椭圆
	DrawEllipse( atomImage, 90 );
	DrawEllipse( atomImage, 0 );
	DrawEllipse( atomImage, 45 );
	DrawEllipse( atomImage, -45 );

	//【1.2】再绘制圆心
	DrawFilledCircle( atomImage, Point( WINDOW_WIDTH/2, WINDOW_WIDTH/2) );

	// ----------------------------<2>绘制组合图-----------------------------
	//【2.1】先绘制出椭圆
	DrawPolygon( rookImage );

	// 【2.2】绘制矩形
	rectangle( rookImage,
		Point( 0, 7*WINDOW_WIDTH/8 ),
		Point( WINDOW_WIDTH, WINDOW_WIDTH),
		Scalar( 0, 255, 255 ),
		-1,
		8 );

	// 【2.3】绘制一些线段
	DrawLine( rookImage, Point( 0, 15*WINDOW_WIDTH/16 ), Point( WINDOW_WIDTH, 15*WINDOW_WIDTH/16 ) );
	DrawLine( rookImage, Point( WINDOW_WIDTH/4, 7*WINDOW_WIDTH/8 ), Point( WINDOW_WIDTH/4, WINDOW_WIDTH ) );
	DrawLine( rookImage, Point( WINDOW_WIDTH/2, 7*WINDOW_WIDTH/8 ), Point( WINDOW_WIDTH/2, WINDOW_WIDTH ) );
	DrawLine( rookImage, Point( 3*WINDOW_WIDTH/4, 7*WINDOW_WIDTH/8 ), Point( 3*WINDOW_WIDTH/4, WINDOW_WIDTH ) );

	// ---------------------------<3>显示绘制出的图像------------------------
	imshow( WINDOW_NAME1, atomImage );
	moveWindow( WINDOW_NAME1, 0, 200 );
	imshow( WINDOW_NAME2, rookImage );
	moveWindow( WINDOW_NAME2, WINDOW_WIDTH, 200 );

	waitKey( 0 );
	return(0);
}



//-------------------------------【DrawEllipse( )函数】--------------------------------
//		描述：自定义的绘制函数，实现了绘制不同角度、相同尺寸的椭圆
//-----------------------------------------------------------------------------------------
void DrawEllipse( Mat img, double angle )
{
	int thickness = 2;
	int lineType = 8;

	ellipse( img,
		Point( WINDOW_WIDTH/2, WINDOW_WIDTH/2 ),
		Size( WINDOW_WIDTH/4, WINDOW_WIDTH/16 ),
		angle,
		0,
		360,
		Scalar( 255, 129, 0 ),
		thickness,
		lineType );
}


//-----------------------------------【DrawFilledCircle( )函数】---------------------------
//		描述：自定义的绘制函数，实现了实心圆的绘制
//-----------------------------------------------------------------------------------------
void DrawFilledCircle( Mat img, Point center )
{
	int thickness = -1;
	int lineType = 8;

	circle( img,
		center,
		WINDOW_WIDTH/32,
		Scalar( 0, 0, 255 ),
		thickness,
		lineType );
}


//-----------------------------------【DrawPolygon( )函数】--------------------------
//		描述：自定义的绘制函数，实现了凹多边形的绘制
//--------------------------------------------------------------------------------------
void DrawPolygon( Mat img )
{
	int lineType = 8;

	//创建一些点
	Point rookPoints[1][20];
	rookPoints[0][0]  = Point(    WINDOW_WIDTH/4,   7*WINDOW_WIDTH/8 );
	rookPoints[0][1]  = Point(  3*WINDOW_WIDTH/4,   7*WINDOW_WIDTH/8 );
	rookPoints[0][2]  = Point(  3*WINDOW_WIDTH/4,  13*WINDOW_WIDTH/16 );
	rookPoints[0][3]  = Point( 11*WINDOW_WIDTH/16, 13*WINDOW_WIDTH/16 );
	rookPoints[0][4]  = Point( 19*WINDOW_WIDTH/32,  3*WINDOW_WIDTH/8 );
	rookPoints[0][5]  = Point(  3*WINDOW_WIDTH/4,   3*WINDOW_WIDTH/8 );
	rookPoints[0][6]  = Point(  3*WINDOW_WIDTH/4,     WINDOW_WIDTH/8 );
	rookPoints[0][7]  = Point( 26*WINDOW_WIDTH/40,    WINDOW_WIDTH/8 );
	rookPoints[0][8]  = Point( 26*WINDOW_WIDTH/40,    WINDOW_WIDTH/4 );
	rookPoints[0][9]  = Point( 22*WINDOW_WIDTH/40,    WINDOW_WIDTH/4 );
	rookPoints[0][10] = Point( 22*WINDOW_WIDTH/40,    WINDOW_WIDTH/8 );
	rookPoints[0][11] = Point( 18*WINDOW_WIDTH/40,    WINDOW_WIDTH/8 );
	rookPoints[0][12] = Point( 18*WINDOW_WIDTH/40,    WINDOW_WIDTH/4 );
	rookPoints[0][13] = Point( 14*WINDOW_WIDTH/40,    WINDOW_WIDTH/4 );
	rookPoints[0][14] = Point( 14*WINDOW_WIDTH/40,    WINDOW_WIDTH/8 );
	rookPoints[0][15] = Point(    WINDOW_WIDTH/4,     WINDOW_WIDTH/8 );
	rookPoints[0][16] = Point(    WINDOW_WIDTH/4,   3*WINDOW_WIDTH/8 );
	rookPoints[0][17] = Point( 13*WINDOW_WIDTH/32,  3*WINDOW_WIDTH/8 );
	rookPoints[0][18] = Point(  5*WINDOW_WIDTH/16, 13*WINDOW_WIDTH/16 );
	rookPoints[0][19] = Point(    WINDOW_WIDTH/4,  13*WINDOW_WIDTH/16 );

	const Point* ppt[1] = { rookPoints[0] };
	int npt[] = { 20 };

	fillPoly( img,
		ppt,
		npt,
		1,
		Scalar( 255, 255, 255 ),
		lineType );
}


//-----------------------------------【DrawLine( )函数】--------------------------
//		描述：自定义的绘制函数，实现了线的绘制
//---------------------------------------------------------------------------------
void DrawLine( Mat img, Point start, Point end )
{
	int thickness = 2;
	int lineType = 8;
	line( img,
		start,
		end,
		Scalar( 0, 0, 0 ),
		thickness,
		lineType );
}
代码结果：
![avatar](/picture/20.png)
![avatar](/picture/21.png)
![avatar](/picture/22.png)
代码：

#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <iostream>  
using namespace std;
using namespace cv;

//-----------------------------------【全局函数声明部分】-----------------------------------
//          描述：全局函数声明
//-----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div);
void ShowHelpText();



//--------------------------------------【main( )函数】---------------------------------------
//          描述：控制台应用程序的入口函数，我们的程序从这里开始执行
//-----------------------------------------------------------------------------------------------
int main()
{
	//【1】创建原始图并显示
	Mat srcImage = imread("1.jpg");
	imshow("原始图像", srcImage);

	//【2】按原始图的参数规格来创建创建效果图
	Mat dstImage;
	dstImage.create(srcImage.rows, srcImage.cols, srcImage.type());//效果图的大小、类型与原图片相同 

	ShowHelpText();

	//【3】记录起始时间
	double time0 = static_cast<double>(getTickCount());

	//【4】调用颜色空间缩减函数
	colorReduce(srcImage, dstImage, 32);

	//【5】计算运行时间并输出
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << "\t此方法运行时间为： " << time0 << "秒" << endl;  //输出运行时间

	//【6】显示效果图
	imshow("效果图", dstImage);
	waitKey(0);
}


//---------------------------------【colorReduce( )函数】---------------------------------
//          描述：使用【指针访问：C操作符[ ]】方法版的颜色空间缩减函数
//----------------------------------------------------------------------------------------------
void colorReduce(Mat& inputImage, Mat& outputImage, int div)
{
	//参数准备
	outputImage = inputImage.clone();  //拷贝实参到临时变量
	int rowNumber = outputImage.rows;  //行数
	int colNumber = outputImage.cols * outputImage.channels();  //列数 x 通道数=每一行元素的个数

	//双重循环，遍历所有的像素值
	for (int i = 0; i < rowNumber; i++)  //行循环
	{
		uchar* data = outputImage.ptr<uchar>(i);  //获取第i行的首地址
		for (int j = 0; j < colNumber; j++)   //列循环
		{
			// ---------【开始处理每个像素】-------------     
			data[j] = data[j] / div * div + div / 2;
			// ----------【处理结束】---------------------
		}  //行处理结束
	}
}

结果：
![avater](/picture/23.png/)
