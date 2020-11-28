># CH04 OpenCV 数据结构与基本绘图

>>## 4.1基础图像容器
####  4.1.1数字图像存储概述
我们可以通过各种各样的方法从现实世界获取到数字图像，如借助相机、扫描仪、计算机摄像头或磁共振成像等。通常由显示屏上看到的都是真实而漂亮的图像，但是这些图像在转化到我们的数字设备中时，记录的却是图像中的每个点的数值。
####  4.1.2Mat结构的使用
关于Mat类，首先我们要知道的是:
(1）不必再手动为其开辟空间。
(2）不必再在不需要时立即将空间释放。
这里指的是手动开辟空间并非必须，但它依旧是存在的——大多数OpenCv函数仍会手动地为输出数据开辟空间。当传递一个已经存在的Mat 对象时，开辟好的矩阵空间会被重用。也就是说，我们每次都使用大小正好的内存来完成任务。
总而言之，Mat是一个类，由两个数据部分组成:矩阵头（包含矩阵尺寸、存储方法、存储地址等信息）和一个指向存储所有像素值的矩阵（根据所选存储方法的不同，矩阵可以是不同的维数）的指针。矩阵头的尺寸是常数值，但矩阵本身的尺寸会依图像的不同而不同，通常比矩阵头的尺寸大数个数量级。因此，当在程序中传递图像并创建副本时，大的开销是由矩阵造成的，而不是信息头。OpenCV是一个图像处理库，囊括了大量的图像处理函数，为了解决问题通常要使用库中的多个函数，因此在函数中传递图像是常有的事。同时不要忘了我们正在讨论的是计算量很大的图像处理算法，因此，除非万不得已，不应该进行大图像的复制，因为这会降低程序的运行速度。
为了解决此问题，OpenCV使用了引用计数机制。其思路是让每个Mat对象有自己的信息头，但共享同一个矩阵。这通过让矩阵指针指向同一地址而实现。而拷贝构造函数则只复制信息头和矩阵指针，而不复制矩阵。

#### 4.1.3 像素值的存储方法
本节我们将讲解如何存储像素值。存储像素值需要指定颜色空间和数据类型。其中，颜色空间是指针对一个给定的颜色，如何组合颜色元素以对其编码。最简单的颜色空间要属灰度级空间，只处理黑色和白色，对它们进行组合便可以产生不同程度的灰色。
对于彩色方式则有更多种类的颜色空间，但不论哪种方式都是把颜色分成三个或者四个基元素，通过组合基元素可以产生所有的颜色。RGB颜色空间是最常用的一种颜色空间，这归功于它也是人眼内部构成颜色的方式。它的基色是红色、绿色和蓝色，有时为了表示透明颜色也会加入第四个元素alpha（A)。
颜色系统有很多，它们各有优势，具体如下。
- RGB是最常见的，这是因为人眼采用相似的工作机制，它也被显示设备所采用
- HSV和 HLS把颜色分解成色调、饱和度和亮度/明度。这是描述颜色更自然的方式，比如可以通过抛弃最后一个元素，使算法对输入图像的光照条件不敏感
- YCrCb在JPEG图像格式中广泛使用
- CIE L*a*b*是一种在感知上均匀的颜色空间，它适合用来度量两个颜色之间的距离

#### 4.1.4显式创建Mat对象的七种方法
- [方法一]使用Mat()构造函数
最常用的方法是直接使用Mat()构造函数,这种方法简单明了,示范代码如下。
Mat M(2,2，CV_ 8UC3，Scalar(0,0,255)) ;
cout<<"M="<<endl<<””<<M<<endl<<endl;
- 【方法二】在CIC++中通过构造函数进行初始化
这种方法为在C\CHH中通过构造函数进行初始化，示范代码如下。int sz[3] ={2,2,2】;
Mat L(3,sz,CV_8UC,Scalar: :all(0));
上面的例子演示了如何创建一个超过两维的矩阵:指定维数，然后传递一个指向一个数组的指针，这个数组包含每个维度的尺寸;后续的两个参数与方法一中的相同。
- 【方法三】为已存在的Ipllmage指针创建信息头
方法三是为已存在的Ipllmage指针创建信息头，示范代码如下。IplImage* img =cvLoadImage("1.jpg",1);
Mat mtx (img);//转换IplImage*-> Mat
- 【方法四】利用 Create()函数
方法四是利用Mat类中的 Create()成员函数进行Mat类的初始化操作，示范代码如下。
M.create(4, 4,Cv_8UC(2));
cout<< "M="<< endl<<" "<<M<<endl<< endl;
- 【方法五】采用Matlab式的初始化方式
方法五采用Matlab形式的初始化方式:zeros()，ones()，eyes()。使用以下方式指定尺寸和数据类型:
Mat E= Mat: :eye(4, 4,CV_64F);
cout<<"E=" <<endl<<" "<<E<<endl <<endl;
MatO=Mat: :ones(2,2,CV__32F);
cout<< "O= "<< endl<<" "<< 0<< endl << endl;
Mat Z=Mat: : zeros(3,3,CV_8UC1);
cout<<"Z="<<endl<<" "<<Z<< endl << endl;
- 【方法六】对小矩阵使用逗号分隔式初始化函数
方法六为对小矩阵使用逗号分隔式初始化函数，示范代码如下。Mat C= (Mat_<double>(3,3)<<0，-1，0，-1，5，-1,0，-1,0);cout<<"C= "<<endl<<" "<<C<< endl << endl;
- 【方法七】为已存在的对象创建新信息头
方法七为使用成员函数clone()或者copyTo()为一个已存在的Mat对象创建一个新的信息头，示范代码如下。
Mat RowClone= C.row (1).clone();
cout<<"RowClone = " << endl<< " "<< RowClone << endl << endl;

#### 基础图像容器Mat类的使用
[![Ddb5Yd.png](https://s3.ax1x.com/2020/11/26/Ddb5Yd.png)](https://imgchr.com/i/Ddb5Yd)

>>## 4.2常用数据结构与函数
#### 4.2.1点的表示:Point类
Point类数据结构表示了二维坐标系下的点,即由其图像坐标x和y指定的2D点。用法如下:
Point point;point.x = 10;point.y = 8;或者
Point point = Point(10,8);另外，在 OpenCV中有如下定义:typedef Point_<int> Point2i;typedef Point2i Point;
typedef Point_<float> Point2f;
所以，Point<int>、Point2i、Point互相等价，Point <float>、Point2f互相等价。
#### 4.2.2 颜色的表示:Scalar类
Scalar()表示具有4个元素的数组,在OpenCV中被大量用于传递像素值,如 RGB颜色值。而RGB 颜色值为三个参数，其实对于Scalar函数来说，如果用不到第四个参数，则不需要写出来;若只写三个参数，OpenCV 会认为我们就想表示三个参数。
来看个例子。如果给出以下颜色参数表达式:
Scalar( a, b,c)
那么定义的RGB颜色值:红色分量为c，绿色分量为b，蓝色分量为a。Scalar类的源头为Scalar_类，而Scalar_类是 Vec4x的一个变种，我们常用的Scalar其实就是Scalar_<double>。这就解释了为什么很多函数的参数输入可以是Mat，也可以是Scalar。
#### 4.2.3尺寸的表示: Size类
通过在代码中对Size类进行“转到定义”操作，我们可以在.......lopencvlsourceslmodulesicorelincludelopencv2lcorelcore.hpp路径下,找到Size类相关的源代码:
typedef Size_<int> Size2i;
typedefSize2i Size;
其中，Size_是个模板类，在这里Size_<int>表示其类体内部的模板所代表的类型为int。那这两句代码的意思，就是首先给已知的数据类型Size_<int>起个新名字，叫 Size2i。然后又给已知的数据类型Size2i起个新名字，叫 Size。所以，连起来就是，Size<int>、Size2i、Size这三个类型名等价。
#### 4.2.4矩形的表示:Rect类
Rect类的成员变量有x、y、width、height，分别为左上角点的坐标和矩形的宽和高。常用的成员函数有:Size()返回值为Size; area()返回矩形的面积;contains(Point)判断点是否在矩形内;inside(Rect)函数判断矩形是否在该矩形内;tlO返回左上角点坐标;br()返回右下角点坐标。值得注意的是，如果想求两个矩形的交集和并集,可以用如下格式:
Rect rect= rect1 &rect2;Rect rect = rectl lrect2;
#### 4.2.5颜色空间转换:cvtColor()函数
cvtColor()函数是 OpenCV 里的颜色空间转换函数，可以实现RGB颜色向HSV、HSI等颜色空间的转换,也可以转换为灰度图像。
原型如下:
C++: void cvtColor(InputArray src,outputArray dst,int code，intdstCn=0)
第一个参数为输入图像，第二个参数为输出图像，第三个参数为颜色空间转换的标识符（具体见表4.1)，第四个参数为目标图像的通道数，若该参数是0,表示目标图像取源图像的通道数。下面是一个调用示例:
//此句代码的 OpencV2版为:
cvtColor(srcImage, dstImage,cV_GRAY2BGR);/ /转换原始图为灰度图//此句代码的 OpencV3版为:
cvtcolor(srcImage,dstImage,COLOR_GRAY2BGR);/ /转换原始图为灰度图
#### 4.2.6其他常用的知识点
- Matx是个轻量级的Mat，必须在使用前规定好大小，比如一个2*3的 float型的Matx，可以声明为 Matx23f。
- Vec是Matx的一个派生类,是一个一维的Matx,跟vector很相似。在OpenCV源码中有如下定义。
template<typename_Tp,int n> class Vec : public Matx<_Tp，n,1> {... ;typedef Vec<uchar,2> Vec2b;
- Range类其实就是为了使 OpenCV的使用更像MATLAB而产生的。比如Range::all)其实就是MATLAB里的符号。而 Range(a, b)其实就是MATLAB中的a: b，注意这里的a和b都应为整型。
- OpenCV中防止内存溢出的函数有alignPtr、alignSize、allocate、deallocate、fastMalloc、fastFree等。
- <math.h>里的一些函数使用起来很方便，有计算向量角度的函数fastAtan2、计算立方根的函数cubeRoot、向上取整函数cvCeil、向下取整函数cvFloor.四舍五入函数cvRound等。还有一些类似MATLAB里面的函数,比如cvIsInf判断自变量是否无穷大，cvlsNaN 判断自变量是否不是一个数。
- 显示文字相关的函数有getTextSize、cvInitFont、 putText。
- 作图相关的函数有circle、clipLine、ellipse、ellipse2Poly、line、rectangle、polylines、类 LineIterator .
- 填充相关的函数有fillConvexPoly、fillPoly。
- OpenCV 中 RNG()函数的作用为初始化随机数状态的生成器。
其他数据结构相关的知识由于使用较少在此不再过多说明，在需要用到的时候读者可以配合OpenCV 文档，并查阅OpenCV的源码进行学习。


>>## 4.3基本图形的绘制
[![DdbO0S.png](https://s3.ax1x.com/2020/11/26/DdbO0S.png)](https://imgchr.com/i/DdbO0S)


># CH05 core组建进阶

>>## 5.1访问图像中的像素
#### 用指针访问像素
[![DdLyqK.png](https://s3.ax1x.com/2020/11/26/DdLyqK.png)](https://imgchr.com/i/DdLyqK)
#### 迭代器访问像素
[![DdL2Ie.png](https://s3.ax1x.com/2020/11/26/DdL2Ie.png)](https://imgchr.com/i/DdL2Ie)
#### 用动态地址计算配合at访问像素
[![DdLWPH.png](https://s3.ax1x.com/2020/11/26/DdLWPH.png)](https://imgchr.com/i/DdLWPH)
#### 遍历图像像素的14种方法
[![DdL4xI.png](https://s3.ax1x.com/2020/11/26/DdL4xI.png)](https://imgchr.com/i/DdL4xI)
>>## 5.2ROI区域图像叠加和图像组合
在图像处理领域，我们常常需要设置感兴趣区域（ROI，region of interest)，来专注或者简化工作过程。也就是从图像中选择的一个图像区域，这个区域是图像分析所关注的重点。我们圈定这个区域，以便进行进一步处理。而且，使用ROI指定想读入的目标，可以减少处理时间，增加精度，给图像处理来带不小的便利。
定义ROI区域有两种方法:第一种是使用表示矩形区域的Rect。它指定矩形的左上角坐标(构造函数的前两个参数）和矩形的长宽(构造函数的后两个参数)以定义一个矩形区域。
>>## 5.3分离颜色通道，多通道组合
#### 初级图像混合
[![DdLIMt.png](https://s3.ax1x.com/2020/11/26/DdLIMt.png)](https://imgchr.com/i/DdLIMt)
>>## 5.4图像对比度，亮度值调整
[![DdLTqf.png](https://s3.ax1x.com/2020/11/26/DdLTqf.png)](https://imgchr.com/i/DdLTqf)
>>## 5.5离散傅里叶变换
[![DdLqIg.png](https://s3.ax1x.com/2020/11/26/DdLqIg.png)](https://imgchr.com/i/DdLqIg)
>>## 5.6输入输出XML和YAML文件
#### XML和YAML文件简介
本节我们将一起认识XML和 YAML这两种文件类型。
所谓XML，即 eXtensible Markup Language，翻译成中文为“可扩展标识语言”。首先，XML是一种元标记语言。所谓“元标记”，就是开发者可以根据自身需要定义自己的标记，比如可以定义标记<book>、<name>。任何满足XML命名规则的名称都可以标记，这就向不同的应用程序打开了的大门。此外，XML是一种语义/结构化语言，它描述了文档的结构和语义。
YAML是“YAML Ain't a Markup Language”(译为“YAML不是一种置标语言”)的递回缩写。在开发的这种语言时，YAML的原意是:“Yet Another MarkupLanguage”（仍是一种置标语言)，但为了强调这种语言以数据为中心,而不是以置标语言为重点，而用返璞词进行重新命名。YAML是一个可读性高，用来表达资料序列的格式。它参考了其他多种语言，包括:XML、C语言、Python、Perl,以及电子邮件格式RFC2822。
#### FileStorage类操作文件的使用引导
XML和YAML 是使用非常广泛的文件格式，可以利用XML或者YAML格式的文件存储和还原各式各样的数据结构。当然，它们还可以存储和载入任意复杂的数据结构，其中就包括了OpenCV相关周边的数据结构，以及各种原始数据类型，如整数和浮点数字和文本字符串。
我们一般使用如下过程来写入或者读取数据到XML 或 YAML文件中。
- 实例化一个FileStorage类的对象，用默认带参数的构造函数完成初始化，或者用FileStorage::open()成员函数辅助初始化。
- 使用流操作符<<进行文件写入操作，或者>>进行文件读取操作，类似C++中的文件输入输出流。
- 使用FileStorage:.release()函数析构掉FileStorage类对象，同时关闭文件。下面分别对这三个步骤进行实例讲解。

  + 【第一步】XML、YAML文件的打开
1)准备文件写操作
FileStorage是 OpenCV 中 XML和 YAML文件的存储类，封装了所有相关的信息。它是OpenCV从文件中读数据或向文件中写数据时必须要使用的一个类。
此类的构造函数为 FileStorage::FileStorage，有两个重载，如下。
C十+: Filestorage : :Filestorage ()
C++:FileStorage::FileStorage (const string& source, int flags,conststring& encoding-string ()
构造函数在实际使用中，方法一般有两种。
1)对于第二种带参数的构造函数，进行写操作范例如下。FileStorage fs( "abc.xml",Filestorage: :WRITE);
2）对于第一种不带参数的构造函数，可以使用其成员函数FileStorage:.open进行数据的写操作，范例如下。
FileStorage fs;
fs.open ("abc.xml",FileStorage: : WRITE);
(2）准备文件读操作
上面讲到的都是以 FileStorage:: WRITE为标识符的写操作，而读操作，采用FileStorage::READ标识符即可,相关示例代码如下。
1）第一种方式
Filestorage fs ("abc.xml",Filestorage: : READ);2）第二种方式
FileStorage fs;
fs.open("abc. xml",FileStorage:: READ);
另外需要注意的是，上面的这些操作示例是对XML文件为例子作演示的，而对 YAML文件，操作方法是类似的，就是将XML文件换为 YAML文件即可。
 + 【第二步】进行文件读写操作
(1）文本和数字的输入和输出
 + 【第三步】vector (arrays）和maps的输入和输出
 + 【第四步】文件关闭

># 章节小结
通过45章的学习，对OpenCV有了更深入的体验，知识的难度也是越来越大了。