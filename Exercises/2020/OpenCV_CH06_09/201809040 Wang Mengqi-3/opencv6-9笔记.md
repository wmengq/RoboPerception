># CH06 图像处理

>>## 6.1线性滤波:方框滤波、均值滤波、高斯滤波
#### 平滑处理
平滑处理（smoothing）也称模糊处理（bluring)，是一种简单且使用频率很高的图像处理方法。平滑处理的用途有很多,最常见的是用来减少图像上的噪点或者失真。在涉及到降低图像分辨率时,平滑处理是非常好用的方法。
#### 图像滤波与滤波器
图像滤波,指在尽量保留图像细节特征的条件下对目标图像的噪声进行抑制,是图像预处理中不可缺少的操作，其处理效果的好坏将直接影响到后续图像处理和分析的有效性和可靠性。
消除图像中的噪声成分叫作图像的平滑化或滤波操作。信号或图像的能量大部分集中在幅度谱的低频和中频段，而在较高频段,有用的信息经常被噪声淹没。因此一个能降低高频成分幅度的波器就能够减弱噪声的影响。
图像滤波的目的有两个:一个是抽出对象的特征作为图像识别的特征模式;另一个是为适应图像处理的要求,消除图像数字化时所混入的噪声。
而对滤波处理的要求也有两条:一是不能损坏图像的轮廓及边缘等重要信息;二是使图像清晰视觉效果好。
平滑滤波是低频增强的空间域滤波技术。它的目的有两类:一类是模糊;另一类是消除噪音。
空间域的平滑滤波一般采用简单平均法进行，就是求邻近像元点的平均亮度值。邻域的大小与平滑的效果直接相关,邻域越大平滑的效果越好,但邻域过大,平滑也会使边缘信息损失的越大，从而使输出的图像变得模糊，因此需合理选择邻域的大小。

#### 方框滤波(box Filter)
方框滤波（box Filter）被封装在一个名为boxblur的函数中，即 boxblur函数的作用是使用方框滤波器(box filter）来模糊一张图片，从 src输入，从dst输出。

#### 均值滤波
均值滤波，是最简单的一种滤波操作,输出图像的每一个像素是核窗口内输入图像对应像素的平均值(所有像素加权系数相等)，其实说白了它就是归一化后的方框滤波。我们在下文进行源码剖析时会发现，blur 函数内部中其实就是调用了一 下boxFilter.

#### 高斯滤波
高斯滤波是一种线性平滑滤波，可以消除高斯噪声，广泛应用于图像处理的减噪过程。通俗地讲，高斯滤波就是对整幅图像进行加权平均的过程，每一个像素点的值，都由其本身和邻域内的其他像素值经过加权平均后得到。高斯滤波的具体操作是:用一个模板（或称卷积、掩模)扫描图像中的每一个像素，用模板确定的邻域内像素的加权平均灰度值去替代模板中心像素点的值。

#### 线性滤波综合示例
[![D6y0a9.png](https://s3.ax1x.com/2020/11/29/D6y0a9.png)](https://imgchr.com/i/D6y0a9)

>>## 6.2非线性波:中值滤波、双边滤波
正如我们在6.1节中讲到的,线性滤波可以实现很多种不同的图像变换。而非线性滤波，如中值滤波器和双边滤波器，有时可以达到更好的实现效果。

#### 中值滤波
中值滤波(Median filter)是一种典型的非线性滤波技术，基本思想是用像素点邻域灰度值的中值来代替该像素点的灰度值,该方法在去除脉冲噪声、椒盐噪声的同时又能保留图像的边缘细节。
中值滤波是基于排序统计理论的一种能有效抑制噪声的非线性信号处理技
术，其基本原理是把数字图像或数字序列中一点的值用该点的一个邻域中各点值的中值代替,让周围的像素值接近真实值，从而消除孤立的噪声点。这对于斑点噪声（speckle noise）和椒盐噪声(salt-and-pepper noise）来说尤其有用,因为它不依赖于邻域内那些与典型值差别很大的值。中值滤波器在处理连续图像窗函数时与线性滤波器的工作方式类似,但滤波过程却不再是加权运算。

#### 双边滤波
双边滤波（Bilateral filter)是一种非线性的滤波方法，是结合图像的空间邻近度和像素值相似度的一种折中处理，同时考虑空域信息和灰度相似性，达到保边去噪的目的,具有简单、非迭代、局部的特点。
双边滤波器的好处是可以做边缘保存（edge preserving)。以往常用维纳滤波或者高斯滤波去降噪，但二者都会较明显地模糊边缘，对于高频细节的保护效果并不明显。双边滤波器顾名思义，比高斯滤波多了一个高斯方差sigma-d,它是基于空间分布的高斯滤波函数，所以在边缘附近,离得较远的像素不会对边缘上的像素值影响太多，这样就保证了边缘附近像素值的保存。但是，由于保存了过多的高频信息，对于彩色图像里的高频噪声，双边滤波器不能够干净地滤掉,只能对于低频信息进行较好地滤波。

#### 图像滤波综合示例
[![D6cSTH.png](https://s3.ax1x.com/2020/11/29/D6cSTH.png)](https://imgchr.com/i/D6cSTH)

>>## 6.3形态学滤波(1):腐蚀与膨胀

#### 形态学概述
形态学(morphology)一词通常表示生物学的一个分支，该分支主要研究动植物的形态和结构。而我们图像处理中的形态学，往往指的是数学形态学。下面一起来了解数学形态学的概念。
数学形态学(Mathematical morphology)是一门建立在格论和拓扑学基础之上的图像分析学科，是数学形态学图像处理的基本理论。其基本的运算包括:二值腐蚀和膨胀、二值开闭运算、骨架抽取、极限腐蚀、击中击不中变换、形态学梯度、Top-hat变换、颗粒分析、流域变换、灰值腐蚀和膨胀、灰值开闭运算、灰值形态学梯度等。
膨胀与腐蚀能实现多种多样的功能,主要如下。
- 消除噪声;
- 分割( isolate）出独立的图像元素,在图像中连接(join）相邻的元素寻找图像中的明显的极大值区域或极小值区域;
- 求出图像的梯度。

#### 膨胀
膨胀（dilate）就是求局部最大值的操作。从数学角度来说，膨胀或者腐蚀操作就是将图像（或图像的一部分区域，称之为A）与核（称之为B）进行卷积。
核可以是任何形状和大小，它拥有一个单独定义出来的参考点，我们称其为锚点( anchorpoint)。多数情况下，核是一个小的，中间带有参考点和实心正方形或者圆盘。其实，可以把核视为模板或者掩码。

#### 腐蚀
大家应该知道，膨胀和腐蚀( erode）是相反的一对操作，所以腐蚀就是求局部最小值的操作。

#### 图像腐蚀与膨胀综合示例
[![D6cO4s.png](https://s3.ax1x.com/2020/11/29/D6cO4s.png)](https://imgchr.com/i/D6cO4s)
[![D6gnKK.png](https://s3.ax1x.com/2020/11/29/D6gnKK.png)](https://imgchr.com/i/D6gnKK)

>>## 6.4形态学滤波(2):开运算、闭运算、形态学梯度、顶帽、黑帽

#### 开运算
开运算（Opening Operation)，其实就是先腐蚀后膨胀的过程。其数学表达式如下:
dst-open (sre,element) =dilate(erode(sre,element))

#### 闭运算
先膨胀后腐蚀的过程称为闭运算（Closing Operation)，其数学表达式如下:dst=clese(src.element)- erode (dilate(sre,element))

#### 形态学梯度
形态学梯度（Morphological Gradient）是膨胀图与腐蚀图之差，数学表达式如下:
dst-morph-grad(src,element)- dilate(src,element)- erode (src,element)

#### 顶帽
顶帽运算(Top Hat）又常常被译为”礼帽“运算,是原图像与上文刚刚介绍的“开运算”的结果图之差,数学表达式如下:
dst-tophat fsrc,element)=srC-open(erc,elementY
因为开运算带来的结果是放大了裂缝或者局部低亮度的区域。因此，从原图中减去开运算后的图,得到的效果图突出了比原图轮廓周围的区域更明亮的区域,且这一操作与选择的核的大小相关。

#### 黑帕
黑帽(Black Hat）运算是团运算的结果图与原图像之差。数学表达式为:dst-blackhat (sre,element-close(src,element)- sSrc

#### 形态学图像处理综合示例
[![D6RKtH.png](https://s3.ax1x.com/2020/11/29/D6RKtH.png)](https://imgchr.com/i/D6RKtH)

>>##  6.5漫水填充

#### 漫水填充的定义
漫水填充法是一种用特定的颜色填充连通区域，通过设置可连通像素的上下限以及连通方式来达到不同的填充效果的方法。漫水填充经常被用来标记或分离图像的一部分，以便对其进行进一步处理或分析，也可以用来从输入图像获取掩码区域，掩码会加速处理过程，或只处理掩码指定的像素点，操作的结果总是某个连续的区域。

#### 漫水填充法的基本思想
所谓漫水填充，简单来说，就是自动选中了和种子点相连的区域,接着将该区域替换成指定的颜色，这是个非常有用的功能,经常用来标记或者分离图像的一部分进行处理或分析。漫水填充也可以用来从输入图像获取掩码区域，掩码会加速处理过程，或者只处理掩码指定的像素点。
以此填充算法为基础，类似 PhotoShop 的魔术棒选择工具就很容易实现了。漫水填充（FloodFill）是查找和种子点连通的颜色相同的点，魔术棒选择工具则是查找和种子点连通的颜色相近的点，把和初始种子像素颜色相近的点压进栈做为新种子。

#### 实现漫水填充算法:floodFill函数
在OpenCV中，漫水填充算法由 floodFill函数实现，其作用是用我们指定的颜色从种子点开始填充一个连接域。连通性由像素值的接近程度来衡量。
- (1)第一个参数，InputOutputArray类型的image,输入/输出1通道或3通道,8位或浮点图像，具体参数由之后的参数指明。
- (2)第二个参数,InputOutputArray类型的mask,这是第二个版本的 floodFill独享的参数,表示操作掩模。它应该为单通道,8位,长和宽上都比输入图像 image大两个像素点的图像。第二个版本的floodFill需要使用以及更新掩膜，所以对于这个 mask参数，我们一定要将其准备好并填在此处。需要注意的是，漫水填充不会填充掩膜 mask的非零像素区域。例如，一个边缘检测算子的输出可以用来作为掩膜，以防止填充到边缘。同样的，也可以在多次的函数调用中使用同一个掩膜，以保证填充的区域不会重叠。另外需要注意的是，掩膜 mask会比需填充的图像大,所以 mask中与输入图像(x,y)像素点相对应的点的坐标为(x+1,y+1)。
- (3)第三个参数，Point类型的seedPoint，漫水填充算法的起始点。
- (4）第四个参数，Scalar类型的newVal，像素点被染色的值，即在重绘区域像素的新值。
- (5）第五个参数，Rect*类型的rect，有默认值0，一个可选的参数，用于设置floodFill函数将要重绘区域的最小边界矩形区域。
- (6)第六个参数，Scalar类型的 loDiff，有默认值 Scalar()，表示当前观察像素值与其部件邻域像素值或者待加入该部件的种子像素之间的亮度或颜色之负差Clower brightness/color difference）的最大值。
- (7）第七个参数，Scalar类型的upDiff，有默认值Scalar()，表示当前观察像素值与其部件邻域像素值或者待加入该部件的种子像素之间的亮度或颜色之正差(lower brightness/color difference)的最大值。
- (8）第八个参数，int类型的flags.操作标志符，此参数包含三个部分:
1.低八位(第0~7位)用于控制算法的连通性,可取4(4为默认值)或者8。如果设为4，表示填充算法只考虑当前像素水平方向和垂直方向的相邻点;如果设为8，除上述相邻点外，还会包含对角线方向的相邻点。
2.高八位部分(16~23位）可以为0或者如下两种选项标识符的组合。
 FLOODFILL_FIXED_RANGE:如果设置为这个标识符，就会考虑当前像素与种子像素之间的差,否则就考虑当前像素与其相邻像素的差。也就是说,这个范围是浮动的。
 FLOODFILL_MASK_ ONLY -如果设置为这个标识符,函数不会去填充改变原始图像(也就是忽略第三个参数newVal),而是去填充掩模图像(mask)。这个标识符只对第二个版本的floodFill有用，因第一个版本里面压根就没有mask参数。
3.中间八位部分,上面关于高八位FLOODFILL_MASK_ONLY标识符中已经说得很明显,需要输入符合要求的掩码。Floodfill的 flags参数的中间八位的值就是用于指定填充掩码图像的值的。但如果flags中间八位的值为0,则掩码会用1来填充。

#### 漫水填充综合示例
[![D6WvoF.png](https://s3.ax1x.com/2020/11/29/D6WvoF.png)](https://imgchr.com/i/D6WvoF)

>>## 图像金字塔与图片尺寸缩放

#### 关于图像金字塔
图像金字塔是图像中多尺度表达的一种，最主要用于图像的分割，是一种以多分辨率来解释图像的有效但概念简单的结构。
图像金字塔最初用于机器视觉和图像压缩，一幅图像的金字塔是一系列以金字塔形状排列的，分辨率逐步降低且来源于同一张原始图的图像集合。其通过梯次向下采样获得，直到达到某个终止条件才停止采样。

#### 高斯金字塔
高斯金字塔是通过高斯平滑和亚采样获得一些列下采样图像，也就是说第K层高斯金字塔通过平滑、亚采样就可以获得K+1层高斯图像。高斯金字塔包含了一系列低通滤波器,其截止频率从上一层到下一层以因子2逐渐增加,所以高斯金字塔可以跨越很大的频率范围。金字塔的图像如图6.57所示。

#### 拉普拉斯金字塔
下式是拉普拉斯金字塔第i层的数学定义:
L=G-UP(G1)R g5x5
式中的G表示第i层的图像。而 UPO操作是将源图像中位置为(x,y)的像素映射到目标图像的(2x+1,2y+1)位置,即在进行向上取样。符号8表示卷积，gsxs为5x5的高斯内核。

#### 图像金字塔和rezize综合示例
放大
[![D6f3ef.png](https://s3.ax1x.com/2020/11/29/D6f3ef.png)](https://imgchr.com/i/D6f3ef)
缩小
[![D6ftYQ.png](https://s3.ax1x.com/2020/11/29/D6ftYQ.png)](https://imgchr.com/i/D6ftYQ)

>>## 阈值化
在对各种图形进行处理操作的过程中，我们常常需要对图像中的像素做出取舍与决策,直接剔除一些低于或者高于一定值的像素。
阙值可以被视作最简单的图像分割方法。比如，从一副图像中利用阀值分割出我们需要的物体部分(当然这里的物体可以是一部分或者整体)。这样的图像分割方法基于图像中物体与背景之间的灰度差异，而且此分割属于像素级的分割。为了从一副图像中提取出我们需要的部分，应该用图像中的每一个像素点的灰度值与选取的阙值进行比较,并作出相应的判断。注意:阈值的选取依赖于具体的问题。即物体在不同的图像中有可能会有不同的灰度值。

#### 固定阈值操作:Threshold(函数
函数Threshold(对单通道数组应用固定住值操作。该函数的典型应用是对灰度图像进行阙值操作得到二值图像，(compare()函数也可以达到此目的)或者是去掉噪声，例如过滤很小或很大象素值的图像点。
C++: double threshold (InputArray src, 0utputArray dst,double thresh,double maxval, int type)

#### 自适应住值操作:adaptiveThreshold(函数
adaptiveThresholdO函数的作用是对矩阵采用自适应阈值操作,支持就地操作。函数原型如下。
C++:void adaptiveThreshold(InputArray src,outputArray dst, doublemaxValue,int adaptiveMethod,int thresholdType,int blockSize,doublec)

#### 基本阈值操作
[![D6fcY4.png](https://s3.ax1x.com/2020/11/29/D6fcY4.png)](https://imgchr.com/i/D6fcY4)


># CH07 图像变换

>>## 7.1边缘检测
#### 边缘检测的一般步骤
【第一步】滤波
边缘检测的算法主要是基于图像强度的一阶和二阶导数，但导数通常对噪声很敏感，因此必须采用滤波器来改善与噪声有关的边缘检测器的性能。常见的滤波方法主要有高斯滤波，即采用离散化的高斯函数产生一组归一化的高斯核，然后基于高斯核函数对图像灰度矩阵的每一点进行加权求和。
【第二步】增强
增强边缘的基础是确定图像各点邻域强度的变化值。增强算法可以将图像灰度点邻域强度值有显著变化的点凸显出来。在具体编程实现时,可通过计算梯度幅值来确定。
【第三步】检测
经过增强的图像，往往邻域中有很多点的梯度值比较大，而在特定的应用中，这些点并不是要找的边缘点，所以应该采用某种方法来对这些点进行取舍。实际工程中,常用的方法是通过阙值化方法来检测。
另外，需要注意，下文中讲到的 Laplacian 算子、sobel算子和 Scharr 算子都是带方向的,所以，示例中我们分别写了X方向、Y方向和最终合成的的效果图。

#### canny 算子
canny算子简介
Canny边缘检测算子是John F.Canny 于1986年开发出来的一个多级边缘检测算法。更为重要的是，Canny 创立了边缘检测计算理论(Computational thcoryofedge detection)，解释了这项技术是如何工作的。Canny边缘检测算法以Canny的名字命名,被很多人推崇为当今最优的边缘检测的算法。
其中，Canny的目标是找到一个最优的边缘检测算法,让我们看一下最优边缘检测的三个主要评价标准。
低错误率:标识出尽可能多的实际边缘，同时尽可能地减少噪声产生的误报。
·高定位性:标识出的边缘要与图像中的实际边缘尽可能接近。
最小响应:图像中的边缘只能标识一次，并且可能存在的图像噪声不应标
识为边缘。
为了满足这些要求，Canny使用了变分法，这是一种寻找满足特定功能的函数的方法。最优检测用4个指数函数项的和表示,但是它非常近似于高斯函数的一阶导数。

#### sobel算子
sobel算子的基本概念
Sobel算子是一个主要用于边缘检测的离散微分算子（(discrete differentiationoperator)。它结合了高斯平滑和微分求导，用来计算图像灰度函数的近似梯度。在图像的任何一点使用此算子，都将会产生对应的梯度矢量或是其法矢量。

#### Laplacian 算子
 Laplacian算子简介
Laplacian算子是n维欧几里德空间中的一个二阶微分算子,定义为梯度grad的散度div。因此如果f是二阶可微的实函数，则f的拉普拉斯算子定义如下。
(1)f的拉普拉斯算子也是笛卡儿坐标系xi中的所有非混合二阶偏导数求和。(2）作为一个二阶微分算子，拉普拉斯算子把C函数映射到C函数。对于k≥2，表达式(1)(或(2))定义了一个算子A:C(R)一C(R);或更一般地，对于任何开集2,定义了一个算子A:C(Q2)→C(Q)。
根据图像处理的原理可知，二阶导数可以用来进行检测边缘。因为图像是“二维”，需要在两个方向进行求导。使用Laplacian算子将会使求导过程变得简单。
Laplacian算子的定义:
Laplace(f-ax2oy
a2f of

#### scharr滤波器
1.计算图像差分:Scharr()函数
使用Scharr 滤波器运算符计算x或y方向的图像差分。其实它的参数变量和Sobel基本上是一样的,除了没有ksize核的大小。
(1)第一个参数，InputArray类型的src，为输入图像，填Mat类型即可。(2)第二个参数，OutputArray类型的 dst，即目标图像，函数的输出参数，需要和源图片有一样的尺寸和类型。
(3)第三个参数，int类型的 ddepth，输出图像的深度,支持如下src.depth()和 ddepth的组合:
- 若src.depth(-CV 8U,取 ddepth=-1/CV_16S/CV_32F/CV_64F
- 若src.depth(=CV_16U/CV_16S,取 ddepth =-1/CV_32F/CV_64F
- 若 src.depthQ -CV_32F,取 ddepth =-1/CV_32F/CV_64F·若 src.depthQ -CV_64F,取ddepth = -1/CV 64F
(4)第四个参数，int类型dx,x方向上的差分阶数。(5)第五个参数,int类型dy,y方向上的差分阶数。
(6）第六个参数，double类型的scale，计算导数值时可选的缩放因子，默认值是1，表示默认情况下是没有应用缩放的。我们可以在文档中查阅gctDerivKernels的相关介绍,来得到这个参数的更多信息。
(7）第七个参数，double类型的 delta，表示在结果存入目标图(第二个参数dst）之前可选的delta值,有默认值0。
(8）第八个参数，int类型的 borderType，边界模式，默认值为 BORDERDEFAULT。这个参数可以在官方文档中 borderInterpolate处得到更详细的信息。

#### 边缘检测综合示例
[![Dc6avV.png](https://s3.ax1x.com/2020/11/29/Dc6avV.png)](https://imgchr.com/i/Dc6avV)

>>## 7.2霍夫变换
#### 霍夫变换概述
霍夫变换（Hough Transform）是图像处理中的一种特征提取技术,该过程在一个参数空间中通过计算累计结果的局部最大值得到一个符合该特定形状的集合作为霍夫变换结果。霍夫变换于1962年由 PaulHough首次提出，最初的 Hough变换是设计用来检测直线和曲线的。起初的方法要求知道物体边界线的解析方程,但不需要有关区域位置的先验知识。这种方法的一个突出优点是分割结果的Robustness，即对数据的不完全或噪声不是非常敏感。然而，要获得描述边界的解析表达常常是不可能的。后于1972年由Richard Duda & Peter Hart推广使用,经典霍夫变换用来检测图像中的直线,后来霍夫变换扩展到任意形状物体的识别,多为圆和缭圆。霍夫变换运用两个坐标空间之间的变换将在一个空间中具有相同形状的曲线或直线映射到另一个坐标空间的一个点上形成峰值，从而把检测任意形状的问题转化为统计峰值问题。

#### OpenCV中的霍夫线变换
OpenCV支持三种不同的霍夫线变换，它们分别是:标准霍夫变换(StandardHough Transform,SHT)、多尺度霍夫变换(Multi-Scale Hough Transform,MSHT)和累计概率霍夫变换（Progressive Probabilistic Hough Transform,PPHT)。
其中，多尺度霍夫变换（MSHT)为经典霍夫变换(SHT)在多尺度下的一个变种。而累计概率霍夫变换（PPHT)算法是标准霍夫变换(SHT)算法的一个改进，它在一定的范围内进行霍夫变换，计算单独线段的方向以及范围，从而减少计算量，缩短计算时间。之所以称PPHT 为“概率”的，是因为并不将累加器平面内的所有可能的点累加，而只是累加其中的一部分，该想法是如果峰值如果足够高,只用一小部分时间去寻找它就够了。按照猜想，可以实质性地减少计算时间。
总结一下，OpenCV中的霍夫线变换有如下三种:
- 标准霍夫变换(StandardHough Transform, SHT),由HoughLines 函数调用。
- 多尺度霍夫变换（Multi-ScaleHough Transform,MSHT)，由 HoughLines函数调用。
- 累计概率霍夫变换（ProgressiveProbabilistic Hough Transform，PPHT)，由HoughLinesP函数调用。

#### 标准霍夫变换:HoughLines()函数
此函数可以找出采用标准霍夫变换的二值图像线条。在 OpenCV中，我们可以用其来调用标准霍夫变换SHT和多尺度霍夫变换 MSHT的 OpenCV内建算法。
C++:void HoughLines(InputArray image,OutputArray lines,double rho,double theta, int threshold,double srn=0，double stn=0,
- 第一个参数，InputArray类型的image，输入图像，即源图像。需为8位的单通道二进制图像,可以将任意的源图载入进来，并由函数修改成此格式后，再填在这里。
- 第二个参数，InputArray类型的lines,经过调用HoughLines函数后储存了霍夫线变换检测到线条的输出矢量。每一条线由具有两个元素的矢量（p,0)表示，其中，P是离坐标原点（0,0）（也就是图像的左上角)的距离，0是弧度线条旋转角度（0度表示垂直线,元t/2度表示水平线)。
- 第三个参数，double类型的rho，以像素为单位的距离精度。另一种表述方式是直线搜索时的进步尺寸的单位半径。(Latex 中/rho即表示P）
- 第四个参数，double类型的theta，以弧度为单位的角度精度。另一种表述方式是直线搜索时的进步尺寸的单位角度。
- 第五个参数，int类型的threshold，累加平面的阅值参数，即识别某部分为图中的一条直线时它在累加平面中必须达到的值。大于阈值 threshold 的线段才可以被检测通过并返回到结果中。
- 第六个参数,double类型的srm，有默认值0。对于多尺度的霍夫变换，这是第三个参数进步尺寸rho的除数距离。粗略的累加器进步尺寸直接是第三个参数rho，而精确的累加器进步尺寸为rho/srn。
- 第七个参数，double类型的 stn，有默认值0，对于多尺度霍夫变换，srn表示第四个参数进步尺寸的单位角度theta的除数距离。且如果srm 和 stn同时为0，就表示使用经典的霍夫变换。否则，这两个参数应该都为正数。

#### 累计概率霍夫变换:HoughLinesPO的数
此函数在HoughLines的基础上，在末尾加了一个代表 Probabilistic（概率）的P，表明它可以采用累计概率霍夫变换（PPHT)来找出二值图像中的直线。
C++:void HoughLinesP(InputArray image,OutputArray lines,double rho,double theta, int threshold, double minLinelength=0, double
maxLineGap-0)
- 第一个参数，InputArray类型的image，输入图像，即源图像。需为8位的单通道二进制图像,可以将任意的源图载入进来后由函数修改成此格式后，再填在这里。
- 第二个参数，InputArray类型的 lines，经过调用HoughLinesP函数后存储了检测到的线条的输出矢量,每一条线由具有4个元素的矢量(x_1.y_1,x_2,y_2)表示,其中，(x_1,y_D)和(x_2,y_2)是是每个检测到的线段的结束点。
- 第三个参数，double类型的rho,以像素为单位的距离精度。另一种表述方式是直线搜索时的进步尺寸的单位半径。
- 第四个参数，double类型的 theta，以弧度为单位的角度精度。另一种表述方式是直线搜索时的进步尺寸的单位角度。
- 第五个参数，int类型的threshold，累加平面的阙值参数，即识别某部分为图中的一条直线时它在累加平面中必须达到的值。大于阅值 threshold的线段才可以被检测通过并返回到结果中。
- 第六个参数，double类型的minLineLength，有默认值0，表示最低线段的长度,比这个设定参数短的线段就不能被显现出来。
- 第七个参数，double类型的maxLineGap，有默认值0，允许将同一行点与点之间连接起来的最大的距离。

#### 霍夫圆变换
霍夫圆变换的基本原理和上面讲的霍夫线变化大体上是很类似的,只是点对应的二维极径极角空间被三维的圆心点x,y和半径r空间取代。说“大体上类似”的原因是，如果完全用相同的方法的话，累加平面会被三维的累加容器所代替一一在这三维中，一维是x,一维是y,另外一维是圆的半径r。这就意味着需要大量的内存而且执行效率会很低,速度会很慢。

#### 霍夫梯度法的原理
霍夫梯度法的原理是这样的:
(1)首先对图像应用边缘检测，比如用canny边缘检测。
(2）然后,对边缘图像中的每一个非零点，考虑其局部梯度,即用 Sobel(函数计算x和y方向的Sobel一阶导数得到梯度。
(3）利用得到的梯度，由斜率指定的直线上的每一个点都在累加器中被累加,这里的斜率是从一个指定的最小值到指定的最大值的距离。
(4）同时,标记边缘图像中每一个非0像素的位置。
(5）然后从二维累加器中这些点中选择候选的中心，这些中心都大于给定颍值并且大于其所有近邻。这些候选的中心按照累加值降序排列，以便于最支持像素的中心首先出现。
(6)接下来对每一个中心,考虑所有的非0像素。
(7)这些像素按照其与中心的距离排序。从到最大半径的最小距离算起，选择非0像素最支持的一条半径。
(8）如果一个中心收到边缘图像非0像素最充分的支持，并且到前期被选择的中心有足够的距离,那么它就会被保留下来。
这个实现可以使算法执行起来更高效，或许更加重要的是，能够帮助解决三维累加器中会产生许多噪声并且使得结果不稳定的稀疏分布问题。
人无完人，金无足赤。同样,这个算法也并不是十全十美的，还有许多需要指出的缺点。

#### 霍夫梯度法的缺点
(1）在霍夫梯度法中，我们使用 Sobel导数来计算局部梯度,那么随之而来的假设是，它可以视作等同于一条局部切线,这并不是一个数值稳定的做法。在大多数情况下，这样做会得到正确的结果，但或许会在输出中产生一些噪声。
(2）在边缘图像中的整个非0像素集被看做每个中心的候选部分。因此，如果把累加器的阙值设置偏低，算法将要消耗比较长的时间。此外，因为每一个中心只选择一个圆，如果有同心圆，就只能选择其中的一个。
(3）因为中心是按照其关联的累加器值的升序排列的，并且如果新的中心过于接近之前已经接受的中心的话，就不会被保留下来。且当有许多同心圆或者是近似的同心圆时，霍夫梯度法的倾向是保留最大的一个圆。可以说这是一种比较极端的做法，因为在这里默认Sobel导数会产生噪声,若是对于无穷分辨率的平滑图像而言的话,这才是必须的。

#### 霍夫线变综合示例
[![DcILkV.png](https://s3.ax1x.com/2020/11/29/DcILkV.png)](https://imgchr.com/i/DcILkV)

>>## 7.3重影射
#### 重映射的概念
重映射,就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。为了完成映射过程，需要获得一些插值为非整数像素的坐标，因为源图像与目标图像的像素坐标不是一一对应的。一般情况下,我们通过重映射来表达每个像素的位置(x,y)，像这样:
g(x.y)-f(h(x.y))
在这里,gO是目标图像，()是源图像，而h(x,y)是作用于(x.y)的映射方法函数。来看个例子。若有一幅图像I，对其按照下面的条件作重映射:
h(x.y)-(1.cols - x. y)

#### 实现重映射: remapO函数
remap()函数会根据指定的映射形式,将源图像进行重映射几何变换，基于的公式如下:
dst(x,y)-src(nap,(x.y),map(x.y))
需要注意,此函数不支持就地（in-place）操作。看看其原型和参数。C++: void remap (InputArray src,outputArraydst,InputArray mapl,InputArray map2, int interpolation,intborderMode=BORDER_CONSTANT,const scalar& bordervalue=Scalar)
- 第一个参数，InputArray类型的src，输入图像,即源图像，填Mat类的对象即可,且需为单通道8位或者浮点型图像。
- 第二个参数，OutputAray类型的dst，函数调用后的运算结果存在这里,即这个参数用于存放函数调用后的输出结果，需和源图片有一样的尺寸和类型。
- 第三个参数，InputArray类型的 mapl，它有两种可能的表示对象。
表示点(x,y)的第一个映射。
表示CV_16SC2、cV_32FC1 或CV_32FC2类型的X值。
- 第四个参数，InputArray类型的 map2，同样,它也有两种可能的表示对象，而且它会根据 map1来确定表示那种对象。
若mapl表示点（x,y）时。这个参数不代表任何值。
表示CV 16UCl.CV_32FCI类型的Y值(第二个值)。
- 第五个参数，int类型的interpolation，插值方式，之前的 resize()函数中有讲到,需要注意,resize()函数中提到的INTER_AREA插值方式在这里是不支持的,所以可选的插值方式如下(需要注意,这些宏相应的OpenCV2版为在它们的宏名称前面加上“CV_”前缀，比如“INTER_LINEAR”的OpenCV2版为“CVINTER LINEAR”):
 INTERNEAREST———最近邻插值
INTER_LINEAR——双线性插值(默认值)
 INTER CUBIC———双三次样条插值（逾4×4像素邻域内的双三次插值) INTER LANCZOS4——Lanczos插值(逾8×8像素邻域的Lanczos 插值)
 - 第六个参数，int类型的 borderMode，边界模式，有默认值 BORDERCONSTANT、表示目标图像中“离群点（outliers)”的像素值不会被此函数修改。
- 第七个参数，const Scalar&类型的borderValue，当有常数边界时使用的值，其有默认值Scalar()，即默认值为0。

#### 实现多种重映射综合示例
4种映射方式
[![Dgpv9J.png](https://s3.ax1x.com/2020/11/29/Dgpv9J.png)](https://imgchr.com/i/Dgpv9J)
[![DgpzcR.png](https://s3.ax1x.com/2020/11/29/DgpzcR.png)](https://imgchr.com/i/DgpzcR)
[![Dg9CB6.png](https://s3.ax1x.com/2020/11/29/Dg9CB6.png)](https://imgchr.com/i/Dg9CB6)
[![Dg9kND.png](https://s3.ax1x.com/2020/11/29/Dg9kND.png)](https://imgchr.com/i/Dg9kND)

>>## 7.4仿射变换
#### 认识仿射变换
仿射变换（Affine Transformation或Affine Map)，又称仿射映射，是指在几何中，一个向量空间进行一次线性变换并接上一个平移,变换为另一个向量空间的过程。它保持了二维图形的“平直性”(直线经过变换之后依然是直线)和“平行性”(二维图形之间的相对位置关系保持不变,平行线依然是平行线,且直线上点的位置顺序不变)。
一个任意的仿射变换都能表示为乘以一个矩阵(线性变换)接着再加上一个向量（平移）的形式。
那么，我们能够用仿射变换来表示如下三种常见的变换形式:旋转,rotation(线性变换)
平移，translation(向量加)缩放，scale(线性变换)

#### 仿射变换的求法
我们知道，仿射变换表示的就是两幅图片之间的一种联系，关于这种联系的信息大致可从以下两种场景获得。
- 已知X和T，而且已知它们是有联系的。接下来的工作就是求出矩阵M。
- 已知M和X，想求得T。只要应用算式T=M·X即可。对于这种联系的信
息可以用矩阵M清晰地表达(即给出明确的2x3矩阵),也可以用两幅图片点之间儿何关系来表达。

#### 进行仿射变换:warpAffineO函数
warpAffine函数的作用是依据以下公式子,对图像做仿射变换。dst(x, y》=src(MG;X+Mpy+ M,Ma1x+ May+Ma
函数原型如下
C++:void warpAffine (InputArray src, outputArray dst,InputArray M,sizedsize,int flags=INTER_LINEAR,intborderMode=BORDER_CONSTANT,constScalar& bordervalue=sealar)

#### 计算二维旋转变换矩阵:getRotationMatrix2D（）的函数
getRotationMatrix2D(O函数用于计算二维旋转变换矩阵。变换会将旋转中心映射到它自身。
C++:Mat getRotationMatrix2D(Point2fcenter,double angle， double scale)
- 第一个参数，Point2f类型的center，表示源图像的旋转中心。
- 第二个参数，double类型的angle，旋转角度。角度为正值表示向逆时针旋转(坐标原点是左上角)。
- 第三个参数,double类型的scale,缩放系数。

#### 仿射变换综合示例
[![Dg9T8H.png](https://s3.ax1x.com/2020/11/29/Dg9T8H.png)](https://imgchr.com/i/Dg9T8H)

>>## 7.5直方图均衡化
#### 实现直方图均衡化: equalizeHist（）函数
在OpcnCV中，直方图均衡化的功能实现由 equalizeHist 函数完成。我们一起看看它的函数描述。
C++: void equalizeHist (InputArray src，outputArray dst)
- 第一个参数，InputArray类型的src，输入图像，即源图像，填Mat类的对象即可，需为8位单通道的图像。
- 第二个参数，OutputArray类型的 dst，函数调用后的运算结果存在这里，需和源图片有一样的尺寸和类型。

#### 直方图均衡化
[![DgCXl9.png](https://s3.ax1x.com/2020/11/29/DgCXl9.png)](https://imgchr.com/i/DgCXl9)

># CH08 图像轮廓与图像分割修复

>>## 8.1查找并绘制轮廓
#### 寻找轮廓: findContours()函数
findContours(函数用于在二值图像中寻找轮廓。
C++: void findContours(InputoutputArray image，outputArrayofArrayscontours，outputArray hierarchy，int mode，int method，Point
offset=Point ( ) )

#### 绘制轮廓:drawContours(函数
drawContours()函数用于在图像中绘制外部或内部轮廓。
C++: void drawContours(InputoutputArray image，InputArrayofArrayscontours,int contourIdx，const scalar& color，int thickness=1，intlineType=8,InputArray hierarchy=noArray( ) , int maxLevel=INT_MAX，Pointoffset=Point ( ) )

#### 轮廓查找基础
[![DgPg76.png](https://s3.ax1x.com/2020/11/29/DgPg76.png)](https://imgchr.com/i/DgPg76)

>>## 8.2寻找物体的凸包
#### 凸包
凸包(Convex Hull)是一个计算几何（图形学）中常见的概念。简单来说，给定二维平面上的点集，凸包就是将最外层的点连接起来构成的凸多边型，它是能包含点集中所有点的。理解物体形状或轮廓的一种比较有用的方法便是计算一个物体的凸包，然后计算其凸缺陷(convexity defects)。很多复杂物体的特性能很好地被这种缺陷表现出来。

#### 寻找凸包: convexHullO函数
上文已经提到过，convexHull()函数用于寻找图像点集中的凸包，其原型声明如下。
C++ : void convexHull(InputArray points，outputArray hull，boolclockwise=false, bool returnPoints=true )
- 第一个参数，InputArray类型的points，输入的二维点集，可以填Mat类型或者std::vector。
- 第二个参数，OutputArray类型的 hull，输出参数，函数调用后找到的凸包。
- 第三个参数，bool类型的clockwise，操作方向标识符。当此标识符为真时，输出的凸包为顺时针方向。否则，就为逆时针方向。并且是假定坐标系的x轴指向右，y轴指向上方。
- 第四个参数，bool类型的returnPoints，操作标志符，默认值 true。当标志符为真时，函数返回各凸包的各个点。否则，它返回凸包各点的指数。当输出数组是std::vector 时，此标志被忽略。

#### 凸包检测基础
[![DgFsdx.png](https://s3.ax1x.com/2020/11/29/DgFsdx.png)](https://imgchr.com/i/DgFsdx)

>>## 8.3使用多边形将轮廓包围
#### 返回外部矩形边界:boundingRect()函数
此函数计算并返回指定点集最外面(up-right）的矩形边界。C++: Rect boundingRect (InputArray points)
其唯一的一个参数为输入的二维点集，可以是std::vector或Mat类型。

#### 寻找最小包围矩形:minAreaRect()函数
此函数用于对给定的2D点集,寻找可旋转的最小面积的包围矩形。C+4: RotatedRect minAreaRect(InputArray points)
其唯一的一个参数为输入的二维点集,可以为std::vector>或Mat类型。

#### 寻找最小包围圆形:minEnclosingCircle()函数
minEnclosingCircle函数的功能是利用一种迭代算法,对给定的2D点集，去寻找面积最小的可包围它们的圆形。
C++: void minEnclosingCircle (InputArray points,Point2f& center,float&radius)

#### 用椭圆拟合二维点集:fitEllipse()函数
此函数的作用是用椭圆拟合二维点集。
c+: RotatedRect fi上Ellipse( inputArray points)
其唯一的一个参数为输入的二维点集,可以为 std::vector>或Mat类型。

#### 逼近多边形曲线:approxPolyDP()函数
approxPolyDP函数的作用是用指定精度逼近多边形曲线。
C++:void approxPolyDPIInputArray curve,outputArray approxcurve,double epsilon, bool closed)
- 第一个参数，InputArray类型的curve，输入的二维点集,可以为 std::vecto或Mat类型。
- 第二个参数，OutputArray类型的 approxCurve，多边形逼近的结果，其类型应该和输入的二维点集的类型一致。
- 第三个参数,double类型的epsilon，逼近的精度，为原始曲线和即近似曲线间的最大值。
- 第四个参数，bool类型的closed,如果其为真,则近似的曲线为封闭曲线（第一个顶点和最后一个顶点相连)，否则,近似的曲线曲线不封闭。
讲解完上述的几个函数，下面我们来通过完整的示例程序理解所学。笔者为大家准备了两个示例程序，分别为用矩形和圆形边界包围生成的轮廓。

#### 创建包围轮廓的矩形边界和圆形边界框
[![DgESiV.png](https://s3.ax1x.com/2020/11/29/DgESiV.png)](https://imgchr.com/i/DgESiV)
 
>>## 8.4图像的矩
矩函数在图像分析中有着广泛的应用，如模式识别、目标分类、目标识别与方位估计、图像编码与重构等。一个从一幅数字图形中计算出来的矩集，通常描述了该图像形状的全局特征，并提供了大量的关于该图像不同类型的几何特性信息，比如大小、位置、方向及形状等。图像矩的这种特性描述能力被广泛地应用在各种图像处理、计算机视觉和机器人技术领域的目标识别与方位估计中。一阶矩与形状有关,二阶矩显示曲线围绕直线平均值的扩展程度，三阶矩则是关于平均值的对称性的测量。由二阶矩和三阶矩可以导出一组共7个不变矩。而不变矩是图像的统计特性，满足平移、伸缩、旋转均不变的不变性,在图像识别领域得到了广泛的应用。
那么，在 OpenCV中，如何计算一个图像的矩呢?一般由moments ，contourArea、 arcLength这三个函数配合求取。
使用 moments计算图像所有的矩(最高到3阶)使用contourArea来计算轮廓面积
使用 arcLength来计算轮廓或曲线长度

#### 矩的计算:moments（）函数
moments()函数用于计算多边形和光瓯形状的最高达三阶的所有矩。矩用来计算形状的重心、面积,主轴和其他形状特征,如7Hu不变量等。
C++: Moments moments(InputArray array,bool binaryImage=false ,
- 第一个参数，InputArray类型的array，输入参数,可以是光栅图像(单通道,8位或浮点的二维数组)或二维数组（IN或N1)。
- 第二个参数，bool类型的binarylmage，有默认值false。若此参数取 true,则所有非零像素为1。此参数仅对于图像使用。

#### 计算轮廓面积:contourArea（）函数
contourArea()函数用于计算整个轮廓或部分轮廓的面积
C+t: double contourArea (InputArray contour,bool oriented=falseh
- 第一个参数，InputArray类型的contour，输入的向量，二维点(轮廓顶点),可以为std::vector 或 Mat类型。
- 第二个参数，bool类型的 oriented，面向区域标识符。若其为true，该函数返回一个带符号的面积值,其正负取决于轮廓的方向(顺时针还是逆时针)。根据这个特性我们可以根据面积的符号来确定轮狙的位置。需要注意的是，这个参数有默认值false,表示以绝对值返回,不带符号。

#### 计算轮廓长度:arcLength函数
arcLength(函数用于计算封闭轮廓的周长或曲线的长度。C+t: double arcLength( InputArray curve, bool ciosed)
- 第一个参数，InputArray类型的curve,输入的二维点集,可以为std:vector
8.4图像的矩
或Mat类型。
- 第二个参数，bool类型的 closed，一个用于指示曲线是否封闭的标识符，有默认值closed，表示曲线封闭。

#### 查找和绘制图片轮廓矩
[![DgQly6.png](https://s3.ax1x.com/2020/11/29/DgQly6.png)](https://imgchr.com/i/DgQly6)

>>##  8.5分水岭算法
分水岭算法，是一种基于拓扑理论的数学形态学的分割方法，其基本思想是把图像看作是测地学上的拓扑地貌，图像中每一点像素的灰度值表示该点的海拔高度，每一个局部极小值及其影响区域称为集水盆，而集水盆的边界则形成分水岭。分水岭的概念和形成可以通过模拟浸入过程来说明:在每一个局部极小值表面，刺穿一个小孔，然后把整个模型慢慢浸入水中，随着浸入的加深，每一个局部极小值的影响域慢慢向外扩展，在两个集水盆汇合处构筑大坝，即形成分水岭。
分水岭的计算过程是一个迭代标注过程。分水岭比较经典的计算方法是由L. Vincent提出的。在该算法中，分水岭计算分两个步骤:一个是排序过程，一个是淹没过程。首先对每个像素的灰度级进行从低到高的排序，然后在从低到高实现淹没的过程中，对每一个局部极小值在h阶高度的影响域采用先进先出（FIFO）结构进行判断及标注。分水岭变换得到的是输入图像的集水盆图像，集水盆之间的边界点，即为分水岭。显然，分水岭表示的是输入图像的极大值点。

#### 实现分水岭算法: watershed（）函数
函数watershed实现的分水岭算法是基于标记的分割算法中的一种。在把图像传给函数之前，我们需要大致勾画标记出图像中的期望进行分割的区域，它们被标记为正指数。所以，每一个区域都会被标记为像素值1、2、3等，表示成为一个或者多个连接组件。这些标记的值可以使用findContours()函数和 drawContours()函数由二进制的掩码检索出来。不难理解，这些标记就是即将绘制出来的分割区域的“种子”，而没有标记清楚的区域，被置为0。在函数输出中，每一个标记中的像素被设置为“种子”的值，而区域间的值被设置为-1。
C++ : void watershed (InputArray image,InputoutputArray markers)
第一个参数，InputArray类型的src，输入图像，即源图像，填Mat类的对象即可，且需为8位三通道的彩色图像。
第二个参数，InputOutputArray类型的markers，函数调用后的运算结果存在这里，输入/输出 32位单通道图像的标记结果。即这个参数用于存放函数调用后的输出结果，需和源图片有一样的尺寸和类型。

#### 分水岭算法
[![DglIgI.png](https://s3.ax1x.com/2020/11/29/DglIgI.png)](https://imgchr.com/i/DglIgI)

>>## 8.6图像修补
在实际应用中，我们的图像常常会被噪声腐蚀，这些噪声或者是镜头上的灰尘或水滴，或者是旧照片的划痕，或者由于图像的部分本身已经损坏。而“图像修复”( Inpainting)，就是妙手回春，解决这些问题的良方。图像修复技术简单来说，就是利用那些已经被破坏区域的边缘，即边缘的颜色和结构，繁殖和混合到损坏的图像中，以达到图像修补的目的。图8.34~8.36就是示例程序截图，演示将图像中的字迹移除的效果。

#### 图像修补
[![Dg3PSA.png](https://s3.ax1x.com/2020/11/29/Dg3PSA.png)](https://imgchr.com/i/Dg3PSA)

># CH09 直方图与匹配

>>## 9.1图像直方图概述
图像直方图（Image Histogram）是用以表示数字图像中亮度分布的直方图，标绘了图像中每个亮度值的像素数。可以借助观察该直方图了解需要如何调整亮度分布。这种直方图中，横坐标的左侧为纯黑、较暗的区域，而右侧为较亮、纯白的区域。因此，一张较暗图片的图像直方图中的数据多集中于左侧和中间部分，而整体明亮、只有少量阴影的图像则相反。计算机视觉领域常借助图像直方图来实现图像的二值化。
直方图的意义如下。
直方图是图像中像素强度分布的图形表达方式。它统计了每一个强度值所具有的像素个数。

>>## 9.2直方图的计算与绘制
直方图的计算在OpenCV中可以使用calcHist()函数，而计算完成之后，可以采用OpenCV中的绘图函数，如绘制矩形的rectangle()函数，绘制线段的line()来完成。

#### 计算直方图:calcHist(函数
在 OpenCV中，calcHist(函数用于计算一个或者多个阵列的直方图。原型如
C++:void calcHist (const Mat* images,int nimages,const int*channels,InputArray mask,OutputArray hist, int dims, const int* histSize, constf1oat*- ranges,bool uniform-true,bool accumulate-false ,

#### 找寻最值:minMaxLoc()函数
minMaxLoc()函数的作用是在数组中找到全局最小值和最大值。它有两个版本的原型,在此介绍常用的那一个版本。
C++:void minMaxLoc (InputArray sre, double* minval,double* maxVal-0,Point* minLoc=0, Point* maXLoc=0，InputArray mask=noArray())
- 第一个参数,InputArray类型的src,输入的单通道阵列。
- 第二个参数，double*类型的 minVal，返回最小值的指针。若无须返回，此值置为 NULL。
- 第三个参数，double*类型的maxVal，返回的最大值的指针。若无须返回,此值置为 NULL。
- 第四个参数，Point*类型的 minLoc,返回最小位置的指针(二维情况下)。若无须返回,此值置为NULL。
- 第五个参数，Point*类型的maxLoc，返回最大位置的指针(二维情况下)。若无须返回，此值置为NULL。
- 第六个参数，InputArray类型的mask，用于选择子阵列的可选掩膜。

>>## 9.3直方图的对比
#### 对比直方图: compareHist()函数
compareHist()函数用于对两幅直方图进行比较。有两个版本的C++原型，如下。
C++: double compareHist(InputArray Hl，InputArray H2，int method)C++ : double compareHist (const SparseMat& Hl,const SparseMat& H2，int method)

#### 直方图对比
[![DgUCAx.png](https://s3.ax1x.com/2020/11/29/DgUCAx.png)](https://imgchr.com/i/DgUCAx)

>>## 9.4反向投影
#### 反向投影的作用
反向投影用于在输入图像（通常较大）中查找与特定图像（通常较小或者仅1个像素，以下将其称为模板图像）最匹配的点或者区域，也就是定位模板图像出现在输入图像的位置。

#### 反向投影的结果
反向投影的结果包含了以每个输入图像像素点为起点的直方图对比结果。可以把它看成是一个二维的浮点型数组、二维矩阵，或者单通道的浮点型图像。

#### 反向投影
[![DgaYRO.png](https://s3.ax1x.com/2020/11/29/DgaYRO.png)](https://imgchr.com/i/DgaYRO)

>>## 9.5模板匹配
#### 模板匹配的概念与原理
模板匹配是一项在一幅图像中寻找与另一幅模板图像最匹配（相似）部分的技术。在 OpenCV2和 OpenCV3中，模板匹配由Match Template()函数完成。需要注意，模板匹配不是基于直方图的，而是通过在输入图像上滑动图像块，对实际的图像块和输入图像进行匹配的一种匹配方法。

#### 实现模板匹配:matchTemplate(的数
matchTemplateO用于匹配出和模板重叠的图像区域。
C+4 ;void matchTemplate(InputArray image，InputArray templ,outputArrayresult, int method)
- 第一个参数，InputArray类型的 image，待搜索的图像，且需为8位或32位浮点型图像。
- 第二个参数，InputArray类型的 templ，搜索模板，需和源图片有一样的数据类型,且尺寸不能大于源图像。
- 第三个参数，OutputArray类型的result，比较结果的映射图像。其必须为单通道、32位浮点型图像.如果图像尺寸是WXH 而templ尺寸是 wxh ,则此参数result一定是(W-w+1)<(H-h+1).
- 第四个参数，int类型的method，指定的匹配方法,OpenCV为我们提供了如下6种图像匹配方法可供使用。
1.平方差匹配法method=TM_SQDIFF
这类方法利用平方差来进行匹配，最好匹配为0。而若匹配越差，匹配值则越大。
2.归一化平方差匹配法 method=TM_SQDIFF_NORMED
3．相关匹配法method=TM_CCORR
4.归一化相关匹配法method=TM_CCORR_NORMED
5．系数匹配法 method=TM_CCOEFF
6.化相关系数匹配法method=TM_CCOEFF_NORMED

#### 模板匹配
[![DgdilD.png](https://s3.ax1x.com/2020/11/29/DgdilD.png)](https://imgchr.com/i/DgdilD)


>>## 心得体会
通过6-9章的学习，学到更多图像处理的方式，也学习到了很多函数有关图像处理的函数，感觉opencv是一个非常有趣的处理图像的软件库，让代码不再那么枯燥了。