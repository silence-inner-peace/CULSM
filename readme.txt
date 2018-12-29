## 对核函数的封装
因为在CUDA中__global__函数不能作为类的成员函数，所以为了将CUDA和C++类更好的结合起来使用需要对核函数进行一层包装。在类的外边定义核函数，在类的成员函数中调用核函数，将类的成员变量作为参数传入。
通过面向对象的方式封装核函数调用部分，将线程索引值获取和判断、显存的申请和释放，以及核函数的调用封装在对象中，通过继承实现纯虚函数来进行计算。编程人员需要了解继承来的内置对象的含义，然后通过这些对象及其方法进行自己的程序编写自己的设备函数，这里通过实现kernel函数，对象调用执行函数时，执行自定义的核函数。
## 对显存的封装
开发CUDA库涉及到一个问题，因为CUDA是一个非常低级的语言，如果库中有复杂的数据结构，就有可能很难管理数据分配、内存传输。在CPU端，C++类可以通过抽象的方式使得开发较容易。理想情况下，希望在GPU上能够做同样的事情。通过模板元编程，能够为现有的类创建CUDA-interface，通过抽象的处理cudaMalloc,cudaMemcpy等低级的GPU内存管理操作，从而大大简化GPU的开发。
CUDA程序的步骤中必不可少的就是申请显存空间，以及将数据从主机端拷贝到设备端，CUDA的Runtime函数主要是控制内存管理，包括内存申请和数据传输。栅格数据在逻辑计算中认为以二维平面分布，显存申请时可以统一处理。CUDA程序步骤中的内存申请以及拷贝部分可以对用户透明。通过定义Array2D_CUDA类封装CUDA数组，隐藏显存的申请、拷贝、释放。
为了区别于CPU版本的Array2D，引入了一个辅助结构体，在结构体中只包含一个值。现在能够创建一个Array2D< Cutype<T>>数组，这与Array2D<T>的对象完全不同。这样就可以通过抽象的方法来调用cudaMalloc,cudaMemcpy等低级函数。例如，可以通过Cutype<float>用CUDA版的array本来存储float型的数据。这样一层数据类型并不会产生额外的开销。在编译阶段已经被float代替掉了。

## 编译
nvcc -O3 -pg -g -G -std=c++11 AbstractBlockClass.cu GDALRead.cpp UnionFind.cpp main.cu -o main -lgdal
./main ../../data/california.tif
gprof ./main gmon.out > result.txt
gprof2dot -n0 -e0 -w ./result.txt > result.dot


valgrind --tool=callgrind ./main ../../data/california.tif
gprof2dot -f callgrind callgrind.out.XXX | dot -o callgrind.out.dot

当前版本中所有的指数都用一个相同的变量名表示，只需要重写这两个类即可实现不同指数的计算
						cal____ForEachPixel,	//用于计算单个像元的指数，以便于后续的统计
						cal____Metrics, 		//用于统计单个像元，得到斑块级别的参数