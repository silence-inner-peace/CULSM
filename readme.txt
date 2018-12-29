nvcc -O3 -pg -g -G -std=c++11 AbstractBlockClass.cu GDALRead.cpp UnionFind.cpp main.cu -o main -lgdal
./main ../../data/california.tif
gprof ./main gmon.out > result.txt
gprof2dot -n0 -e0 -w ./result.txt > result.dot


valgrind --tool=callgrind ./main ../../data/california.tif
gprof2dot -f callgrind callgrind.out.XXX | dot -o callgrind.out.dot

当前版本中所有的指数都用一个相同的变量名表示，只需要重写这两个类即可实现不同指数的计算
						cal____ForEachPixel,	//用于计算单个像元的指数，以便于后续的统计
						cal____Metrics, 		//用于统计单个像元，得到斑块级别的参数