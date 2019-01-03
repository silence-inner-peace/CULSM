## 重写CCL算法用CUDA实现超大图像的分块合并以及连续标记   
此版本的主要修改目的:  
将连通域发现算法与景观指数的计算分离，先完成CCL、合并、重标记;  
然后再用标记完的连续标记数组来计算各个景观指数;  
这可以减少GPU内存空间的使用，免于对每个指数考虑不同的合并算法。何况有些指数不能合并;  
修改指数的计算流程，将标记后的数组存储到磁盘，在需要的时候读入，用GDAL写  


重写CCL算法，添加以下几个功能  
	- 四/八连通选项  
	- 生成指定用地类型的连通域  
	- 是否标记为连续序号（标记怎么做）先将所有根节点重标记，在对每个pixel进行重标记（CUDA）  
	- 提供h_src和d_src两种构造CCL的方法，同样可返回h_label或d_label;  

------

### 各个文件的作用  
- utils.h 用于定义一些工具函数，比如错误检测  
- timer.h	用于封装CPU，GPU的计时类  
- singleton.h	提供单例模式类模板，传入类型定义单例类  
- reduction_template.cuh	CUDA并行规约函数模板  
- compact_template.cuh	CUDA去除数组中的重复项函数模板  
- cudaConfig.cuh		CUDA线程块配置  
- UnionFind.cpp		用Unionfind 实现的分块合并处理  
- GlobalConfiguration.h	用户定义的全局配置参数  
- GDALRead._			GDAL读数据  
- GDALWrite._			GDAL写数据  
- AbstractBlockClass.h	定义分块类，用于保存分块的起始信息，分块src值，label值，提供重标记函数  
- AbstractBlockClass.cu	分块类的实现  
- cuCCL.cuh				CUDA实现并行连通域发现算法（connected component labeling）  
- BigImgCCL.cu			超大图像下CUDA实现并行连通域发现算法，主要在cuCCL.cuh的基础上添加了分块处理  

