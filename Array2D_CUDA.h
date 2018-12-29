#ifndef ARRAY2D_CUDA_H
#define ARRAY2D_CUDA_H
#include <iostream>
#include "Array2D.h"
#include "cuda.h"
#include "cuda_runtime_api.h"
using namespace std;

template <class T>
struct Cutype{
    T val;
};


template <class U>
class Array2D< Cutype<U> > {
public:
    Array2D(const size_t& _nrows,
            const size_t& _ncols);
    Array2D(U* _h_data,
            const size_t& _nrows,
            const size_t& _ncols);
    Array2D(const Array2D<U>&);
    Array2D< Cutype<U> >& operator=(const Array2D<U>& other);
    U& operator[](int i);
    ~Array2D();
    size_t get_nrows() const {return h_nrows;}
    size_t get_ncols() const {return h_ncols;}
    size_t size() const {return h_N;}
    U* getDevData(void);
    U* getHostData(void);
    void set(U* h_data){cudaMemcpy(d_data,h_data,sizeof(U)*(size()),cudaMemcpyHostToDevice);}
    U* begin()const{return d_data;}
    U* end()const{return d_data + this->size();}
    U* begin(){return d_data;}
    U* end(){return d_data + this->size();}
    void show(void);
private:
    U* d_data;
    size_t h_nrows;
    size_t h_ncols;
    size_t h_N;
};

template <class U>
Array2D< Cutype<U> >::Array2D(const size_t& _nrows,
                            const size_t& _ncols){
    size_t N_tmp = _nrows * _ncols;

    h_N = N_tmp;
    h_nrows = _nrows;
    h_ncols = _ncols;

    cudaMalloc((void**)&d_data , sizeof(U) * N_tmp);
    cudaMemset(d_data, 0, sizeof(U)*N_tmp);
};

template <class U>
Array2D< Cutype<U> >::Array2D(U* _h_data,
                    const size_t& _nrows,
                    const size_t& _ncols):d_data(_h_data){
    size_t N_tmp = _nrows * _ncols;
    
    h_N = N_tmp;
    h_nrows = _nrows;
    h_ncols = _ncols;

    cudaMalloc((void**)&d_data , sizeof(U) * N_tmp);
    cudaMemcpy(d_data,  _h_data  , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);
};

template <class U>
Array2D< Cutype<U> >::Array2D(const Array2D<U>& other){
    size_t N_tmp = other.size();
    h_nrows = other.get_nrows();
    h_ncols = other.get_ncols();
    h_N = other.size();

    cudaMalloc((void**)&d_data , sizeof(U) * N_tmp);
    U *other_data = other.begin();
    cudaMemcpy(d_data,  other_data  , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);
}


template <class U>
Array2D< Cutype<U> >& Array2D< Cutype<U> >::operator=(const Array2D<U>& other){
    size_t N_tmp = other.size();
    h_nrows = other.get_nrows();
    h_ncols = other.get_ncols();
    h_N = other.size();

    cudaMalloc((void**)&d_data , sizeof(U) * N_tmp);
    U *other_data = other.begin();
    cudaMemcpy(d_data,  other_data , sizeof(U)*N_tmp, cudaMemcpyHostToDevice);

    return *this;
}

template <class U>
U& Array2D< Cutype<U> >::operator[](int i)
{
    if(i > size())
    {
        return d_data[0];
    }
    else
    {
        return d_data[i];
    }
}

template <class U>
U* Array2D< Cutype<U> >::getDevData(void)
{
    if(d_data!=NULL)
        return d_data;
    else
        return NULL;
}
template <class U>
Array2D< Cutype<U> >::~Array2D(){
    cout<<"cudaFree"<<endl;
    cudaFree(d_data);
}


template <class U>
U* Array2D< Cutype<U> >::getHostData(void){
    int length = h_N;
    U* h_data = new U[length];
    cudaMemcpy(h_data, getDevData(), sizeof(U)*length, cudaMemcpyDeviceToHost);
    return h_data;
}

template <class U>
void Array2D< Cutype<U> >::show(void)
{
    int length = h_N;
    U *h_data = getHostData();
    for (int i=0; i<length; i++) 
    {
        std::cout << h_data[i] << "\t";
    }
    std::cout << std::endl;
}
#endif //ARRAY2D_CUDA_H
