#ifndef __SINGLETON_HPP_
#define __SINGLETON_HPP_
#include <stddef.h>  // defines NULL
#include <iostream>
#include <cassert>

template <class T>
class Singleton
{
public:
//static修饰的函数是属于类的，所以没有this指针，
//所以static类成员函数不能访问非static的类成员，只能访问static修饰的类成员
  static T& Instance() {
      if(!m_pInstance){
        m_pInstance = new T;
        std::cout << "New Instance created" << std::endl;
      } 
      assert(m_pInstance != NULL);
      return *m_pInstance;
  }
protected:
  Singleton();
  ~Singleton();
private:
  //将复制构造函数和“=”操作符也设为私有，防止被复制
  Singleton(Singleton const&);
  Singleton& operator=(Singleton const&);
  static T* m_pInstance;
};

//类的静态成员，独立于一切类的对象存在，必须先在类外进行初始化。static修饰的变量先与类对象存在，所以必须要在类外先进行初始化。
//static修饰的变量在静态存储区生成
template <class T> T* Singleton<T>::m_pInstance=NULL;

#endif