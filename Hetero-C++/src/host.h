#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iterator>
#include <time.h>
#include <chrono>
#include <vector>
#include <unistd.h>
#include <iostream>
#include <fstream>

template <typename T>
struct aligned_allocator
{
  using value_type = T;
  T* allocate(std::size_t num)
  {
    void* ptr = nullptr;
    if (posix_memalign(&ptr,4096,num*sizeof(T)))
      throw std::bad_alloc();
    return reinterpret_cast<T*>(ptr);
  }
  void deallocate(T* p, std::size_t num)
  {
    free(p);
  }
};

int train = 3;

#define N_CLASS		26	//number of classes. (e.g., isolet: 26, ucihar 12)
#define Dhv				2048  //hypervectors length
std::string X_train_path = "./isolet_trainX.bin";
std::string y_train_path = "./isolet_trainY.bin";
std::string X_test_path = "./isolet_testX.bin";
std::string y_test_path = "./isolet_testY.bin";


