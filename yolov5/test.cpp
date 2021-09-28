//#include <iostream>
//#include <chrono>
//#include "cuda_runtime_api.h"
//#include "logging.h"
//#include "common.hpp"
#include "kk.h"

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id

int main(int argc, char** argv) {
    test_func(1,2);
    return 0;
}
