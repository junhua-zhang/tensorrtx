//#include <iostream>
//#include <chrono>
//#include "cuda_runtime_api.h"
//#include "logging.h"
//#include "common.hpp"
//#include "yolov5.h"
#include "detector.h"

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id

int main(){
    init("../", "../", DEVICE);
    bbox_t_container res_container;
    detect_image("../images/20210716040356.jpg", res_container);
}
