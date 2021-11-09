//#include <iostream>
//#include <chrono>
//#include "cuda_runtime_api.h"
//#include "logging.h"
//#include "common.hpp"
//#include "yolov5.h"
#include "roidetector.h"

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id

int main(int argc, char *argv[])
{

    init("defect_yolov5s", "./Debug/resources/defect_yolov5s.wts", DEVICE, 7);

    bbox_t_container* res_container = new bbox_t_container[25];
	int res_len = detect_image("./data", "test010006", res_container);
	std::cout << res_len << std::endl;
}
