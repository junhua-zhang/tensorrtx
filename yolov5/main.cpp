//#include <iostream>
//#include <chrono>
//#include "cuda_runtime_api.h"
//#include "logging.h"
//#include "common.hpp"
//#include "yolov5.h"
#include "roidetector.h"

//#define USE_FP16  // comment out this if want to use FP32
#define DEVICE 0  // GPU id

const char* viewName[25] = {"top", "x00","x01", "x02", "x03", 
							"x04", "x05", "x06", "x07", "x08",
							"x09", "x10", "x11", "x12", "x13",
							"x14", "x15", "x16", "x17", "x18",
							"x19", "x20", "x21", "x22", "x23" };

int main(int argc, char *argv[])
{

    init("defect_yolov5s", "./Debug/resources/defect_yolov5s.wts", DEVICE, 7);
	int img_count = sizeof(viewName) / sizeof(viewName[0]);

    bbox_t_container* res_container = new bbox_t_container[25];
	int res_len = detect_image("./data", "test010006", viewName, img_count,  res_container);
	std::cout << res_len << std::endl;
}
