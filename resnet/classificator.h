#pragma once
#include "dirent.h"
#include <opencv2/opencv.hpp>
#include "common.h"

extern "C" __declspec(dllexport) int init(const char* model_cfg, const char* model_weights, int gpu);
extern "C" __declspec(dllexport) int class_image(const char* file_name, float& conf);
extern "C" __declspec(dllexport) int class_mat(const uint8_t* data, const size_t data_length, float& conf);
extern "C" __declspec(dllexport) int dispose();
