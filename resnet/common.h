#ifndef RESNET_COMMON_H_
#define RESNET_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names);

cv::Mat preprocess_img(cv::Mat& img, int INPUT_W, int INPUT_H);

bool compare_filetime(std::string& engine_name, std::string& wts_name);

#endif

