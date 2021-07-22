#ifndef YOLOV5_COMMON_H_
#define YOLOV5_COMMON_H_

#include <fstream>
#include <map>
#include <sstream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include "NvInfer.h"

int read_files_in_dir(const char *p_dir_name, std::vector<std::string> &file_names) {
    DIR *p_dir = opendir(p_dir_name);
    if (p_dir == nullptr) {
        return -1;
    }

    struct dirent* p_file = nullptr;
    while ((p_file = readdir(p_dir)) != nullptr) {
        if (strcmp(p_file->d_name, ".") != 0 &&
            strcmp(p_file->d_name, "..") != 0) {
            //std::string cur_file_name(p_dir_name);
            //cur_file_name += "/";
            //cur_file_name += p_file->d_name;
            std::string cur_file_name(p_file->d_name);
            file_names.push_back(cur_file_name);
        }
    }

    closedir(p_dir);
    return 0;
}


cv::Mat preprocess_img(cv::Mat& img, int INPUT_W, int INPUT_H) {
    int x, y;
    //float r_w = INPUT_W / (img.cols*1.0);
    //float r_h = INPUT_H / (img.rows*1.0);
    //if (r_h > r_w) {
    //    w = INPUT_W;
    //    h = r_w * img.rows;
    //    x = 0;
    //    y = (INPUT_H - h) / 2;
    //} else {
    //    w = r_h * img.cols;
    //    h = INPUT_H;
    //    x = (INPUT_W - w) / 2;
    //    y = 0;
    //}
    //cv::Mat re(h, w, CV_8UC3);
    //cv::resize(img, re, re.size(), 0, 0, cv::INTER_LINEAR);
    x = (INPUT_W - img.cols) / 2;
	y = (INPUT_H - img.rows) / 2;
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    img.copyTo(out(cv::Rect(x, y, img.cols, img.rows)));
	//cv::rotate(out, out, cv::ROTATE_90_CLOCKWISE);
    return out;
}

#endif

