#pragma once
#include "dirent.h"
#include <opencv2/opencv.hpp>

#define C_SHARP_MAX_OBJECTS 20
typedef unsigned char uint8_t;

struct Result{
    int id = -1;
    float prob = 0.f;
    cv::Rect rect;
};

typedef std::vector<Result> BatchResult;

struct bbox_t {
    unsigned int x, y , w, h;
    float prob;
    unsigned int obj_id;
    unsigned int track_id;
    unsigned int frames_counter;
	float x_3d, y_3d, z_3d;  // 3-D coordinates, if there is used 3D-stereo camera
};

struct bbox_t_container {
    bbox_t candidates[C_SHARP_MAX_OBJECTS];
};

extern "C" __declspec(dllexport) int init(const char* model_cfg, const char* model_weights, int gpu, int class_num);
extern "C" __declspec(dllexport) int detect_image(const char* root_dir, const char* object_name, bbox_t_container* container);
extern "C" __declspec(dllexport) int detect_mat(const uint8_t* data, const int* data_length, bbox_t_container* container);
extern "C" __declspec(dllexport) int dispose();

