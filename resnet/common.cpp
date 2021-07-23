#include "common.h"

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
    x = (INPUT_W - img.cols) / 2;
	y = (INPUT_H - img.rows) / 2;
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    img.copyTo(out(cv::Rect(x, y, img.cols, img.rows)));
	//cv::rotate(out, out, cv::ROTATE_90_CLOCKWISE);
    return out;
}
