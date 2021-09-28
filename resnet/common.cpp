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
    int x, y, w, h, x_temp, y_temp;
    x = (INPUT_W - img.cols) / 2;
	y = (INPUT_H - img.rows) / 2;
	w = x > 0 ? INPUT_W: img.cols;
	h = y > 0 ? INPUT_H: img.rows;
	x_temp = x > 0 ? 0 : -x;
	y_temp = y > 0 ? 0 : -y;
	x = x > 0 ? x : 0;
	y = y > 0 ? y : 0;
	cv::Mat temp(h, w, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
    img.copyTo(temp(cv::Rect(x, y, img.cols, img.rows)));
	temp(cv::Rect(x_temp, y_temp, INPUT_W, INPUT_H)).copyTo(out);
	//cv::rotate(out, out, cv::ROTATE_90_CLOCKWISE);
	//cv::imwrite("../patch.jpg", out);
    return out;
}

bool compare_filetime(std::string& engine_name, std::string& wts_name) {
    // check if engine need to be regenerated
	HANDLE engine_handle = CreateFile(engine_name.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
	HANDLE weight_handle = CreateFile(wts_name.c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    FILETIME engine_write;
    FILETIME weight_write;
    bool outdated = false;

    if (!GetFileTime(engine_handle, NULL, NULL, &engine_write))
    {
        outdated = true;
    }

    if (!GetFileTime(weight_handle, NULL, NULL, &weight_write))
    {
        std::cerr << "could not open model weights" << std::endl;
		CloseHandle(engine_handle);
		CloseHandle(weight_handle);
        return -1;
    }

	if (CompareFileTime(&weight_write, &engine_write) > 0)
	{
        outdated = true;
	}
	CloseHandle(engine_handle);
	CloseHandle(weight_handle);
	return outdated;
}
