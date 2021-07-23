#include "classificator.h"

int main(int argc, char** argv)
{
    init("../", "../", 0);

    std::vector<std::string> file_names;
	//const char* infer_root = "D:/Project/blur/test_blur";
	const char* infer_root = "../test_blur";
    if (read_files_in_dir(infer_root, file_names) < 0) {
        std::cout << "read_files_in_dir failed." << std::endl;
        return -1;
    }
    std::cout << "# of files in dirs: " << file_names.size() << std::endl;

    for (int f = 0; f < (int)file_names.size(); f++) {
        class_image((std::string(infer_root) + "/" + file_names[f]).data());
    }
    
    dispose();


    return 0;
}