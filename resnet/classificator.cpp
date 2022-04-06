#pragma once
#include "dirent.h"
#include <opencv2/opencv.hpp>
#include "classificator.h"
#include "resnet18.h"
#include <fstream>

IRuntime* runtime{ nullptr};
ICudaEngine* engine {nullptr};
IExecutionContext* context {nullptr};


int init(const char* model_cfg, const char* model_weights, int gpu) {
    char *trtModelStream{ nullptr };
    size_t size{ 0 };

    std::string engine_name = std::string(model_cfg) + ".engine";
    std::string wts_name = std::string(model_weights);

    // check if engine need to be regenerated
	bool outdated = compare_filetime(engine_name, wts_name);

    // generate resent18.engine
    if (outdated){
        IHostMemory* modelStream{ nullptr };
        APIToModel(1, &modelStream, wts_name);
        assert(modelStream != nullptr);

        std::ofstream p(engine_name, std::ios::binary|std::ofstream::out);
        if(!p){
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy(); 
        p.close();
    }

    // load resnet18.engine
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()) {
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
    }

    runtime = createInferRuntime(gLogger);
    assert(runtime != nullptr);
    engine = runtime->deserializeCudaEngine(trtModelStream, size, nullptr);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
}

void class_blur(cv::Mat &img, float& conf){
    assert(!img.empty());

    cv::Mat pr_img = preprocess_img(img, INPUT_W, INPUT_H);

    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];

    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
			//mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            data[i] = ((float)uc_pixel[2] / 255.0 - 0.485) / 0.229;
            data[i + INPUT_H * INPUT_W] = ((float)uc_pixel[1] / 255.0 - 0.456) / 0.224;
            data[i + 2 * INPUT_H * INPUT_W] = ((float)uc_pixel[0] / 255.0 - 0.406) / 0.225;
            //for ( int c = 0; c < 3; ++c){
            //	data[b * 3 * INPUT_H * INPUT_W + i + c * INPUT_H * INPUT_W] = ((float)uc_pixel[2-c] / 255.0 - mean[c]) / std1[c];
            //}
            uc_pixel += 3;
            ++i;
        }
    }
    //auto start = std::chrono::system_clock::now();
    doInference(*context, data, prob, 1);
    //auto end = std::chrono::system_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    //std::cout << prob[0] << ", " << prob[1] << std::endl;
	conf = 1 / (1 + exp(- prob[0]));
}

void class_image(const char* file_name, float& conf){
	cv::Mat img = cv::imread(file_name);
    class_blur(img, conf);
}

void class_mat(const uint8_t* data, const size_t data_length, float& conf){
    std::vector<char> vdata(data, data + data_length);
	cv::Mat img = imdecode(cv::Mat(vdata), 1);
    class_blur(img, conf);
}


int dispose(){
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 1;
}
