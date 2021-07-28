#include "roidetector.h"
#include "yolov5.h"

IRuntime *runtime{ nullptr};
ICudaEngine *engine{ nullptr };
IExecutionContext *context{ nullptr};
void* buffers[2];
int inputIndex = 0;
int outputIndex = 0;
cudaStream_t stream;

int init(const char* model_cfg, const char* model_weights, int gpu){
    cudaSetDevice(gpu);
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::string engine_name = STR2(NET);
    engine_name = "yolov5" + engine_name + ".engine";

	bool outdated = compare_filetime(engine_name, model_weights);

    //// generate yolov5s.engine
    if (outdated){
        IHostMemory* modelStream{ nullptr };
        APIToModel(BATCH_SIZE, &modelStream, model_weights);
        assert(modelStream != nullptr);
        std::ofstream p(engine_name, std::ios::binary);
        if (!p){
            std::cerr << "could not open plan output file" << std::endl;
            return -1;
        }
        p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
        modelStream->destroy();
        p.close();
    }

    // load yolov5s.engine
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good()){
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
    engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;
    assert(engine->getNbBindings() == 2);
    // In order to bind the buffers, we need to know the names of the input and output tensors.
    // Note that indices are guaranteed to be less than IEngine::getNbBindings()
    inputIndex = engine->getBindingIndex(INPUT_BLOB_NAME);
    outputIndex = engine->getBindingIndex(OUTPUT_BLOB_NAME);
    assert(inputIndex == 0);
    assert(outputIndex == 1);
    // Create GPU buffers on device
    CHECK(cudaMalloc(&buffers[inputIndex], BATCH_SIZE * 3 * INPUT_H * INPUT_W * sizeof(float)));
    CHECK(cudaMalloc(&buffers[outputIndex], BATCH_SIZE * OUTPUT_SIZE * sizeof(float)));
    // Create stream
    CHECK(cudaStreamCreate(&stream));
    return 1;
}

int detect_image(const char* file_name, bbox_t_container &container){
    //std::vector<BatchResult> batch_res;
    std::vector<bbox_t> return_res;
    cv::Mat img = cv::imread(file_name);
    // assert(img.empty());
    cv::Mat pr_img = preprocess_img(img);

    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<Yolo::Detection> res;
    //auto& res = batch_res;
    nms(res, prob, CONF_THRESH, NMS_THRESH);
    int res_len = res.size();
    for (size_t j = 0; j < res_len; j++){
        auto r = res[j];
        bbox_t res_j;
        res_j.x = r.bbox[0] > 0 ? r.bbox[0] : 0;
        res_j.y = r.bbox[1] > 0 ? r.bbox[1] : 0;
        res_j.w = r.bbox[2] > 0 ? r.bbox[2] : 0;
        res_j.h = r.bbox[3] > 0 ? r.bbox[3] : 0;
        res_j.prob = r.conf;
        res_j.obj_id = r.class_id;
        
        container.candidates[j] = res_j;
    }

    //std::cout << res_len << std::endl;
    //for (size_t j = 0; j < res.size(); j++) {
    //    cv::Rect r = get_rect(img, res[j].bbox);
    //    cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
    //    cv::putText(img, std::to_string((int)res[j].class_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
    //}
    //cv::imwrite("_test.jpg", img);

    return res_len;
}

int detect_mat(const uint8_t* mat, const size_t data_length, bbox_t_container &container){
    //TODO
    std::vector<char> vdata(mat, mat + data_length);
	cv::Mat img = imdecode(cv::Mat(vdata), 1);

    cv::Mat pr_img = preprocess_img(img);

    static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
    static float prob[BATCH_SIZE * OUTPUT_SIZE];
    int i = 0;
    for (int row = 0; row < INPUT_H; ++row) {
        uchar* uc_pixel = pr_img.data + row * pr_img.step;
        for (int col = 0; col < INPUT_W; ++col) {
            data[i] = (float)uc_pixel[2] / 255.0;
            data[i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
            data[i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
            uc_pixel += 3;
            ++i;
        }
    }

    // Run inference
    auto start = std::chrono::system_clock::now();
    doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
    auto end = std::chrono::system_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    std::vector<Yolo::Detection> res;
    //auto& res = batch_res;
    nms(res, prob, CONF_THRESH, NMS_THRESH);
    int res_len = res.size();
    for (size_t j = 0; j < res_len; j++){
        auto r = res[j];
        bbox_t res_j;
        //res_j.x = r.rect.x > 0 ? r.rec.x : 0;
        //res_j.y = r.rect.y > 0 ? r.rec.y : 0;
        //res_j.w = r.rect.width > 0 ? r.rect.width : 0;
        //res_j.h = r.rect.height > 0 ? r.rect.height : 0;
        //res_j.prob = r.prob;
        //res_j.obj_id = r.id;
        res_j.x = r.bbox[0] > 0 ? r.bbox[0] : 0;
        res_j.y = r.bbox[1] > 0 ? r.bbox[1] : 0;
        res_j.w = r.bbox[2] > 0 ? r.bbox[2] : 0;
        res_j.h = r.bbox[3] > 0 ? r.bbox[3] : 0;
        res_j.prob = r.conf;
        res_j.obj_id = r.class_id;
        
        container.candidates[j] = res_j;
    }

    return res_len;
}

int dispose(){
    // Release stream and buffers
    cudaStreamDestroy(stream);
    CHECK(cudaFree(buffers[inputIndex]));
    CHECK(cudaFree(buffers[outputIndex]));
    // Destroy the engine
    context->destroy();
    engine->destroy();
    runtime->destroy();
    return 1;
}