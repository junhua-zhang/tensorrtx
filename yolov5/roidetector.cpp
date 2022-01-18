#include "roidetector.h"
#include "yolov5.h"
#include <thread>

IRuntime *runtime{ nullptr};
ICudaEngine *engine{ nullptr };
IExecutionContext *context{ nullptr};
void* buffers[2];
int inputIndex = 0;
int outputIndex = 0;
cudaStream_t stream;


void decode(const uint8_t* mat, int* start_list, int index, std::vector<cv::Mat>& img_vec)
{
	std::vector<char> vdata(mat + start_list[index], mat + start_list[index+1]);
	cv::Mat img = imdecode(cv::Mat(vdata), 1);
	img_vec[index] = img;
}


void copyimg(cv::Mat& img, float* data)
{
	assert(!img.empty());
	cv::Mat pr_img = preprocess_img(img);
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
}

void readimg(int i, const char* root_dir, const char* object_name, const char** viewName, std::vector<cv::Mat> &img_vec)
{
	std::string img_name = std::string(root_dir) + "/" + object_name + "/" + object_name + "_" + viewName[i] +".jpg";
    cv::Mat img = cv::imread(img_name);
	img_vec[i] = img; 
}

int init(const char* model_cfg, const char* model_weights, int gpu, int class_num){
    cudaSetDevice(gpu);
    char *trtModelStream{ nullptr };
    size_t size{ 0 };
    std::string engine_name = std::string(model_cfg) + ".engine";
	std::string wts_name = std::string(model_weights);

	bool outdated = compare_filetime(engine_name, wts_name);

    //// generate yolov5s.engine
    if (outdated){
        IHostMemory* modelStream{ nullptr };
		float gd = 0.0f, gw = 0.0f;
		parse_net(model_cfg, gd, gw);
        APIToModel(BATCH_SIZE, &modelStream, gd, gw, std::string(model_weights), class_num);
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

int detect_roi(cv::Mat img, bbox_t_container& container)
{
	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	static float prob[BATCH_SIZE * OUTPUT_SIZE];

	auto start = std::chrono::system_clock::now();
	copyimg(img, data);
	auto end = std::chrono::system_clock::now();
	std::cout << "preprocess: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	// Run inference
	start = std::chrono::system_clock::now();
	doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
	end = std::chrono::system_clock::now();
	std::cout << "inference: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

	start = std::chrono::system_clock::now();
	std::vector<Yolo::Detection> res;
	nms_id(res, prob, CONF_THRESH, NMS_THRESH);

	int res_len = res.size();
	for (size_t j = 0; j < res_len; j++) {
		auto r = res[j];
		bbox_t res_j;
		cv::Rect corr_r = get_rect(img, r.bbox);
		res_j.x = corr_r.x;
		res_j.y = corr_r.y;
		res_j.w = corr_r.width;
		res_j.h = corr_r.height;
		res_j.prob = r.conf;
		res_j.obj_id = r.class_id;
		container.candidates[j] = res_j;
	}
    return res_len;
}


bool com(const bbox_t a, const bbox_t b)
{
	return a.prob > b.prob;
}


int detect_image(const char* file_name, bbox_t_container &container){
 cv::Mat img = cv::imread(file_name);
    int res_len = detect_roi(img, container);
    return res_len;
}

int detect_mat(const uint8_t* mat, const size_t data_length, bbox_t_container &container){
    std::vector<char> vdata(mat, mat + data_length);
    cv::Mat img = imdecode(cv::Mat(vdata), 1);
    int res_len = detect_roi(img, container);
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



