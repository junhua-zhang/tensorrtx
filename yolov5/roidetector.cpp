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


void decode(const uint8_t* mat, int* start_list, int index, std::vector<cv::Mat> & img_vec)
{
		std::vector<char> vdata(mat + start_list[index], mat + start_list[index+1]);
		cv::Mat img = imdecode(cv::Mat(vdata), 1);
		img_vec[index] = img;
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

int detect_roi(std::vector<cv::Mat> &img_vec, bbox_t_container* container_vec)
{
    assert(!img_vec.empty());
	int ftotal = img_vec.size();
	int fcount = 0;

	static float data[BATCH_SIZE * 3 * INPUT_H * INPUT_W];
	static float prob[BATCH_SIZE * OUTPUT_SIZE];

	auto start = std::chrono::system_clock::now();
	for (int f = 0; f < ftotal; f++)
	{
		fcount++;
		if (fcount < BATCH_SIZE && f + 1 != ftotal)
			continue;
		//std::cout << fcount << " started" << std::endl;
		for (int b = 0; b < fcount; b++)
		{
			cv::Mat img = img_vec[f-fcount + 1 + b];
			cv::Mat pr_img = preprocess_img(img);
			int i = 0;
			for (int row = 0; row < INPUT_H; ++row) {
				uchar* uc_pixel = pr_img.data + row * pr_img.step;
				for (int col = 0; col < INPUT_W; ++col) {
					int start = b * 3 * INPUT_H * INPUT_W;
					data[start + i] = (float)uc_pixel[2] / 255.0;
					data[start + i + INPUT_H * INPUT_W] = (float)uc_pixel[1] / 255.0;
					data[start + i + 2 * INPUT_H * INPUT_W] = (float)uc_pixel[0] / 255.0;
					uc_pixel += 3;
					++i;
				}
			}
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "preprocess: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

		// Run inference
		start = std::chrono::system_clock::now();
		doInference(*context, stream, buffers, data, prob, BATCH_SIZE);
		end = std::chrono::system_clock::now();
		std::cout << "inference: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

		start = std::chrono::system_clock::now();
		std::vector<std::vector<Yolo::Detection>> batch_res(fcount);
		for (int b = 0; b < fcount; b++) {
			auto& res = batch_res[b];
			nms_id(res, &prob[b*OUTPUT_SIZE], CONF_THRESH, NMS_THRESH);

			int res_len = res.size();
			//std::cout << b <<  ": everything cool: " << res_len << std::endl;
			for (size_t j = 0; j < res_len; j++) {
				auto r = res[j];
				bbox_t res_j;
				cv::Rect corr_r = get_rect(img_vec[f-fcount+1+b], r.bbox);
				res_j.x = corr_r.x > 0 ? corr_r.x : 0;
				res_j.y = corr_r.y > 0 ? corr_r.y : 0;
				res_j.w = corr_r.width > 0 ? corr_r.width : 0;
				res_j.h = corr_r.height > 0 ? corr_r.height : 0;
				res_j.prob = r.conf;
				res_j.obj_id = r.class_id;
				//std::cout << res_j.x << "," << res_j.y << "," << res_j.w << "," << res_j.h << std::endl;

				container_vec[f-fcount+1+b].candidates[j] = res_j;
			}
			//std::cout << b << "," <<f-fcount+1+b<<  ": everything cool" << std::endl;
		}
		end = std::chrono::system_clock::now();
		std::cout << "nms: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
		fcount = 0;
	}

    return 1;
}


bool com(const bbox_t a, const bbox_t b)
{
	return a.prob > b.prob;
}


int detect_image(const char* root_dir, const char* object_name, bbox_t_container* container){
	std::vector<cv::Mat> img_vec(TOTAL_ANGLE);
	std::string top_name = std::string(root_dir) + "/" + object_name + "/" + object_name + "_top.jpg";
    cv::Mat top = cv::imread(top_name);
	img_vec[0] = top; 
	for (int i = 0; i < TOTAL_ANGLE - 1; i++)
	{
		std::string img_name;
		if (i < 10)
			img_name = std::string(root_dir) + "/" + object_name + "/" + object_name + "_x0" + std::to_string(i) + ".jpg";
		else
			img_name = std::string(root_dir) + "/" + object_name + "/" + object_name + "_x" + std::to_string(i) + ".jpg";
		cv::Mat img = cv::imread(img_name);
		img_vec[i + 1] = img;
	}

    int res_len = detect_roi(img_vec, container);
	for (int b = 0; b < TOTAL_ANGLE; b++)
	{
		std::sort(container[b].candidates, container[b].candidates+20, com);
		for each (auto res in container[b].candidates)
		{
			cv::Rect r = cv::Rect(res.x, res.y, res.w, res.h);
			cv::rectangle(img_vec[b], r, cv::Scalar(0x27, 0xC1, 0x36), 2);
			cv::putText(img_vec[b], std::to_string((int)res.obj_id), cv::Point(res.x, res.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
		}
		//cv::Mat patch = cv::Mat(img_vec[b], r);
		std::string out_name;
		if (b == 0)
			out_name = "res/" + std::string(object_name) + "_top.jpg";
		else if( b < 11)
			out_name = "res/" + std::string(object_name) + "_x0" + std::to_string(b-1) + ".jpg";
		else
			out_name = "res/" + std::string(object_name) + "_x" + std::to_string(b-1) + ".jpg";


		cv::imwrite(out_name, img_vec[b]);
	}
    return res_len;
}

int detect_mat(const uint8_t* mat, const int* data_length, bbox_t_container* container){
	std::vector<cv::Mat> img_vec(TOTAL_ANGLE);
	int* start_list = new int[TOTAL_ANGLE + 1];
	start_list[0] = 0;
	for (int b = 0; b < TOTAL_ANGLE; b++)
	{
		start_list[b+1] = start_list[b] + data_length[b];
	}

	std::thread thread_list[TOTAL_ANGLE];
	//auto start_time = std::chrono::system_clock::now();
	for (int b = 0; b < TOTAL_ANGLE; b++)
	{
		//std::vector<char> vdata(mat + start_list[b], mat + start_list[b+1]);
		////auto start_each = std::chrono::system_clock::now();
		//cv::Mat img = imdecode(cv::Mat(vdata), 1);
		////auto end_each = std::chrono::system_clock::now();
		////std::cout << "prepare decode: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_each - start_each).count() << "ms" << std::endl;
		//img_vec[b] = img;
		thread_list[b] = std::thread(decode, mat, start_list, b, img_vec);
	}
	for (int b = 0; b < TOTAL_ANGLE; b++)
	{
		thread_list[b].join();
	}
	//auto end = std::chrono::system_clock::now();
	//std::cout << "prepare decode: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time).count() << "ms" << std::endl;

    int res_len = detect_roi(img_vec, container);

	//start_time = std::chrono::system_clock::now();

	for (int b = 0; b < TOTAL_ANGLE; b++)
	{
		std::sort(container[b].candidates, container[b].candidates+20, com);
		
		for each(auto res in container[b].candidates)
		{
			cv::Rect r = cv::Rect(res.x, res.y, res.w, res.h);
			//cv::rectangle(img_vec[b], r, cv::Scalar(0x27, 0xC1, 0x36), 2);
			//cv::putText(img_vec[b], std::to_string((int)res.obj_id), cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
			////cv::Mat patch = cv::Mat(img_vec[b], r);
			//std::string out_name;
			//if (b == 0)
			//	out_name =  out_dir + "/" + "_top.jpg";
			//else if( b < 11)
			//	out_name = out_dir + "/" + "_x0" + std::to_string(b-1) + ".jpg";
			//else
			//	out_name = out_dir + "/" + "_x" + std::to_string(b-1) + ".jpg";
			//cv::imwrite(out_name, img_vec[b]);
		}
	}
	//end = std::chrono::system_clock::now();
	//std::cout << "sort result: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start_time).count() << "ms" << std::endl;
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



