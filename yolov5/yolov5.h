#ifndef _YOLO_V5_H
#define _YOLO_V5_H

#include <iostream>
#include <chrono>
#include "cuda_runtime_api.h"
#include "logging.h"
#include "common.h"

#define NMS_THRESH 0.3
#define CONF_THRESH 0.5
#define BATCH_SIZE 1

// stuff we know about the network and the input/output blobs
static const int INPUT_H = Yolo::INPUT_H;
static const int INPUT_W = Yolo::INPUT_W;
//static const int CLASS_NUM = Yolo::CLASS_NUM;
static const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * sizeof(Yolo::Detection) / sizeof(float) + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
static const char* INPUT_BLOB_NAME = "data";
static const char* OUTPUT_BLOB_NAME = "prob";
static Logger gLogger;

//#define NET s  // s m l x
//#define NETSTRUCT(str) createEngine_##str
//#define CREATENET(net) NETSTRUCT(net)
//#define STR1(x) #x
//#define STR2(x) STR1(x)

// Creat the engine using only the API and not any parser.
ICudaEngine* build_engine(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt, float& gd, float& gw, std::string& wts_name, int class_num);

//ICudaEngine* createEngine_s(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
//
//ICudaEngine* createEngine_m(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
//
//ICudaEngine* createEngine_l(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);
//
//ICudaEngine* createEngine_x(unsigned int maxBatchSize, IBuilder* builder, IBuilderConfig* config, DataType dt);

void APIToModel(unsigned int maxBatchSize, IHostMemory** modelStream, float& gd, float& gw, std::string& wts_name, int class_num);

void doInference(IExecutionContext& context, cudaStream_t& stream, void **buffers, float* input, float* output, int batchSize);

void parse_net(const char* net, float& gd, float& gw);

#endif
