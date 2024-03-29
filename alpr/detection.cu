#include <chrono> 
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudawarping.hpp>
#include <opencv2/cudaarithm.hpp>

#include "NvInfer.h"
#include "NvUffParser.h"
#include "alpr_api.cc"
#include "buffer.cu"

using Severity = nvinfer1::ILogger::Severity;

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) override {
        if((severity == Severity::kERROR) || (severity == Severity::kINTERNAL_ERROR)) std::cout << msg << std::endl;
    }
} gLogger;

struct TRTDestroy {
    template<typename T>
    void operator()(T* object) const {
        if(object) object->destroy();
    }
};

template<typename T> using unique_ptr = std::unique_ptr<T, TRTDestroy>;

namespace alpr {
    class Detection {
    private:
        std::shared_ptr<nvinfer1::ICudaEngine> engine;
        unique_ptr<nvinfer1::IExecutionContext> context;        
        detection::Params params;
        bool build_engine(
            unique_ptr<nvinfer1::IBuilder>& builder,
            unique_ptr<nvinfer1::INetworkDefinition>& network,
            unique_ptr<nvinfer1::IBuilderConfig>& config,
            unique_ptr<nvuffparser::IUffParser>& parser
        );
        bool preprocessing(cv::Mat image, BufferManager &buffer, const nvinfer1::Dims& dims);
        std::vector<alpr::detection::Label> filter_detection(cv::Size size, float* output_buffer, 
            const nvinfer1::Dims& dims, float threshold, const int batch_size);
        cv::Mat extract_plate(cv::Mat image, cv::Mat points, cv::Size size);
    public:
        Detection(detection::Params params);
        bool build();
        bool infer(cv::Mat &image, float threshold, std::pair<detection::Label, cv::Mat>* output);
    };
};

alpr::Detection::Detection(alpr::detection::Params params) {
    this->params = params;
};

struct InferDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj)
        {
            obj->destroy();
        }
    }
};

bool alpr::Detection::build(){
    {
        std::vector<char> trtModelStream;
        size_t size{0};
        std::ifstream file("detection.engine", std::ios::binary);
        if (file.good()) {
            file.seekg(0, file.end);
            size = file.tellg();
            file.seekg(0, file.beg);
            trtModelStream.resize(size);
            file.read(trtModelStream.data(), size);
            file.close();
        }
        nvinfer1::IRuntime* infer = nvinfer1::createInferRuntime(gLogger);
        this->engine = std::shared_ptr<nvinfer1::ICudaEngine>(
            infer->deserializeCudaEngine(trtModelStream.data(), size, nullptr), InferDeleter());
        if(this->engine) {
            this->context = unique_ptr<nvinfer1::IExecutionContext>(this->engine->createExecutionContext());
            if(!this->context) {
                std::cerr << "[Detection] Could not create engine context" << std::endl;
                return false;
            }
            return true;
        }
    }
    auto builder = unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if(!builder) {
        std::cerr << "[Detection] Could not create a builder" << std::endl;
        return false;
    }
    auto network = unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetwork());
    if(!network) {
        std::cerr << "[Detection] Could not create a network" << std::endl;
        return false;
    }
    auto config = unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config) {
        std::cerr << "[Detection] Could not create a builder config" << std::endl;
        return false;
    }
    auto parser = unique_ptr<nvuffparser::IUffParser>(nvuffparser::createUffParser());
    if(!parser) {
        std::cerr << "[Detection] Could not create a uff parser" << std::endl;
        return false;
    }
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    return this->build_engine(builder, network, config, parser);
};

bool alpr::Detection::build_engine(
    unique_ptr<nvinfer1::IBuilder>& builder,
    unique_ptr<nvinfer1::INetworkDefinition>& network,
    unique_ptr<nvinfer1::IBuilderConfig>& config,
    unique_ptr<nvuffparser::IUffParser>& parser) {

    parser->registerInput(this->params.input_tensor.c_str(), nvinfer1::DimsCHW(3, 512, 912), nvuffparser::UffInputOrder::kNCHW);
    parser->registerOutput(this->params.output_tensor.c_str());

    auto is_parsed = parser->parse(this->params.path_to_model.c_str(), *network, nvinfer1::DataType::kFLOAT);
    if(!is_parsed) {
        std::cerr << "[Detection] Could not parse '" << this->params.path_to_model.c_str() << "' model" << std::endl;
        return false; 
    }

    builder->setMaxBatchSize(this->params.batch_size);
    config->setMaxWorkspaceSize(1 << 30);
    this->engine = std::shared_ptr<nvinfer1::ICudaEngine>(builder->buildEngineWithConfig(*network, *config), TRTDestroy());
    if(!this->engine) {
        std::cerr << "[Detection] Could not build engine" << std::endl;
        return false;
    }

    std::ofstream __file("detection.engine", std::ios::binary);
    if(!__file) return false;
    nvinfer1::IHostMemory* memory = this->engine->serialize(); assert(memory);
    __file.write(reinterpret_cast<const char*>(memory->data()), memory->size());
    memory->destroy();
    __file.close();
    std::cout << "Engine has restored" << std::endl;

    this->context = unique_ptr<nvinfer1::IExecutionContext>(this->engine->createExecutionContext());
    if(!this->context) {
        std::cerr << "[Detection] Could not create engine context" << std::endl;
        return false;
    }
    return true;
};

bool alpr::Detection::preprocessing(cv::Mat image, BufferManager &buffers, const nvinfer1::Dims& dims) {
    cv::resize(image, image, cv::Size(dims.d[2], dims.d[1]), cv::INTER_CUBIC);
    float* buffer = static_cast<float*>(buffers.getHostBuffer(this->params.input_tensor));
    for(int i = 0, volume_image = dims.d[0] * dims.d[1] * dims.d[2]; i < this->params.batch_size; ++i) {
        for (unsigned j = 0, volume_c = dims.d[1] * dims.d[2]; j < volume_c; ++j) {
            buffer[i * volume_image + 0 * volume_c + j] = float(image.data[j * dims.d[0] + 2 - 0]) / 255.0;
            buffer[i * volume_image + 1 * volume_c + j] = float(image.data[j * dims.d[0] + 2 - 1]) / 255.0;
            buffer[i * volume_image + 2 * volume_c + j] = float(image.data[j * dims.d[0] + 2 - 2]) / 255.0;
        }
    }
    return true;
}

bool alpr::Detection::infer(cv::Mat &image, float threshold, std::pair<alpr::detection::Label, cv::Mat>* output) {
    BufferManager buffers(this->engine, this->params.batch_size);
    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;
    for (size_t i = 0; i < this->engine->getNbBindings(); ++i) {
        if(engine->bindingIsInput(i)) input_dims.emplace_back(engine->getBindingDimensions(i));
        else output_dims.emplace_back(engine->getBindingDimensions(i));
    }
    if (input_dims.empty() || output_dims.empty()) {
        std::cerr << "[Detection] Expect at least one input and one output for network" << std::endl;
        return false;
    }
    if(!this->preprocessing(image, buffers, input_dims[0])) {
        return false;
    }
    buffers.copyInputToDevice();
    bool status = this->context->enqueue(this->params.batch_size, buffers.getDeviceBindings().data(), 0, nullptr);
    if(!status) {
        std::cout << "Model is not infered successfully" << std::endl;
    }
    buffers.copyOutputToHost();
    float* output_buffer = static_cast<float*>(buffers.getHostBuffer(this->params.output_tensor));
    std::vector<alpr::detection::Label> labels = this->filter_detection(cv::Size(input_dims[0].d[2], input_dims[0].d[1]), output_buffer, 
            output_dims[0], threshold, this->params.batch_size);
    std::vector<alpr::detection::Label> selections = alpr::nms(labels, threshold);
    if(selections.size() == 0) return false;

    alpr::detection::Label result = selections.at(0);
    cv::Size size = image.size();
    result.points.row(0) *= size.height;
    result.points.row(1) *= size.width;
    output->first  = result;
    output->second = this->extract_plate(image, result.points.clone(), cv::Size(240, 80));
    return true;
};

std::vector<alpr::detection::Label> alpr::Detection::filter_detection(cv::Size size, float* output_buffer, 
    const nvinfer1::Dims& dims, float threshold, const int batch_size) {
    float side = 7.75;
    std::vector<float> cpu_output(output_buffer, output_buffer + alpr::get_size_by_dim(dims) * batch_size);
    cv::Mat transpose_matrix = (cv::Mat_<float>(3, 4) << -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 1.0);
    std::vector<alpr::detection::Label> labels;
    size_t index = 0;
    for(size_t batch = 1; batch <= batch_size; batch++) {
        for(size_t x = 0; x < dims.d[0]; x++) {
            for(size_t y = 0; y < dims.d[1]; y++) {
                float prob = cpu_output.at(index);
                float data[2][3] = {
                    {cpu_output.at(index + 2), cpu_output.at(index + 3), cpu_output.at(index + 4)},
                    {cpu_output.at(index + 5), cpu_output.at(index + 6), cpu_output.at(index + 7)}
                };
                index += 8;
                if(prob <= threshold) continue;
                cv::Mat affine = cv::Mat(2, 3, CV_32F, data);
                affine.at<float>(0, 0) = max(affine.at<float>(0, 0), 0.0);
                affine.at<float>(1, 1) = max(affine.at<float>(1, 1), 0.0);
                cv::Mat points = affine * transpose_matrix * side;
                points.row(0) += float(y) + 0.5;
                points.row(1) += float(x) + 0.5;
                points.row(0) /= 1.0 * size.height / std::pow(2.0, 4);
                points.row(1) /= 1.0 * size.width / std::pow(2.0, 4);
                alpr::detection::Label label(points, prob);
                labels.push_back(label);
            }
        }
    }

    std::sort(labels.begin(), labels.end(), [](alpr::detection::Label label_1, alpr::detection::Label label_2) -> bool {
        return label_1.prob > label_2.prob;
    });
    return labels;
};

cv::Mat alpr::Detection::extract_plate(cv::Mat image, cv::Mat points, cv::Size output_size) {
    cv::Mat matrix; cv::vconcat(points, cv::Mat::ones(1, 4, points.type()), matrix);
    cv::Mat transpose_matrix = (cv::Mat_<float>(3, 4) << 
        0.0, output_size.width, output_size.width, 0.0,
        0.0, 0.0, output_size.height, output_size.height, 
        1.0, 1.0, 1.0, 1.0
    );
    cv::Mat affine = cv::Mat::zeros(8, 9, points.type());
    
    for(size_t i = 0; i < 4; i++) {
        cv::Mat x = matrix.col(i).t();
        cv::Mat transpose = transpose_matrix.col(i);
        cv::Mat v1 = -1.0 * transpose.at<float>(2) * x;
        cv::Mat v2 = transpose.at<float>(1) * x;
        cv::Mat v3 = transpose.at<float>(2) * x;
        cv::Mat v4 = -1.0 * transpose.at<float>(0) * x;
        cv::Mat corr_row = (cv::Mat_<float>(1, 9) << 
            0.0, 0.0, 0.0,
            v1.at<float>(0), v1.at<float>(1), v1.at<float>(2),
            v2.at<float>(0), v2.at<float>(1), v2.at<float>(2)
        );
        cv::Mat next_row = (cv::Mat_<float>(1, 9) << 
            v3.at<float>(0), v3.at<float>(1), v3.at<float>(2),
            0.0, 0.0, 0.0,
            v4.at<float>(0), v4.at<float>(1), v4.at<float>(2)
        );
        affine.row(i * 2)     += corr_row;
        affine.row(i * 2 + 1) += next_row;
    }

    cv::SVD svd = cv::SVD(affine, 4);
    cv::Mat transform;
    svd.vt.row(svd.vt.size().height - 1).reshape(0, 3).assignTo(transform, CV_64F);
    cv::Mat plate;
    cv::warpPerspective(image, plate, transform, output_size, 1, 0.0);
    cv::cvtColor(plate, plate, cv::COLOR_BGR2GRAY);
    cv::cvtColor(plate, plate, cv::COLOR_GRAY2BGR);
    return plate;
}
