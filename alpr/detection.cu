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
    };
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
        void preprocessing(cv::cuda::GpuMat image, float* input_buffer, const nvinfer1::Dims& dims);
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

bool alpr::Detection::build(){
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

    std::fstream file("detection.engine", std::ios::in | std::ios::out | std::ios::binary | std::ios::trunc);
    if(file.is_open()) {
        // TODO: restore the engine
        file.close();
    }
    else {
        std::cout << "[Detection] Could not store engine" << std::endl;
    }

    this->context = unique_ptr<nvinfer1::IExecutionContext>(this->engine->createExecutionContext());
    if(!this->context) {
        std::cerr << "[Detection] Could not create engine context" << std::endl;
        return false;
    }
    return true;
};

bool alpr::Detection::infer(cv::Mat &image, float threshold, std::pair<alpr::detection::Label, cv::Mat>* output) {
    std::vector<void*> buffers(this->engine->getNbBindings());
    std::vector<nvinfer1::Dims> input_dims;
    std::vector<nvinfer1::Dims> output_dims;
    for (size_t i = 0; i < this->engine->getNbBindings(); ++i) {
        auto binding_size = alpr::get_size_by_dim(engine->getBindingDimensions(i)) * this->params.batch_size * sizeof(float);
        cudaMalloc(&buffers[i], binding_size);
        if(engine->bindingIsInput(i)) input_dims.emplace_back(engine->getBindingDimensions(i));
        else output_dims.emplace_back(engine->getBindingDimensions(i));
    }
    if (input_dims.empty() || output_dims.empty()) {
        std::cerr << "[Detection] Expect at least one input and one output for network" << std::endl;
        return false;
    }
    try{
        cv::cuda::GpuMat gpu_image;
        gpu_image.upload(image);
        this->preprocessing(gpu_image, (float *) buffers[0], input_dims[0]);
        gpu_image.release();
        this->context->enqueue(this->params.batch_size, buffers.data(), 0, nullptr);
        std::vector<alpr::detection::Label> labels = this->filter_detection(cv::Size(input_dims[0].d[2], input_dims[0].d[1]), (float *) buffers[1], 
            output_dims[0], threshold, this->params.batch_size);
        for (void* buffer : buffers) cudaFree(buffer);
        std::vector<alpr::detection::Label> selections = alpr::nms(labels, threshold);
        if(selections.size() == 0) return false;

        alpr::detection::Label result = selections.at(0);

        cv::Size size = image.size();
        result.points.row(0) *= size.height;
        result.points.row(1) *= size.width;

        output->first  = result;
        output->second = this->extract_plate(image, result.points.clone(), cv::Size(240, 80));

        return true;
    } catch(cv::Exception &e) {
        return false;
    }
};

void alpr::Detection::preprocessing(cv::cuda::GpuMat image, float* input_buffer, const nvinfer1::Dims& dims) {
    cv::cuda::GpuMat resized;
    cv::cuda::resize(image, resized, cv::Size(dims.d[2], dims.d[1]), 0, 0, cv::INTER_NEAREST);
    cv::cuda::GpuMat scaled_image;
    resized.convertTo(scaled_image, CV_32F, 1.f / 255);
    std::vector<cv::cuda::GpuMat> chw;
    for (size_t i = 0; i < dims.d[0]; ++i) {
        chw.emplace_back(cv::cuda::GpuMat(cv::Size(dims.d[2], dims.d[1]), CV_32FC1, input_buffer + i * dims.d[2] * dims.d[1]));
    }
    cv::cuda::split(scaled_image, chw);
};

std::vector<alpr::detection::Label> alpr::Detection::filter_detection(cv::Size size, float* output_buffer, 
    const nvinfer1::Dims& dims, float threshold, const int batch_size) {
    float side = 7.75;
    std::vector<float> cpu_output(alpr::get_size_by_dim(dims) * batch_size);
    cudaMemcpy(cpu_output.data(), output_buffer, cpu_output.size() * sizeof(float), cudaMemcpyDeviceToHost);
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
