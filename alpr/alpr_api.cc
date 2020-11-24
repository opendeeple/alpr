#pragma once
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

#include "NvInfer.h"

namespace alpr {
    std::string sysmbols = "0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ";
    namespace detection {
        class Label {
        public:
            Label();
            Label(cv::Mat points, float prob);
            Label(cv::Mat left_top, cv::Mat right_bottom, float prob, int value);
            cv::Mat points, bottom_right, top_left, wh;
            float prob;
            int value;
        };

        struct Params {
            std::string path_to_model = "detection.uff";
            std::string input_tensor  = "Input";
            std::string output_tensor = "concatenate_1/concat";
            int batch_size = 1;
        };
    };

    size_t get_size_by_dim(const nvinfer1::Dims& dims);
    cv::Mat find_transform_matrix(cv::Mat matrix, cv::Mat transpose_matrix);
    std::vector<detection::Label> nms(std::vector<detection::Label> labels, float threshold);
    float iou(detection::Label label_1, detection::Label label_2);
};

alpr::detection::Label::Label() {
    this->prob = 0.0;
}

alpr::detection::Label::Label(cv::Mat points, float prob) {
    this->points = points;
    this->prob   = prob;
    cv::reduce(points, this->top_left, 1, cv::REDUCE_MIN, CV_32F);
    cv::reduce(points, this->bottom_right, 1, cv::REDUCE_MAX, CV_32F);
    this->wh = (cv::Mat)this->bottom_right - (cv::Mat)this->top_left;
}

alpr::detection::Label::Label(cv::Mat top_left, cv::Mat bottom_right, float prob, int value) {
    this->value = value;
    this->prob = prob;
    this->top_left = top_left;
    this->bottom_right = bottom_right;
    this->wh = (cv::Mat)this->bottom_right - (cv::Mat)this->top_left;
}

size_t alpr::get_size_by_dim(const nvinfer1::Dims& dims) {
    size_t size = 1;
    for (size_t i = 0; i < dims.nbDims; ++i) {
        size *= dims.d[i];
    }
    return size;
};

float alpr::iou(alpr::detection::Label label_1, alpr::detection::Label label_2) {
    cv::Mat intersection_wh = cv::max(
        cv::min(label_1.bottom_right, label_2.bottom_right) - 
        cv::max(label_1.top_left, label_2.top_left), (cv::Mat_<float>(2, 1) << 0.0, 0.0));
    
    float intersection_area = intersection_wh.at<float>(0) * intersection_wh.at<float>(1);
    float area_1            = label_1.wh.at<float>(0) * label_1.wh.at<float>(1);
    float area_2            = label_2.wh.at<float>(0) * label_2.wh.at<float>(1);
    float union_area        = area_1 + area_2 - intersection_area;
    return intersection_area / union_area;
}

std::vector<alpr::detection::Label> alpr::nms(std::vector<alpr::detection::Label> labels, float threshold) {
    std::vector<alpr::detection::Label> selections;
    for(std::vector<alpr::detection::Label>::iterator label_it = labels.begin(); label_it != labels.end(); label_it++) {
        bool is_overlay = true;
        for(std::vector<alpr::detection::Label>::iterator selection_it = selections.begin(); 
            selection_it != selections.end(); selection_it++) {
            if(alpr::iou(*label_it, *selection_it) > threshold) {
                is_overlay = false;
            }
        }
        if(is_overlay) selections.push_back(*label_it);
    }
    return selections;
};
