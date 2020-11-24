#pragma once
#include <string>
#include <opencv2/opencv.hpp>
#include "alpr_api.cc"

namespace alpr {
    class Recognation {
    private:
        cv::dnn::Net net;
        std::string filter_recognation(cv::Mat prediction);
    public:
        Recognation(std::string path_to_model, std::string path_to_weights);
        std::string infer(cv::Mat plate) {
            cv::Mat blob, prediction;
            cv::dnn::blobFromImage(plate, blob, 1.0 / 255.0);
            net.setInput(blob);
            net.forward(prediction);
            return this->filter_recognation(prediction);
        }    
    };
};

alpr::Recognation::Recognation(std::string path_to_model, std::string path_to_weights) {
    this->net = cv::dnn::readNetFromDarknet(path_to_model, path_to_weights);
    this->net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    this->net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

std::string alpr::Recognation::filter_recognation(cv::Mat prediction) {
    cv::Mat scores = prediction(cv::Range::all(), cv::Range(5, 40));
    std::vector<alpr::detection::Label> labels;
    for(size_t i = 0; i < prediction.size().height; i++) {
        double min, max;
        cv::Point min_index, max_index;
        cv::minMaxLoc(scores.row(i), &min, &max, &min_index, &max_index);
        float score = scores.row(i).at<float>(max_index);
        if(score > 0.45) {
            float center_x = (float) prediction.row(i).at<float>(0);
            float center_y = (float) prediction.row(i).at<float>(1);
            float width    = (float) prediction.row(i).at<float>(2);
            float height   = (float) prediction.row(i).at<float>(3);
            float left     = (float) (center_x - width  / 2);
            float top      = (float) (center_y - height / 2);
            float right    = (float) (center_x + width  / 2);
            float bottom   = (float) (center_y + height / 2);
            cv::Mat top_left     = (cv::Mat_<float>(2, 1) << top, left);
            cv::Mat bottom_right = (cv::Mat_<float>(2, 1) << bottom, right);
            alpr::detection::Label label(top_left, bottom_right, score, (int)max_index.x);
            labels.push_back(label);
        }
    }
    std::sort(labels.begin(), labels.end(), [](alpr::detection::Label label_1, alpr::detection::Label label_2) -> bool {
        return label_1.prob > label_2.prob;
    });
    std::vector<alpr::detection::Label> selections = alpr::nms(labels, 0.45);
    std::sort(selections.begin(), selections.end(), [](alpr::detection::Label label_1, alpr::detection::Label label_2) -> bool {
        return label_1.top_left.at<float>(1) < label_2.top_left.at<float>(1);
    });
    std::string result = "";
    for(std::vector<alpr::detection::Label>::iterator it = selections.begin(); it != selections.end(); it++){
        result += alpr::sysmbols[(*it).value];
    }
    return result;
}
