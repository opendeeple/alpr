#pragma once
#include <iostream>
#include <thread>
#include <math.h>
#include <opencv2/opencv.hpp>
#include "detection.cu"
#include "recognation.cc"

namespace alpr {
    class Alpr {
    private:
        void start();
        void stop();
        void infer(int index);
        std::vector<std::thread *> threads;
        std::vector<cv::Mat> frames;
        std::vector<cv::Mat> draws;
        int number_of_cameras;
    public:
        std::vector<Detection* > detections;
        std::vector<Recognation* > recognations;
        Alpr(int number_of_cameras);
        ~Alpr();
        void append(int index, cv::Mat &frame);
        cv::Mat frame(int width, int height);
    };
};

cv::Mat alpr::Alpr::frame(int width, int height) {
    return alpr::merge_frames(width, height, this->draws);
}

void alpr::Alpr::append(int index, cv::Mat &frame) {
    this->frames.at(index) = frame;
}

void alpr::Alpr::start() {
    std::thread* stream;
    cv::Mat frame;
    for(int index = 0; index < this->number_of_cameras; index++) {
        stream = new std::thread(&Alpr::infer, this, index);
        this->threads.push_back(stream);
        this->frames.push_back(frame);
        this->draws.push_back(frame);
    }
}

void alpr::Alpr::infer(int index) {
    while(true) {
        if(this->frames.at(index).empty()) continue;
        std::pair<alpr::detection::Label, cv::Mat> detection;
        cv::Mat frame = this->frames.at(index).clone();
        if(this->detections.at(index)->infer(frame, 0.5, &detection)) {
            std::string plate_number = this->recognations.at(index)->infer(detection.second);
            std::cout << "LP[" << index << "] " << plate_number << std::endl;
            for(size_t i = 0; i < 4; i++) {
                cv::line(frame, cv::Point(detection.first.points.col(i)), 
                    cv::Point(detection.first.points.col((i + 1) % 4)), cv::Scalar(0, 0, 255), 2);
            }
            cv::Mat point; cv::reduce(detection.first.points, point, 1, cv::REDUCE_MIN, CV_32F);
            cv::putText(frame, plate_number, cv::Point(point), 
                cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
            this->draws.at(index) = frame;
        }
    }
}

alpr::Alpr::Alpr(int number_of_cameras) {
    alpr::detection::Params params;
    for(int index = 0; index < number_of_cameras; index++) {
        this->recognations.push_back(new alpr::Recognation("recognation.cfg", "recognation.weights"));
        this->detections.push_back(new alpr::Detection(params));
        if(!this->detections.at(index)->build()) {
            std::cout << "Could not build model" << std::endl;
            return;
        }
    }

    this->number_of_cameras = number_of_cameras;
    this->start();
}

alpr::Alpr::~Alpr() {
}
