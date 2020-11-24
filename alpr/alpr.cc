#pragma once
#include <iostream>
#include <thread>
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
        int number_of_cameras;
    public:
        std::vector<Detection* > detections;
        std::vector<Recognation* > recognations;
        Alpr(int number_of_cameras);
        ~Alpr();
        void append(int index, cv::Mat &frame);
    };
};

void alpr::Alpr::append(int index, cv::Mat &frame) {
    this->frames.at(index) = frame;
}

void alpr::Alpr::start() {
    std::thread* stream;
    cv::Mat frame;
    for(int index = 0; index < this->number_of_cameras; index++) {
        stream = new std::thread(&Alpr::infer, this, index);
        this->threads.push_back(stream);
        frames.push_back(frame);
    }
}

void alpr::Alpr::infer(int index) {
    while(true) {
        if(this->frames.at(index).empty()) continue;
        std::pair<alpr::detection::Label, cv::Mat> detection;
        if(this->detections.at(index)->infer(this->frames.at(index), 0.5, &detection)) {
            std::string plate_number = this->recognations.at(index)->infer(detection.second);
            std::cout << "LP[" << index << "] " << plate_number << std::endl;
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
