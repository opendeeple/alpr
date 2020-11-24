#pragma once
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <opencv2/opencv.hpp>

namespace alpr {
    class Camera {
    private:
        int number_of_cameras;
        void start();
        void stop();
        void capture(int index);

    public:
        std::vector<std::string> sources;
        std::vector<std::thread*> threads;
        std::vector<cv::VideoCapture *> captures;
        std::vector<cv::Mat> frames;
        Camera(std::vector<std::string> sources);
        bool next(int index, cv::Mat &frame);
        ~Camera();
    };
};

alpr::Camera::Camera(std::vector<std::string> sources) {
    this->sources = sources;
    this->number_of_cameras = this->sources.size();
    this->start();
};

void alpr::Camera::capture(int index) {
    cv::VideoCapture *capture = this->captures[index];
    while(true) {
        cv::Mat frame;
        capture->read(frame);
        this->frames[index] = frame;
    }
}

void alpr::Camera::start() {
    cv::VideoCapture *capture;
    std::thread* stream;
    cv::Mat frame;
    for (int index = 0; index < this->number_of_cameras; index++) {
        std::string source = this->sources[index];
        capture = new cv::VideoCapture(source, cv::CAP_FFMPEG);
        std::cout << "Camera [" << index << "] has initialized: " << source << std::endl; 
        this->captures.push_back(capture);
        stream = new std::thread(&Camera::capture, this, index);
        this->threads.push_back(stream);
        frames.push_back(frame);
    }
}

void alpr::Camera::stop() {
    cv::VideoCapture *capture;
    for(int index = 0; index < this->number_of_cameras; index++) {
        capture = this->captures[index];
        if(capture->isOpened()) {
            capture->release();
            std::cout << "Camera [" << index << "] has released" << std::endl;
        }
    }
}

bool alpr::Camera::next(int index, cv::Mat &frame) {
    if(this->frames.at(index).empty()) {
        std::cout << "Camera[" << index << "] Frame is empty" << std::endl;
        return false;
    }
    // this->frames.at(index).copyTo(frame);
    frame = this->frames.at(index);
    return true;
}

alpr::Camera::~Camera() {
    this->stop();
}
