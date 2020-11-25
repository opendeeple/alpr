#include <iostream>
#include <memory>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "camera.cc"
#include "detection.cu"
#include "recognation.cc"
#include "alpr_api.cc"

int main() {
    alpr::detection::Params params;
    alpr::Detection detection(params);
    if(!detection.build()) return -1;

    alpr::Recognation recognation("recognation.cfg", "recognation.weights");

    std::vector<std::string> sources = {"in.avi", "out.avi"};
    alpr::Camera camera(sources);
    bool init = true;
    while(cv::waitKey(20) != 27) {
        std::vector<cv::Mat> frames;
        auto start = std::chrono::high_resolution_clock::now();
        for(int index = 0; index < sources.size(); index++) {
            cv::Mat frame;
            if(camera.next(index, frame)) {
                cv::resize(frame, frame, cv::Size(917, 512));
                std::pair<alpr::detection::Label, cv::Mat> prediction;
                if(detection.infer(frame, 0.5, &prediction)) {
                    std::string plate_number = recognation.infer(prediction.second);
                    std::cout << "LP[" << index << "] " << plate_number << std::endl;
                    for(size_t i = 0; i < 4; i++) {
                        cv::line(frame, cv::Point(prediction.first.points.col(i)), 
                            cv::Point(prediction.first.points.col((i + 1) % 4)), cv::Scalar(0, 0, 255), 2);
                    }
                    cv::Mat point; cv::reduce(prediction.first.points, point, 1, cv::REDUCE_MIN, CV_32F);
                    cv::putText(frame, plate_number, cv::Point(point), 
                        cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
                }
                frames.push_back(frame);
            }
        }
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "[STREAM] Performance: " << elapsed.count() << std::endl;
        if(frames.size() != 0) {
            cv::Mat win = alpr::merge_frames(1280, 720, frames);
            if(init) cv::namedWindow("Live", cv::WINDOW_NORMAL);
            cv::imshow("Live", win);
            init = false;
        }
    }
    return 0;
}
