#include <iostream>
#include <memory>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "camera.cc"
#include "alpr.cc"

int main() {
    std::vector<std::string> sources = {"in.avi", "out.avi"};
    alpr::Alpr app(sources.size());
    alpr::Camera camera(sources);

    while(cv::waitKey(20) != 27) {
        std::vector<cv::Mat> frames;
        std::vector<std::pair<alpr::detection::Label, cv::Mat>> detections;
        for(int index = 0; index < sources.size(); index++) {
            cv::Mat frame;
            if(camera.next(index, frame)) {
                cv::resize(frame, frame, cv::Size(917, 512));
                app.append(index, frame);
                cv::imshow("Life:" + std::to_string(index), frame);
            }
        }
    }

    return 0;
}
