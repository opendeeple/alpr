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
    bool init = true;
    while(cv::waitKey(20) != 27) {
        for(int index = 0; index < sources.size(); index++) {
            cv::Mat frame;
            if(camera.next(index, frame)) {
                cv::resize(frame, frame, cv::Size(917, 512));
                app.append(index, frame);
            }
        }
        cv::Mat win = app.frame(1280, 720);
        if(init) cv::namedWindow("Live", cv::WINDOW_NORMAL);
        cv::imshow("Live", win);
        init = false;
    }
    return 0;
}
