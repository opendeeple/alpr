#include <opencv2/opencv.hpp>
#include <memory>
#include <cstdlib>

int main() {
    cv::VideoCapture cap1("in.avi");

    while(cv::waitKey(20) != 27) {
        auto start = std::chrono::high_resolution_clock::now();
        cv::Mat frame1, frame2;
        cap1 >> frame1; cap2 >> frame2;
        cv::imshow("Life: 1", frame1);
        auto finish = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = finish - start;
        std::cout << "Performance: " << elapsed.count() << std::endl;
    }

    return 0;
}
