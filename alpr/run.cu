#include <opencv2/opencv.hpp>
#include <memory>
#include <cstdlib>
#include "detection.cu"
#include "recognation.cc"
#include "alpr_api.cc"

int main() {
  alpr::detection::Params params;
  alpr::Detection detection(params);
  if(!detection.build()) return -1;
  alpr::Recognation recognation("recognation.cfg", "recognation.weights");
  cv::VideoCapture cap("in.avi");
  while(cv::waitKey(20) != 27) {
    cv::Mat frame, resized;
    cap >> frame;
    cv::resize(frame, frame, cv::Size(1280, 720));
    std::pair<alpr::detection::Label, cv::Mat> prediction;
    if(detection.infer(frame, 0.5, &prediction)) {
      std::string plate_number = recognation.infer(prediction.second);
      std::cout << "LP: " << plate_number << std::endl;
      for(size_t i = 0; i < 4; i++) {
        cv::line(frame, cv::Point(prediction.first.points.col(i)), 
          cv::Point(prediction.first.points.col((i + 1) % 4)), cv::Scalar(0, 0, 255), 2);
      }
      cv::Mat point; cv::reduce(prediction.first.points, point, 1, cv::REDUCE_MIN, CV_32F);
      cv::putText(frame, plate_number, cv::Point(point), 
        cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 0), 2);
    }
    cv::imshow("App", frame);
  }
  return 0;
}