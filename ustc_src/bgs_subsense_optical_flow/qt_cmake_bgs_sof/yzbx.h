#ifndef YZBX_H
#define YZBX_H
#include <opencv2/opencv.hpp>

void im2col(cv::Mat &input,cv::Mat col);
void lbsp(cv::Mat &input1,cv::Mat &input2,cv::Mat &lbspDistance);
void opticalFlow(cv::Mat &input1,cv::Mat &input2,cv::Mat &vx,cv::Mat &vy);
#endif // YZBX_H

