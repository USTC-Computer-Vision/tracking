#include "bgs_sof.h"

BGS_SOF::BGS_SOF() : firstTime(true), enableThreshold(true), threshold(15), showOutput(true)
{
  std::cout << "BGS_SOF()" << std::endl;
}

BGS_SOF::~BGS_SOF()
{
  std::cout << "~BGS_SOF()" << std::endl;
}

void BGS_SOF::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
  if(img_input.empty())
    return;

  loadConfig();

  if(firstTime)
    saveConfig();

  if(img_input_prev.empty())
  {
    img_input.copyTo(img_input_prev);
    return;
  }

  cv::absdiff(img_input_prev, img_input, img_foreground);

  if(img_foreground.channels() == 3)
    cv::cvtColor(img_foreground, img_foreground, CV_BGR2GRAY);

  if(enableThreshold)
    cv::threshold(img_foreground, img_foreground, threshold, 255, cv::THRESH_BINARY);

  if(showOutput)
    cv::imshow("Frame Difference", img_foreground);

  img_foreground.copyTo(img_output);

  img_input.copyTo(img_input_prev);

  firstTime = false;
}

void BGS_SOF::saveConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/BGS_SOF.xml", 0, CV_STORAGE_WRITE);

  cvWriteInt(fs, "enableThreshold", enableThreshold);
  cvWriteInt(fs, "threshold", threshold);
  cvWriteInt(fs, "showOutput", showOutput);

  cvReleaseFileStorage(&fs);
}

void BGS_SOF::loadConfig()
{
  CvFileStorage* fs = cvOpenFileStorage("./config/BGS_SOF.xml", 0, CV_STORAGE_READ);

  enableThreshold = cvReadIntByName(fs, 0, "enableThreshold", true);
  threshold = cvReadIntByName(fs, 0, "threshold", 15);
  showOutput = cvReadIntByName(fs, 0, "showOutput", true);

  cvReleaseFileStorage(&fs);
}
