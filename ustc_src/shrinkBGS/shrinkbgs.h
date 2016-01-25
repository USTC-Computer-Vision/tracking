#ifndef SHRINKBGS_H
#define SHRINKBGS_H
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.hpp>

#include "../../package_bgs/IBGS.h"
#include "../../package_bgs/pl/RandUtils.h"
#include "../../package_bgs/pl/DistanceUtils.h"

#define _DEBUG 1

#if _DEBUG
#define LOG_MESSAGE(x) std::cout << __FILE__ << " (" << __LINE__ << "): " << x << std::endl;
#define LOG_MESSAGE_POINT(x) if(i==0&&j==0){ \
    std::cout << __FILE__ << " (" << __LINE__ << "): " << x << std::endl; }
#else
#define LOG_MESSAGE(x)
#define LOG_MESSAGE_POINT(x)
#endif

using namespace cv;
class shrinkBGS : public IBGS
{
public:
    shrinkBGS();
    void process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel);
    void setRoi(const cv::Mat &img_input_roi);
private:
    void init();
    void refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate=false);
    void saveConfig();
    void loadConfig();
    void updateBound();
    bool learnStepCheck(Vec3b anCurrColor,size_t x,size_t y,Vec3b learnStep);

    //update and detect
    void getRawForegroundMask();
    void getPureForegroundMask();
    void getWeightedForegroundMask();
    void updateBackground();
    void updateForegroundAsBackground();
    void updateDistanceThreshold();
    void updateWeightedDistanceThreshold();

    //debug
    void debug();
    void drawHist(cv::vector<Vec3b> models);
public:
    //start from 0 and init with 0;
    int frameNum;
    //image size
    int img_cols,img_rows;
    cv::Size img_size;

    //local parameter
    cv::Mat img_background;
    cv::Mat img_foreground;
    cv::Mat img_rawForegroundMask,img_pureForegroundMask;
    cv::Mat img_roi,input;
    cv::Mat img_upperBound8UC3,img_lowerBound8UC3;
    cv::Mat img_distanceThreshold32F;
    cv::Mat img_neighborSpreadNum;
    cv::Mat img_backgroundLearnStep;
    cv::Mat img_backgroundLearningRateNum;
    cv::Mat img_weightedRawForegroundMask,img_weightedPureForegroundMask;
    cv::Mat img_weightedDistanceThreshold32F;
    cv::Mat img_distanceWeight32FC3;
    cv::Mat img_Dmin32F,img_weightedDmin32F;

    std::vector<cv::Mat> vec_BGColorSample;
    std::vector<cv::Mat> vec_BGDescSample;

    //global adaptive parameter
    float distance_learningRate=0.05,weightedDistance_learningRate=0.05;

    //const
    const int SampleNum=30;
    const int boundUpdateCycle=SampleNum;
    const int m_nRequiredBGSamples=2;
    //foregroundAcceptRate=1/foregroundAcceptNum;
    const int foregroundAcceptNum=2;
    Vec3b L1Threshold;

    //debug
    Mat img_TotalSumDist;
    Mat img_label;

    //unusing variable
    bool enableThreshold;
    int threshold;
    bool showOutput;
};

bool L1Check(Vec3b input,Vec3b model,Vec3b threshold){
    for(int i=0;i<3;i++){
        if(input[i]<model[i]-threshold[i])
            return false;
        else if(input[i]>model[i]+threshold[i])
            return false;
    }
    return true;
}

void img_cross(Mat &a,Mat &b,Mat &ret){
    std::cout<<"a: "<<a.size()<<" "<<a.type()<<std::endl;
    std::cout<<"b: "<<b.size()<<" "<<b.type()<<std::endl;
    cv::vector<Mat> mats1,mats2;
    split(a,mats1);
    split(b,mats2);
    Mat m;
    cv::vector<Mat> mats3;
    m=mats1[1].mul(mats2[2])-mats1[2].mul(mats2[1]);
    std::cout<<"m: "<<m.size()<<" "<<m.type()<<std::endl;

    mats3.push_back(m);
    m=-(mats1[0].mul(mats2[2])-mats1[2].mul(mats2[0]));
    mats3.push_back(m);
    m=mats1[0].mul(mats2[1])-mats1[1].mul(mats2[0]);
    mats3.push_back(m);

    merge(mats3,ret);
    std::cout<<"ret: "<<ret.size()<<" "<<ret.type()<<std::endl;
//    return ret;
}

void img_show(string str,Mat src){
    double maxVal,minVal;
    minMaxIdx(src,&minVal,&maxVal);
    Mat des=(src-minVal)/(maxVal-minVal);
    imshow(str,des);
}

#endif // SHRINKBGS_H
