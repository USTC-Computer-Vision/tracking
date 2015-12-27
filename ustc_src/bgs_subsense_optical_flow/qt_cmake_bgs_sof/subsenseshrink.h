#ifndef SUBSENSESHRINK_H
#define SUBSENSESHRINK_H
#include "../../pl/BackgroundSubtractorSuBSENSE.h"

using namespace std;
using namespace cv;
class Yzbx{
public:
    double yzbxNoiseRate;
    int yzbxFrameNum=1;

    Mat BoxUp,BoxDown;
    Mat hitCountUp,hitCountDown;
    Mat rawFG,FG;
    Mat difImage,BoxGap;
    int learnStep=3;

    //boxup and boxdown
    Mat getSingleShrinkFGMask(Mat input, Mat m_oLastFGMask);

    Mat mean,difmax,difmaxCount;
    //mean,difmax
    Mat getSingleShrinkFGMask2(Mat input, Mat m_oLastFGMask);
};

class subsenseShrink: public BackgroundSubtractorSuBSENSE
{
public:
    subsenseShrink(float fRelLBSPThreshold=BGSSUBSENSE_DEFAULT_LBSP_REL_SIMILARITY_THRESHOLD,
                   size_t nDescDistThresholdOffset=BGSSUBSENSE_DEFAULT_DESC_DIST_THRESHOLD_OFFSET,
                   size_t nMinColorDistThreshold=BGSSUBSENSE_DEFAULT_MIN_COLOR_DIST_THRESHOLD,
                   size_t nBGSamples=BGSSUBSENSE_DEFAULT_NB_BG_SAMPLES,
                   size_t nRequiredBGSamples=BGSSUBSENSE_DEFAULT_REQUIRED_NB_BG_SAMPLES,
                   size_t nSamplesForMovingAvgs=BGSSUBSENSE_DEFAULT_N_SAMPLES_FOR_MV_AVGS)
        :BackgroundSubtractorSuBSENSE(fRelLBSPThreshold,nDescDistThresholdOffset,nMinColorDistThreshold,
                                      nBGSamples,nRequiredBGSamples,nSamplesForMovingAvgs){}

    void operator()(cv::InputArray image, cv::OutputArray fgmask, double learningRateOverride=0);
    Mat getNoiseImg();
    Mat getShrinkFGMask(Mat input);
    Mat getRandShrinkFGMask(Mat  input);
    Mat getRandShrinkFGMask2(Mat input);
    Mat getRandShrinkFGMask3(Mat input);
public:
    Mat yzbxRawFGMask;
    Mat yzbxNoiseOffset;
    Mat yzbxNoiseCount;
    double yzbxNoiseRate;
    int yzbxFrameNum=1;

    Mat BoxUp,BoxDown;
    Mat hitCountUp,hitCountDown;
    Mat rawFG,FG;
    Mat difImage,BoxGap;
    int learnStep=3;

    int randMaskNum=1;
    Mat mean;
    vector<Yzbx> yzbxs;
};

#endif // SUBSENSESHRINK_H