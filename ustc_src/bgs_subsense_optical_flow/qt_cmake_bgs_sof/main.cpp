#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
//#include "yzbx.h"
//#include "bgs_sof.h"
//#include "bgs_shrink.h"
#include "subsenseshrink.h"
#include "../../pl/BackgroundSubtractorSuBSENSE.h"

using namespace std;
using namespace cv;
int main()
{
    cout << "Hello World!" << endl;
    cv::Mat oCurrInputFrame, oCurrSegmMask, oCurrReconstrBGImg;
//    string fileDir="D:\\firefoxDownload\\matlab\\dataset2012\\dataset\\dynamicBackground\\boats\\input";
    string fileDir="/media/yzbx/D/firefoxDownload/matlab/dataset2012/dataset/dynamicBackground/boats/input";

    std::cout<<"fileDir is "<<fileDir<<std::endl;
    stringstream ss;
    int frameNum=6900;
    char strnum[10];
    sprintf(strnum,"%06d",frameNum);
    printf("strnum is %s",strnum);

    string numstr=strnum;
    cout<<"numstr is "<<numstr<<endl;

    ss<<fileDir<<"/in"<<strnum<<".jpg";
    string fileName;
    ss>>fileName;
    std::cout<<"fileName is "<<fileName<<std::endl;
    fileName=fileDir+"/in"+numstr+".jpg";
    std::cout<<"fileName is "<<fileName<<std::endl;

    oCurrInputFrame=imread(fileName);
    if(oCurrInputFrame.empty())
    {
        cout<<"cannot open file "<<fileName<<endl;
        return -1;
    }

    oCurrSegmMask.create(oCurrInputFrame.size(),CV_8UC1);
    oCurrReconstrBGImg.create(oCurrInputFrame.size(),oCurrInputFrame.type());
    cv::Mat oSequenceROI(oCurrInputFrame.size(),CV_8UC1,cv::Scalar_<uchar>(255)); // for optimal results, pass a constrained ROI to the algorithm (ex: for CDnet, use ROI.bmp)
    cv::namedWindow("input",cv::WINDOW_NORMAL);
    cv::namedWindow("mask",cv::WINDOW_NORMAL);
    cv::namedWindow("rawFG",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("FG",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("difImage",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("boxGap",cv::WINDOW_AUTOSIZE);
    cv::namedWindow("addtion",cv::WINDOW_AUTOSIZE);
    subsenseShrink oBGSAlg;
//    BackgroundSubtractorSuBSENSE oBGSAlg;
    oBGSAlg.initialize(oCurrInputFrame,oSequenceROI);
    for(int k=0;; ++k)
    {
        ss.clear();

        frameNum++;
        sprintf(strnum,"%06d",frameNum);
        ss<<fileDir<<"/in"<<strnum<<".jpg";
        ss>>fileName;

        if(frameNum>7400)   break;

        oCurrInputFrame=imread(fileName);
        if(oCurrInputFrame.empty())
            break;

        std::cout<<"frame number is "<<frameNum<<std::endl;
        //        cvtColor(oCurrInputFrame, oCurrInputFrame, CV_RGB2Lab);
        oBGSAlg(oCurrInputFrame,oCurrSegmMask,double(k<=100)); // lower rate in the early frames helps bootstrap the model when foreground is present
//        oBGSAlg.getBackgroundImage(oCurrReconstrBGImg);
        imshow("input",oCurrInputFrame);
        imshow("mask",oCurrSegmMask);
        oBGSAlg.getShrinkFGMask(oCurrInputFrame);

        Mat gray;
        if(k>1){
            imshow("rawFG",oBGSAlg.rawFG);
            imshow("FG",oBGSAlg.FG);
//            cv::cvtColor(oBGSAlg.difImage,gray,CV_RGB2GRAY);
            imshow("difImage",oBGSAlg.difImage);
//            imshow("difImage",gray);
            imshow("boxGap",oBGSAlg.BoxGap);
        }


        if(cv::waitKey(1)==27)
            break;
    }
    return 0;
}

