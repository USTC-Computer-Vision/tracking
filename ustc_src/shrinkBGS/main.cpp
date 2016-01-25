#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include "shrinkbgs.h"

using namespace std;
using namespace cv;

int main()
{
    cout << "Hello World!" << endl;
    cv::Mat oCurrInputFrame, oCurrSegmMask, oCurrReconstrBGImg;
//    string fileDir="D:\\firefoxDownload\\matlab\\dataset2012\\dataset\\dynamicBackground\\boats\\input";
    string fileDir="/media/yzbx/D/firefoxDownload/matlab/dataset2012/dataset/dynamicBackground/boats/input";
//    string fileDir="/media/yzbx/D/firefoxDownload/matlab/dataset2012/dataset/baseline/office/input";

    int roi[]={6900,7500};
    std::cout<<"fileDir is "<<fileDir<<std::endl;
    stringstream ss;
    int frameNum=roi[0]-100;
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

    oCurrInputFrame=imread(fileName,CV_LOAD_IMAGE_COLOR);
    if(oCurrInputFrame.empty())
    {
        cout<<"cannot open file "<<fileName<<endl;
        return -1;
    }

//    cvtColor(oCurrInputFrame,oCurrInputFrame,CV_RGB2HLS_FULL);
//    oCurrSegmMask.create(oCurrInputFrame.size(),CV_8UC1);
//    oCurrReconstrBGImg.create(oCurrInputFrame.size(),oCurrInputFrame.type());
//    cv::Mat oSequenceROI(oCurrInputFrame.size(),CV_8UC1,cv::Scalar_<uchar>(255)); // for optimal results, pass a constrained ROI to the algorithm (ex: for CDnet, use ROI.bmp)
    shrinkBGS oBGSAlg;
    namedWindow("input",WINDOW_NORMAL);
    vector<vector<Vec3b>> points;
    int psize;
//    vector<Vec3b> labels;
    for(int k=0;; ++k)
    {
        ss.clear();

        frameNum++;
        sprintf(strnum,"%06d",frameNum);
        ss<<fileDir<<"/in"<<strnum<<".jpg";
        ss>>fileName;

        if(frameNum>roi[1])   break;

        LOG_MESSAGE(fileName);
        oCurrInputFrame=imread(fileName,CV_LOAD_IMAGE_COLOR);
        if(oCurrInputFrame.empty())
            break;


        std::cout<<"frame number is "<<frameNum<<std::endl;
        //        cvtColor(oCurrInputFrame, oCurrInputFrame, CV_RGB2Lab);
        oBGSAlg.process(oCurrInputFrame,oCurrSegmMask,oCurrReconstrBGImg); // lower rate in the early frames helps bootstrap the model when foreground is present
//        oBGSAlg.getBackgroundImage(oCurrReconstrBGImg);
//        imshow("input",oCurrInputFrame);
//        imshow("mask",oCurrSegmMask);

        if(cv::waitKey(1)==27)
            break;
    }

//    cv::waitKey(0);
    return 0;
}

