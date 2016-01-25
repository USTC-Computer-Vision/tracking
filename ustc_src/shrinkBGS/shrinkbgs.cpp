#include "shrinkbgs.h"

shrinkBGS::shrinkBGS()
{
    std::cout<<"shrinkBGS()"<<std::endl;
    frameNum=0;

    enableThreshold=true;
    threshold=15;
    showOutput=true;

    L1Threshold[0]=10;
    L1Threshold[1]=5;
    L1Threshold[2]=5;
}

//TODO may be I need split this big function
//set the raw_foregroundMask and pure_foregroundMask
void shrinkBGS::process(const cv::Mat &img_input, cv::Mat &img_output, cv::Mat &img_bgmodel)
{
    if(img_input.empty())
        return;
    if(img_input.type()!=CV_8UC3){
        std::cout<<"image type should be CV_8UC3"<<std::endl;
        return;
    }

    input=img_input.clone();
    //    img_input.copyTo(input);
    if(showOutput)
        cv::imshow("input", input);

    if(frameNum==0){
        loadConfig();
        init();
    }
    else{
        //        img_input.copyTo(input);
        img_rawForegroundMask=Scalar(0);

        double t = (double)getTickCount();
        getRawForegroundMask();
        t = ((double)getTickCount() - t)/getTickFrequency();
        LOG_MESSAGE(t);

        getPureForegroundMask();

        t = (double)getTickCount();
        updateBackground();
        t = ((double)getTickCount() - t)/getTickFrequency();
        LOG_MESSAGE(t);

         t = (double)getTickCount();
        updateForegroundAsBackground();
        t = ((double)getTickCount() - t)/getTickFrequency();
        LOG_MESSAGE(t);

         t = (double)getTickCount();
        updateDistanceThreshold();
        t = ((double)getTickCount() - t)/getTickFrequency();
        LOG_MESSAGE(t);


        if((frameNum%SampleNum)==(SampleNum-1)){
             t = (double)getTickCount();
            updateBound();
            t = ((double)getTickCount() - t)/getTickFrequency();
            LOG_MESSAGE(t);
        }

        if(frameNum>SampleNum){
            t = (double)getTickCount();
            getWeightedForegroundMask();
            t = ((double)getTickCount() - t)/getTickFrequency();
            LOG_MESSAGE(t);

            medianBlur(img_weightedRawForegroundMask,img_weightedPureForegroundMask,5);
            t = (double)getTickCount();
            updateWeightedDistanceThreshold();
            t = ((double)getTickCount() - t)/getTickFrequency();
            LOG_MESSAGE(t);

            imshow("weighted raw foreground",img_weightedRawForegroundMask);
            img_show("weighted Dmin",img_weightedDmin32F);
            img_show("Dmin",img_Dmin32F);
            img_show("weighted distance threshold",img_weightedDistanceThreshold32F);
            img_show("distance threshold",img_distanceThreshold32F);
        }

        debug();
    }

    LOG_MESSAGE("addtion work");

    img_output=img_rawForegroundMask;
    frameNum=frameNum+1;
}

void shrinkBGS::getRawForegroundMask(){
    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
            LOG_MESSAGE_POINT("empty img_roi");
            if(!img_roi.empty()){
                if(img_roi.at<uchar>(i,j)==0){
                    continue;
                }
            }

            Vec3b anCurrColor=input.at<Vec3b>(i,j);
            size_t nGoodSamplesCount=0, nSampleIdx=0;
            size_t nCurrTotColorDistThreshold=(size_t)img_distanceThreshold32F.at<float>(i,j);
            float DThreshold=img_distanceThreshold32F.at<float>(i,j);
            float Dmin=DThreshold;
            LOG_MESSAGE_POINT("start model loop");

            while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<SampleNum) {
                Vec3b anBGColor=vec_BGColorSample[nSampleIdx].at<Vec3b>(i,j);
                size_t nTotSumDist = 0;
                bool failed=false;
                for(size_t c=0;c<3; ++c) {
                    const size_t nColorDist = L1dist(anCurrColor[c],anBGColor[c]);
                    if(!L1Check(anCurrColor,anBGColor,L1Threshold)){
                        failed=true;
                        break;
                    }
                    //                        goto failedcheck3ch;
                    nTotSumDist += nColorDist;
                }

                if(nTotSumDist>nCurrTotColorDistThreshold){
                    failed=true;
                }
                //                    goto failedcheck3ch;
                if(!failed){
                    nGoodSamplesCount++;
                    if(Dmin>nTotSumDist){
                        Dmin=nTotSumDist;
                    }
                }

                //                failedcheck3ch:
                nSampleIdx++;
            }
            LOG_MESSAGE_POINT("end model loop");

            img_Dmin32F.at<float>(i,j)=Dmin;
            LOG_MESSAGE_POINT(Dmin);

            if(nGoodSamplesCount<m_nRequiredBGSamples){
                //foreground
                img_rawForegroundMask.at<uchar>(i,j)=255;
            }
            else{
                //background
                img_rawForegroundMask.at<uchar>(i,j)=0;
                img_distanceThreshold32F.at<float>(i,j)=\
                        DThreshold*(1-distance_learningRate)+distance_learningRate*Dmin;
            }
        }
    }
}

void shrinkBGS::saveConfig()
{
    CvFileStorage* fs = cvOpenFileStorage("./config/shrinkBGS.xml", 0, CV_STORAGE_WRITE);

    cvWriteInt(fs, "enableThreshold", enableThreshold);
    cvWriteInt(fs, "threshold", threshold);
    cvWriteInt(fs, "showOutput", showOutput);

    cvReleaseFileStorage(&fs);
}

void shrinkBGS::loadConfig()
{
    std::cout<<"load settings from ./config/shrinkBGS.xml"<<std::endl;
    CvFileStorage* fs = cvOpenFileStorage("./config/shrinkBGS.xml", 0, CV_STORAGE_READ);

    enableThreshold = cvReadIntByName(fs, 0, "enableThreshold", true);
    threshold = cvReadIntByName(fs, 0, "threshold", 15);
    showOutput = cvReadIntByName(fs, 0, "showOutput", true);

    cvReleaseFileStorage(&fs);
    std::cout<<"load settings from ./config/shrinkBGS.xml end......................."<<std::endl;
}

void shrinkBGS::setRoi(const cv::Mat &img_input_roi)
{
    img_roi=img_input_roi.clone();
}

//use in init and resisted abrupt change in background.
void shrinkBGS::refreshModel(float fSamplesRefreshFrac, bool bForceFGUpdate){
    LOG_MESSAGE("refreshModel");

    const size_t nModelsToRefresh = fSamplesRefreshFrac<1.0f?(size_t)(fSamplesRefreshFrac*SampleNum):SampleNum;
    const size_t nRefreshStartPos = fSamplesRefreshFrac<1.0f?rand()%SampleNum:0;

    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
            if(!img_roi.empty()){
                if(img_roi.at<uchar>(i,j)==0){
                    continue;
                }
            }

            if(bForceFGUpdate||img_rawForegroundMask.at<uchar>(i,j)==0){
                for(size_t nCurrModelIdx=nRefreshStartPos; nCurrModelIdx<nRefreshStartPos+nModelsToRefresh; ++nCurrModelIdx) {
                    int x,y;
                    //                    getRandNeighborPosition_3x3(x,y,i,j,0,img_size);
                    //                    getRandSamplePosition(x,y,i,j,0,img_size);
                    getRandSamplePosition(y,x,j,i,0,img_size);
                    if(bForceFGUpdate ||img_rawForegroundMask.at<uchar>(x,y)==0) {
                        const size_t nCurrRealModelIdx = nCurrModelIdx%SampleNum;
                        for(size_t c=0; c<3; ++c) {
                            vec_BGColorSample[nCurrRealModelIdx].at<Vec3b>(i,j)[c]=input.at<Vec3b>(x,y)[c];
                        }
                    }
                }
            }
        }
    }

}

void shrinkBGS::init(){
    std::cout<<"init ............."<<std::endl;
    img_cols=input.cols;
    img_rows=input.rows;
    img_size=input.size();
    Mat init_img_fgMask(img_size,CV_8U,Scalar(0));
    img_rawForegroundMask=init_img_fgMask;
    Mat init_img_distanceThreshold32F(img_size,CV_32FC1,Scalar(15.0));
    img_distanceThreshold32F=init_img_distanceThreshold32F;
    Mat init_img_backgroundLearnStep(img_size,CV_8U,Scalar(5));
    img_backgroundLearnStep=init_img_backgroundLearnStep;
    Mat init_img_neighborSpreadNum(img_size,CV_8U,Scalar(5));
    img_neighborSpreadNum=init_img_neighborSpreadNum;

    Mat init_img_backgroundLearningRateNum(img_size,CV_8U,Scalar(1));
    img_backgroundLearningRateNum=init_img_backgroundLearningRateNum;

    size_t nLearningRate=img_backgroundLearningRateNum.at<uchar>(100,100);
    LOG_MESSAGE(nLearningRate);
    Mat init_img_lowerBound8UC3(img_size,CV_8UC3,Scalar(30,10,10));
    img_lowerBound8UC3=input-init_img_lowerBound8UC3;
    Mat init_img_upperBound8UC3(img_size,CV_8UC3,Scalar(30,10,10));
    img_upperBound8UC3=input+init_img_upperBound8UC3;
    img_Dmin32F.create(img_size,CV_32F);
    img_Dmin32F=20.0;
    img_weightedDmin32F.create(img_size,CV_32F);
    img_weightedDmin32F=20.0;
    img_weightedDistanceThreshold32F.create(img_size,CV_32F);
    img_weightedDistanceThreshold32F=20.0;
    img_weightedRawForegroundMask.create(img_size,CV_8U);

    //    Vec3b t(30,10,10);
    //    t[0]=30;
    //    t[1]=10;
    //    t[2]=10;
    //    cv::vector<Mat> mats1,mats2,mats3;
    //    split(img_lowerBound8UC3,mats1);
    //    split(img_upperBound8UC3,mats2);
    //    split(input,mats3);
    //    for(int i=0;i<3;i++){
    //        mats1[i]=mats3[i]-t[i];
    //        mats2[i]=mats3[i]+t[i];
    //    }
    //    merge(mats2,img_lowerBound8UC3);
    //    merge(mats3,img_upperBound8UC3);

    std::cout<<"after merge"<<std::endl;
    LOG_MESSAGE("after merge");
    vec_BGColorSample.resize(SampleNum);
    vec_BGDescSample.resize(SampleNum);
    for(size_t s=0; s<SampleNum; ++s) {
        vec_BGColorSample[s].create(img_size,CV_8UC3);
        vec_BGColorSample[s] = cv::Scalar_<uchar>::all(0);
        vec_BGDescSample[s].create(img_size,CV_16UC3);
        vec_BGDescSample[s] = cv::Scalar_<uchar>::all(0);
    }

    refreshModel(1.0f);

    LOG_MESSAGE("init end");

#if _DEBUG
    img_TotalSumDist.create(img_size,CV_8U);
    img_label.create(img_size,CV_8U);
#endif
}

void shrinkBGS::updateBound(){
    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
            if(!img_roi.empty()){
                if(img_roi.at<uchar>(i,j)==0){
                    continue;
                }
            }

            Vec3b vmax,vmin;

            for(size_t idx=0;idx<SampleNum;idx++){
                for(size_t c=0;c<3;c++){
                    if(idx==0||vmax[c]<vec_BGColorSample[idx].at<Vec3b>(i,j)[c]){
                        vmax[c]=vec_BGColorSample[idx].at<Vec3b>(i,j)[c];
                    }

                    if(idx==0||vmin[c]>vec_BGColorSample[idx].at<Vec3b>(i,j)[c]){
                        vmin[c]=vec_BGColorSample[idx].at<Vec3b>(i,j)[c];
                    }
                }
            }

            for(int c=0;c<3;c++){
                img_upperBound8UC3.at<Vec3b>(i,j)[c]=vmax[c];
                img_lowerBound8UC3.at<Vec3b>(i,j)[c]=vmin[c];
            }

        }
    }

    cv::vector<Mat> mats1,mats2,mats3;
    mats3.resize(3);
    split(img_upperBound8UC3,mats1);
    split(img_lowerBound8UC3,mats2);
    for(int c=0;c<3;c++){
//        medianBlur(mats1[c],mats1[c],5);
//        medianBlur(mats2[c],mats2[c],5);
        mats3[c]=mats1[c]-mats2[c];
//        medianBlur(mats3[c],mats3[c],5);
    }
    Mat directorVector;
    merge(mats3,directorVector);

    imshow("directorVector",directorVector);
    Mat normDirectorVector(img_size,CV_32FC3);
    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
            Vec3b v=directorVector.at<Vec3b>(i,j);
            float norm=sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]+0.1);

            for(int c=0;c<3;c++){
                normDirectorVector.at<Vec3f>(i,j)[c]=sqrt(3)*v[c]/norm;
            }
        }
    }
//    img_show("normDirector",normDirectorVector);
//    double maxVal,minVal;
//    minMaxIdx(normDirectorVector,&minVal,&maxVal);
//    LOG_MESSAGE(minVal);
//    LOG_MESSAGE(maxVal);
    img_distanceWeight32FC3=normDirectorVector;
}

bool shrinkBGS::learnStepCheck(Vec3b anCurrColor,size_t x,size_t y,Vec3b learnStep){
    size_t nGoodSamplesCount=0;
    size_t m_nRequiredBGSamples=2;
    size_t nSampleIdx=0;
    size_t nCurrTotColorDistThreshold=(size_t)img_distanceThreshold32F.at<float>(x,y);
    for(int i=0;i<3;i++){
        nCurrTotColorDistThreshold+=learnStep[i];
    }
    while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<SampleNum) {
        Vec3b anBGColor=vec_BGColorSample[nSampleIdx].at<Vec3b>(x,y);
        size_t nTotSumDist = 0;

        for(size_t c=0;c<3; ++c) {
            const size_t nColorDist = L1dist(anCurrColor[c],anBGColor[c]);
            if(!L1Check(anCurrColor,anBGColor,L1Threshold+learnStep))
                goto failedcheck3ch;
            size_t nSumDist=nColorDist;
            nTotSumDist += nSumDist;
        }

        if(nTotSumDist>nCurrTotColorDistThreshold)
            goto failedcheck3ch;

        nGoodSamplesCount++;
failedcheck3ch:
        nSampleIdx++;
    }

    if(nGoodSamplesCount<m_nRequiredBGSamples){
        return false;
    }
    else{
        return true;
    }
}

void shrinkBGS::getPureForegroundMask(){
    medianBlur(img_rawForegroundMask,img_pureForegroundMask,5);
}

void shrinkBGS::updateBackground(){

    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
            if(!img_roi.empty()){
                if(img_roi.at<uchar>(i,j)==0){
                    continue;
                }
            }

            Vec3b anCurrColor=input.at<Vec3b>(i,j);
            size_t nLearningRate=img_backgroundLearningRateNum.at<uchar>(i,j);
            if(img_pureForegroundMask.at<uchar>(i,j)==0&&img_rawForegroundMask.at<uchar>(i,j)==0){
                if((rand()%nLearningRate)==0) {
                    const size_t s_rand = rand()%SampleNum;
                    for(size_t c=0; c<3; ++c) {
                        vec_BGColorSample[s_rand].at<Vec3b>(i,j)[c]=anCurrColor[c];
                    }
                }

                LOG_MESSAGE_POINT("update neighbor");
                int y, x;
                //                getRandNeighborPosition_3x3(x,y,i,j,0,img_size);
                getRandNeighborPosition_3x3(y,x,j,i,0,img_size);

                const size_t n_rand = rand();
                size_t neighborSpreadNum=img_neighborSpreadNum.at<uchar>(i,j);
                if(neighborSpreadNum==0){
                    neighborSpreadNum++;
                    LOG_MESSAGE("neighborSpreadNum==0");
                    std::cout<<"i= "<<i<<" j="<<j<<std::endl;
                }
                if((n_rand%neighborSpreadNum==0)) {
                    const size_t s_rand = rand()%SampleNum;
                    for(size_t c=0; c<3; ++c) {
                        vec_BGColorSample[s_rand].at<Vec3b>(x,y)[c]=anCurrColor[c];
                    }
                }
            }
            else if(img_pureForegroundMask.at<uchar>(i,j)==0||img_rawForegroundMask.at<uchar>(i,j)==0){
                Vec3b learnStep=img_backgroundLearnStep.at<Vec3b>(i,j);
                if(learnStepCheck(anCurrColor,i,j,learnStep)){
                    if((rand()%nLearningRate)==0) {
                        const size_t s_rand = rand()%SampleNum;
                        for(size_t c=0; c<3; ++c) {
                            vec_BGColorSample[s_rand].at<Vec3b>(i,j)[c]=anCurrColor[c];
                        }
                    }

#if _DEBUG
                    img_label.at<uchar>(i,j)=100;
#endif
                }
            }
        }
    }

}

void shrinkBGS::updateForegroundAsBackground(){
    bool forceForegroundAccept=false;
    if(frameNum<SampleNum){
        forceForegroundAccept=true;
    }

    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
            if(!img_roi.empty()){
                if(img_roi.at<uchar>(i,j)==0){
                    continue;
                }
            }

            if(img_pureForegroundMask.at<uchar>(i,j)==0){
                if(forceForegroundAccept && (rand()%(size_t)foregroundAcceptNum)==0) {
                    const size_t s_rand = rand()%SampleNum;
                    for(size_t c=0; c<3; ++c) {
                        vec_BGColorSample[s_rand].at<Vec3b>(i,j)[c]=input.at<Vec3b>(i,j)[c];
                    }
                }
            }
        }
    }

}

void shrinkBGS::updateDistanceThreshold(){
    Mat img_noise=img_rawForegroundMask-img_pureForegroundMask;
    Scalar s=sum(img_noise);
    int roiArea;

    if(img_roi.empty()){
        Scalar roi=sum((img_pureForegroundMask==0));
        roiArea=roi[0];
    }
    else{
        Scalar roi=sum((img_roi>0)&(img_pureForegroundMask==0));
        roiArea=roi[0];
    }
    float rate=1.0*s[0]/roiArea;

    if(rate<0.05){
//        img_distanceThreshold32F=img_distanceThreshold32F-1;
        if(distance_learningRate<0.1) distance_learningRate+=0.01;
    }
    else if(rate>0.1){
//        img_distanceThreshold32F=img_distanceThreshold32F+1;
        if(distance_learningRate>0) distance_learningRate-=0.01;
    }
}

void shrinkBGS::updateWeightedDistanceThreshold(){
    //update weight distance
    Mat img_weightedNoise=img_weightedRawForegroundMask-img_weightedPureForegroundMask;
    Scalar s=sum(img_weightedNoise);
    int roiArea;
    if(img_roi.empty()){
        Scalar roi=sum((img_weightedPureForegroundMask==0));
        roiArea=roi[0];
    }
    else{
        Scalar roi=sum((img_roi>0)&(img_weightedPureForegroundMask==0));
        roiArea=roi[0];
    }
    float rate=1.0*s[0]/roiArea;

    if(rate<0.05){
//        img_weightedDistanceThreshold32F=img_weightedDistanceThreshold32F-1;
        if(weightedDistance_learningRate<0.1) weightedDistance_learningRate+=0.01;
    }
    else if(rate>0.1){
//        img_weightedDistanceThreshold32F=img_weightedDistanceThreshold32F+1;
        if(weightedDistance_learningRate>0) weightedDistance_learningRate-=0.01;
    }
}

void shrinkBGS::debug(){
    int x=150,y=200;
    for(int i=x-5;i<x+5;i++){
        for(int j=y-5;j<y+5;j++){
            input.at<Vec3b>(i,j)[0]=255;
            input.at<Vec3b>(i,j)[1]=255;
            input.at<Vec3b>(i,j)[2]=0;
        }
    }

    x=150,y=150;
    for(int i=x-5;i<x+5;i++){
        for(int j=y-5;j<y+5;j++){
            input.at<Vec3b>(i,j)[0]=255;
            input.at<Vec3b>(i,j)[1]=0;
            input.at<Vec3b>(i,j)[2]=0;
        }
    }

    imshow("debug",input);

    cv::vector<Vec3b> models;
    for(int k=0;k<SampleNum;k++){
        Vec3b p;
        for(int c=0;c<3;c++){
            p[c]=vec_BGColorSample[k].at<Vec3b>(x,y)[c];
        }

        models.push_back(p);
    }

    drawHist(models);
    imshow("models",vec_BGColorSample[10]);
//    Mat distance;
//    img_distanceThreshold32F.convertTo(distance,CV_8U);
//    imshow("distanceThreshold",distance);
//    imshow("totalSumDist",img_TotalSumDist*10);
    imshow("rawFG",img_rawForegroundMask);
    imshow("pureFG",img_pureForegroundMask);
//    imshow("img_label",img_label);
    imshow("lowbound",img_lowerBound8UC3);
    imshow("upbound",img_upperBound8UC3);
}


void shrinkBGS::drawHist(cv::vector<Vec3b> models)
{
    int size=models.size();
    Mat hist(256+5,size*3+5,CV_8U,Scalar(0));
    int c=0;
    for(int i=0;i<size;i++){
        Vec3b p=models.at(i);
        for(int a=0;a<5;a++){
            for(int b=0;b<5;b++){
                hist.at<uchar>(p[c]+a,i*3+b)=255;
            }
        }
    }

    imshow("hist",hist);
}

void shrinkBGS::getWeightedForegroundMask(){
    for(int i=0;i<img_rows;i++){
        for(int j=0;j<img_cols;j++){
            if(!img_roi.empty()){
                if(img_roi.at<uchar>(i,j)==0){
                    continue;
                }
            }
            Vec3b uchar_anCurrColor=input.at<Vec3b>(i,j);
            Vec3f float_anCurrColor;
            for(int c=0;c<3;c++){
                float_anCurrColor[c]=(float)uchar_anCurrColor[c];
            }
            float nCurrTotColorDistThreshold=img_distanceThreshold32F.at<float>(i,j);
            size_t nGoodSamplesCount=0;
//            size_t
            float DThreshold=img_weightedDistanceThreshold32F.at<float>(i,j);
            float Dmin=DThreshold;
            Vec3f anBGColor=img_distanceWeight32FC3.at<float>(i,j);
            for(int k=0;k<SampleNum;k++){
                Vec3b uchar_anBGColor=vec_BGColorSample[k].at<Vec3b>(i,j);
                Vec3f dif;


                for(int c=0;c<3;c++){
                    dif[c]=abs(float_anCurrColor[c]-(float)uchar_anBGColor[c]);
                }

                if(k==0){
                    LOG_MESSAGE_POINT(dif);
                    LOG_MESSAGE_POINT(anBGColor);
                }

                float a=0+dif[1]*anBGColor[2]-dif[2]*anBGColor[1];
                a=abs(a)+abs(dif[0]*anBGColor[2]-dif[2]*anBGColor[0]);
                a=a+abs(dif[0]*anBGColor[1]-dif[1]*anBGColor[0]);

                if(k==0)
                    LOG_MESSAGE_POINT(a);

                if(a<nCurrTotColorDistThreshold){
                    nGoodSamplesCount++;
                    if(Dmin>a){
                        Dmin=a;
                    }
                    if(nGoodSamplesCount>=m_nRequiredBGSamples){
                        break;
                    }
                }
            }
            img_weightedDmin32F.at<float>(i,j)=Dmin;
            LOG_MESSAGE_POINT(Dmin);
            if(nGoodSamplesCount<m_nRequiredBGSamples){
                img_weightedRawForegroundMask.at<uchar>(i,j)=255;
            }
            else{
                img_weightedRawForegroundMask.at<uchar>(i,j)=0;
                img_weightedDistanceThreshold32F.at<float>(i,j)=\
                        DThreshold*(1-weightedDistance_learningRate)+weightedDistance_learningRate*Dmin;
            }
        }
    }
//    Mat matchCount(img_size,CV_8U,Scalar(0));
//    Mat tmp;
//    for(int k=0;k<SampleNum;k++){
//        Mat absDistance;
//        absdiff(vec_BGColorSample[k],input,absDistance);
//        absDistance.convertTo(absDistance,CV_32FC3);
//        //FIXME cross
////        Mat weightDistance=img_distanceWeight.cross(absDistance);
//        Mat weightDistance;
//        img_cross(img_distanceWeight,absDistance,weightDistance);
//        std::cout<<"weightDistance "<<weightDistance.size()<<" "<<weightDistance.type()<<std::endl;
//        cv::vector<Mat> mats;
//        split(weightDistance,mats);
//        Mat distance;
//        distance=abs(mats[0])+abs(mats[1])+abs(mats[2]);
//        tmp=matchCount+1;
//        add(matchCount,1,matchCount,distance<img_weightedDistanceThreshold32F);
//    }

//    img_weightedRawForegroundMask=(matchCount<2);
}

//TODO set parameter for distance and learnrate ...
//TODO add salient detection
//TODO add Dmin
