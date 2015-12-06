#include "ustc_bgs.h"
#include <iostream>
USTC_BGS::USTC_BGS(int type){
    frameNum=0;
    int i=type;
    CV_Assert(i>=0&&i<=37);

    if(i==0) bgs = new FrameDifferenceBGS;
    if(i==1) bgs = new StaticFrameDifferenceBGS;
    if(i==2) bgs = new WeightedMovingMeanBGS;
    if(i==3) bgs = new WeightedMovingVarianceBGS;
    if(i==4) bgs = new MixtureOfGaussianV1BGS;
    if(i==5) bgs = new MixtureOfGaussianV2BGS;
    if(i==6) bgs = new AdaptiveBackgroundLearning;
    if(i==7) bgs = new AdaptiveSelectiveBackgroundLearning;
    if(i==8) bgs = new GMG;

    /*** DP Package (thanks to Donovan Parks) ***/
    if(i==9) bgs = new DPAdaptiveMedianBGS;
    if(i==10) bgs = new DPGrimsonGMMBGS;
    if(i==11) bgs = new DPZivkovicAGMMBGS;
    if(i==12) bgs = new DPMeanBGS;
    if(i==13) bgs = new DPWrenGABGS;
    if(i==14) bgs = new DPPratiMediodBGS;
    if(i==15) bgs = new DPEigenbackgroundBGS;
    if(i==16) bgs = new DPTextureBGS;

    /*** TB Package (thanks to Thierry Bouwmans, Fida EL BAF and Zhenjie Zhao) ***/
    if(i==17) bgs = new T2FGMM_UM;
    if(i==18) bgs = new T2FGMM_UV;
    if(i==19) bgs = new T2FMRF_UM;
    if(i==20) bgs = new T2FMRF_UV;
    if(i==21) bgs = new FuzzySugenoIntegral;
    if(i==22) bgs = new FuzzyChoquetIntegral;

    /*** JMO Package (thanks to Jean-Marc Odobez) ***/
    if(i==23) bgs = new MultiLayerBGS;

    /*** PT Package (thanks to Martin Hofmann, Philipp Tiefenbacher and Gerhard Rigoll) ***/
    //       if(i==24) bgs = new PixelBasedAdaptiveSegmenter;

    /*** LB Package (thanks to Laurence Bender) ***/
    if(i==25) bgs = new LBSimpleGaussian;
    if(i==26) bgs = new LBFuzzyGaussian;
    if(i==27) bgs = new LBMixtureOfGaussians;
    if(i==28) bgs = new LBAdaptiveSOM;
    if(i==29) bgs = new LBFuzzyAdaptiveSOM;

    /*** LBP-MRF Package (thanks to Csaba Kertész) ***/
    if(i==30) bgs = new LbpMrf;

    /*** AV Package (thanks to Lionel Robinault and Antoine Vacavant) ***/
    if(i==31) bgs = new VuMeter;

    /*** EG Package (thanks to Ahmed Elgammal) ***/
    if(i==32) bgs = new KDE;

    /*** DB Package (thanks to Domenico Daniele Bloisi) ***/
    if(i==33) bgs = new IndependentMultimodalBGS;

    /*** SJN Package (thanks to SeungJong Noh) ***/
    if(i==34) bgs = new SJN_MultiCueBGS;

    /*** BL Package (thanks to Benjamin Laugraud) ***/
    if(i==35) bgs = new SigmaDeltaBGS;

    /*** PL Package (thanks to Pierre-Luc) ***/
    if(i==36) bgs = new SuBSENSEBGS();
    if(i==37) bgs = new LOBSTERBGS();
}

USTC_BGS::~USTC_BGS(){
}

void USTC_BGS::Release(){
    delete bgs;
}

IplImage* USTC_BGS::GetMask() {
    // CV_Assert(frameNum!=0);
    if(frameNum==0) return NULL;
    // cvShowImage( "c_mask",c_mask);
    // std::cout<<"GetMask "<<c_mask<<" "<<frameNum<<std::endl;
    return c_mask;
}

void    USTC_BGS::Process(IplImage* pImg) {
    // std::cout<<"ustc_bgs::process start, frameNum="<<frameNum<<std::endl;
    img_input=cv::Mat(pImg);
    // std::cout<<"img_input: "<<img_input.rows<<" , "<<img_input.cols<<std::endl;
    // imshow("img_input",img_input);

    // std::cout<<"img_input: "<<img_input.rows<<" , "<<img_input.cols<<std::endl;
    bgs->process(img_input, img_mask, img_bkgmodel);
     
     // std::cout<<"img_input: "<<img_input.rows<<" , "<<img_input.cols<<std::endl;
     if(!img_mask.empty()){
         // std::cout<<"img_mask: "<<img_mask.rows<<" , "<<img_mask.cols<<std::endl;
        b=img_mask.operator IplImage();
        c_mask=&b;
        // if(frameNum==0) cvNamedWindow( "c_mask",0);
        // cvShowImage( "c_mask",c_mask);
        frameNum++;
        // std::cout<<"&b is "<<&b<<" "<<frameNum<<std::endl;
     }
     else{
         std::cout<<"img_mask is empty "<<frameNum<<std::endl;
         frameNum++;
     }
     
    
    // std::cout<<"ustc_bgs::process end, frameNum="<<frameNum<<std::endl;
}