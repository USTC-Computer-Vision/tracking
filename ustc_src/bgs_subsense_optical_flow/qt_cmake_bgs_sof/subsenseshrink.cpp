#include "subsenseshrink.h"
#include "../../pl/DistanceUtils.h"
#include "../../pl/RandUtils.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iomanip>

/*
 *
 * Intrinsic parameters for our method are defined here; tuning these for better
 * performance should not be required in most cases -- although improvements in
 * very specific scenarios are always possible.
 *
 */
//! defines the threshold value(s) used to detect long-term ghosting and trigger the fast edge-based absorption heuristic
#define GHOSTDET_D_MAX (0.010f) // defines 'negligible' change here
#define GHOSTDET_S_MIN (0.995f) // defines the required minimum local foreground saturation value
//! parameter used to scale dynamic distance threshold adjustments ('R(x)')
#define FEEDBACK_R_VAR (0.01f)
//! parameters used to adjust the variation step size of 'v(x)'
#define FEEDBACK_V_INCR  (1.000f)
#define FEEDBACK_V_DECR  (0.100f)
//! parameters used to scale dynamic learning rate adjustments  ('T(x)')
#define FEEDBACK_T_DECR  (0.2500f)
#define FEEDBACK_T_INCR  (0.5000f)
#define FEEDBACK_T_LOWER (2.0000f)
#define FEEDBACK_T_UPPER (256.00f)
//! parameters used to define 'unstable' regions, based on segm noise/bg dynamics and local dist threshold values
#define UNSTABLE_REG_RATIO_MIN (0.100f)
#define UNSTABLE_REG_RDIST_MIN (3.000f)
//! parameters used to scale the relative LBSP intensity threshold used for internal comparisons
#define LBSPDESC_NONZERO_RATIO_MIN (0.100f)
#define LBSPDESC_NONZERO_RATIO_MAX (0.500f)
//! parameters used to define model reset/learning rate boosts in our frame-level component
#define FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD  (m_nMinColorDistThreshold/2)
#define FRAMELEVEL_ANALYSIS_DOWNSAMPLE_RATIO (8)

// local define used to display debug information
#define DISPLAY_SUBSENSE_DEBUG_INFO 0
// local define used to specify the default frame size (320x240 = QVGA)
#define DEFAULT_FRAME_SIZE cv::Size(320,240)
// local define used to specify the color dist threshold offset used for unstable regions
#define STAB_COLOR_DIST_OFFSET (m_nMinColorDistThreshold/5)
// local define used to specify the desc dist threshold offset used for unstable regions
#define UNSTAB_DESC_DIST_OFFSET (m_nDescDistThresholdOffset)

static const size_t s_nColorMaxDataRange_1ch = UCHAR_MAX;
static const size_t s_nDescMaxDataRange_1ch = LBSP::DESC_SIZE*8;
static const size_t s_nColorMaxDataRange_3ch = s_nColorMaxDataRange_1ch*3;
static const size_t s_nDescMaxDataRange_3ch = s_nDescMaxDataRange_1ch*3;
//subsenseShrink::subsenseShrink()
//{

//}

void subsenseShrink::operator()(cv::InputArray _image, cv::OutputArray _fgmask, double learningRateOverride) {
    //yzbx get rand fg
    FG=getRandShrinkFGMask(_image.getMat());

    yzbxInput=_image.getMat();
    // == process
    CV_Assert(m_bInitialized);
    cv::Mat oInputImg = _image.getMat();
    CV_Assert(oInputImg.type()==m_nImgType && oInputImg.size()==m_oImgSize);
    CV_Assert(oInputImg.isContinuous());
    _fgmask.create(m_oImgSize,CV_8UC1);
    cv::Mat oCurrFGMask = _fgmask.getMat();
    memset(oCurrFGMask.data,0,oCurrFGMask.cols*oCurrFGMask.rows);
    size_t nNonZeroDescCount = 0;
    const float fRollAvgFactor_LT = 1.0f/std::min(++m_nFrameIndex,m_nSamplesForMovingAvgs);
    const float fRollAvgFactor_ST = 1.0f/std::min(m_nFrameIndex,m_nSamplesForMovingAvgs/4);
    if(m_nImgChannels==1) {
        for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
            const size_t nPxIter = m_aPxIdxLUT[nModelIter];
            const size_t nDescIter = nPxIter*2;
            const size_t nFloatIter = nPxIter*4;
            const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
            const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
            const uchar nCurrColor = oInputImg.data[nPxIter];
            size_t nMinDescDist = s_nDescMaxDataRange_1ch;
            size_t nMinSumDist = s_nColorMaxDataRange_1ch;
            float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+nFloatIter);
            float* pfCurrVariationFactor = (float*)(m_oVariationModulatorFrame.data+nFloatIter);
            float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+nFloatIter));
            float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+nFloatIter));
            float* pfCurrMeanMinDist_LT = ((float*)(m_oMeanMinDistFrame_LT.data+nFloatIter));
            float* pfCurrMeanMinDist_ST = ((float*)(m_oMeanMinDistFrame_ST.data+nFloatIter));
            float* pfCurrMeanRawSegmRes_LT = ((float*)(m_oMeanRawSegmResFrame_LT.data+nFloatIter));
            float* pfCurrMeanRawSegmRes_ST = ((float*)(m_oMeanRawSegmResFrame_ST.data+nFloatIter));
            float* pfCurrMeanFinalSegmRes_LT = ((float*)(m_oMeanFinalSegmResFrame_LT.data+nFloatIter));
            float* pfCurrMeanFinalSegmRes_ST = ((float*)(m_oMeanFinalSegmResFrame_ST.data+nFloatIter));
            ushort& nLastIntraDesc = *((ushort*)(m_oLastDescFrame.data+nDescIter));
            uchar& nLastColor = m_oLastColorFrame.data[nPxIter];
            const size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[nPxIter])*STAB_COLOR_DIST_OFFSET))/2;
            const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThresholdOffset+(m_oUnstableRegionMask.data[nPxIter]*UNSTAB_DESC_DIST_OFFSET);
            ushort nCurrInterDesc, nCurrIntraDesc;
            LBSP::computeGrayscaleDescriptor(oInputImg,nCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nCurrColor],nCurrIntraDesc);
            m_oUnstableRegionMask.data[nPxIter] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
            size_t nGoodSamplesCount=0, nSampleIdx=0;
            while(nGoodSamplesCount<m_nRequiredBGSamples && nSampleIdx<m_nBGSamples) {
                const uchar& nBGColor = m_voBGColorSamples[nSampleIdx].data[nPxIter];
                {
                    const size_t nColorDist = L1dist(nCurrColor,nBGColor);
                    if(nColorDist>nCurrColorDistThreshold)
                        goto failedcheck1ch;
                    const ushort& nBGIntraDesc = *((ushort*)(m_voBGDescSamples[nSampleIdx].data+nDescIter));
                    const size_t nIntraDescDist = hdist(nCurrIntraDesc,nBGIntraDesc);
                    LBSP::computeGrayscaleDescriptor(oInputImg,nBGColor,nCurrImgCoord_X,nCurrImgCoord_Y,m_anLBSPThreshold_8bitLUT[nBGColor],nCurrInterDesc);
                    const size_t nInterDescDist = hdist(nCurrInterDesc,nBGIntraDesc);
                    const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
                    if(nDescDist>nCurrDescDistThreshold)
                        goto failedcheck1ch;
                    const size_t nSumDist = std::min((nDescDist/4)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
                    if(nSumDist>nCurrColorDistThreshold)
                        goto failedcheck1ch;
                    if(nMinDescDist>nDescDist)
                        nMinDescDist = nDescDist;
                    if(nMinSumDist>nSumDist)
                        nMinSumDist = nSumDist;
                    nGoodSamplesCount++;
                }
failedcheck1ch:
                nSampleIdx++;
            }
            const float fNormalizedLastDist = ((float)L1dist(nLastColor,nCurrColor)/s_nColorMaxDataRange_1ch+(float)hdist(nLastIntraDesc,nCurrIntraDesc)/s_nDescMaxDataRange_1ch)/2;
            *pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
            if(nGoodSamplesCount<m_nRequiredBGSamples) {
                // == foreground
                const float fNormalizedMinDist = std::min(1.0f,((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
                *pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
                *pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
                *pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
                *pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
                oCurrFGMask.data[nPxIter] = UCHAR_MAX;
                if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
                    const size_t s_rand = rand()%m_nBGSamples;
                    *((ushort*)(m_voBGDescSamples[s_rand].data+nDescIter)) = nCurrIntraDesc;
                    m_voBGColorSamples[s_rand].data[nPxIter] = nCurrColor;
                }
            }
            else {
                // == background
                const float fNormalizedMinDist = ((float)nMinSumDist/s_nColorMaxDataRange_1ch+(float)nMinDescDist/s_nDescMaxDataRange_1ch)/2;
                *pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
                *pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
                *pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
                *pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
                const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
                if((rand()%nLearningRate)==0) {
                    const size_t s_rand = rand()%m_nBGSamples;
                    *((ushort*)(m_voBGDescSamples[s_rand].data+nDescIter)) = nCurrIntraDesc;
                    m_voBGColorSamples[s_rand].data[nPxIter] = nCurrColor;
                }
                int nSampleImgCoord_Y, nSampleImgCoord_X;
                const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[nPxIter];
                if(bCurrUsing3x3Spread)
                    getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                else
                    getRandNeighborPosition_5x5(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                const size_t n_rand = rand();
                const size_t idx_rand_uchar = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
                const size_t idx_rand_flt32 = idx_rand_uchar*4;
                const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
                const float fRandMeanRawSegmRes = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
                if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
                        || (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
                    const size_t idx_rand_ushrt = idx_rand_uchar*2;
                    const size_t s_rand = rand()%m_nBGSamples;
                    *((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt)) = nCurrIntraDesc;
                    m_voBGColorSamples[s_rand].data[idx_rand_uchar] = nCurrColor;
                }
            }
            if(m_oLastFGMask.data[nPxIter] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[nPxIter])) {
                if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
                    *pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
            }
            else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
                *pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
            if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
                *pfCurrLearningRate = m_fCurrLearningRateLowerCap;
            else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
                *pfCurrLearningRate = m_fCurrLearningRateUpperCap;
            if(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && m_oBlinksFrame.data[nPxIter])
                (*pfCurrVariationFactor) += FEEDBACK_V_INCR;
            else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
                (*pfCurrVariationFactor) -= m_oLastFGMask.data[nPxIter]?FEEDBACK_V_DECR/4:m_oUnstableRegionMask.data[nPxIter]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
                if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
                    (*pfCurrVariationFactor) = FEEDBACK_V_DECR;
            }
            if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
                (*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
            else {
                (*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
                if((*pfCurrDistThresholdFactor)<1.0f)
                    (*pfCurrDistThresholdFactor) = 1.0f;
            }
            if(popcount(nCurrIntraDesc)>=2)
                ++nNonZeroDescCount;
            nLastIntraDesc = nCurrIntraDesc;
            nLastColor = nCurrColor;
        }
    }
    else { //m_nImgChannels==3
        for(size_t nModelIter=0; nModelIter<m_nTotRelevantPxCount; ++nModelIter) {
            //nPxIter,nPxIterRGB,nDescIterRGB,nFloatIter 均为索引，图像数据按ROI被初始化后，采取这种索引方式
            const size_t nPxIter = m_aPxIdxLUT[nModelIter];
            const int nCurrImgCoord_X = m_aPxInfoLUT[nPxIter].nImgCoord_X;
            const int nCurrImgCoord_Y = m_aPxInfoLUT[nPxIter].nImgCoord_Y;
            const size_t nPxIterRGB = nPxIter*3;
            const size_t nDescIterRGB = nPxIterRGB*2;
            const size_t nFloatIter = nPxIter*4;
            const uchar* const anCurrColor = oInputImg.data+nPxIterRGB;
            size_t nMinTotDescDist=s_nDescMaxDataRange_3ch;
            size_t nMinTotSumDist=s_nColorMaxDataRange_3ch;
            float* pfCurrDistThresholdFactor = (float*)(m_oDistThresholdFrame.data+nFloatIter);
            float* pfCurrVariationFactor = (float*)(m_oVariationModulatorFrame.data+nFloatIter);
            float* pfCurrLearningRate = ((float*)(m_oUpdateRateFrame.data+nFloatIter));
            float* pfCurrMeanLastDist = ((float*)(m_oMeanLastDistFrame.data+nFloatIter));
            float* pfCurrMeanMinDist_LT = ((float*)(m_oMeanMinDistFrame_LT.data+nFloatIter));
            float* pfCurrMeanMinDist_ST = ((float*)(m_oMeanMinDistFrame_ST.data+nFloatIter));
            float* pfCurrMeanRawSegmRes_LT = ((float*)(m_oMeanRawSegmResFrame_LT.data+nFloatIter));
            float* pfCurrMeanRawSegmRes_ST = ((float*)(m_oMeanRawSegmResFrame_ST.data+nFloatIter));
            float* pfCurrMeanFinalSegmRes_LT = ((float*)(m_oMeanFinalSegmResFrame_LT.data+nFloatIter));
            float* pfCurrMeanFinalSegmRes_ST = ((float*)(m_oMeanFinalSegmResFrame_ST.data+nFloatIter));
            ushort* anLastIntraDesc = ((ushort*)(m_oLastDescFrame.data+nDescIterRGB));
            uchar* anLastColor = m_oLastColorFrame.data+nPxIterRGB;
            size_t nCurrColorDistThreshold = (size_t)(((*pfCurrDistThresholdFactor)*m_nMinColorDistThreshold)-((!m_oUnstableRegionMask.data[nPxIter])*STAB_COLOR_DIST_OFFSET));
            const size_t nCurrDescDistThreshold = ((size_t)1<<((size_t)floor(*pfCurrDistThresholdFactor+0.5f)))+m_nDescDistThresholdOffset+(m_oUnstableRegionMask.data[nPxIter]*UNSTAB_DESC_DIST_OFFSET);

            const size_t nCurrTotColorDistThreshold = nCurrColorDistThreshold*3;
            const size_t nCurrTotDescDistThreshold = nCurrDescDistThreshold*3;
            const size_t nCurrSCColorDistThreshold = nCurrTotColorDistThreshold/2;
            ushort anCurrInterDesc[3], anCurrIntraDesc[3];
            const size_t anCurrIntraLBSPThresholds[3] = {m_anLBSPThreshold_8bitLUT[anCurrColor[0]],m_anLBSPThreshold_8bitLUT[anCurrColor[1]],m_anLBSPThreshold_8bitLUT[anCurrColor[2]]};
            LBSP::computeRGBDescriptor(oInputImg,anCurrColor,nCurrImgCoord_X,nCurrImgCoord_Y,anCurrIntraLBSPThresholds,anCurrIntraDesc);
            m_oUnstableRegionMask.data[nPxIter] = ((*pfCurrDistThresholdFactor)>UNSTABLE_REG_RDIST_MIN || (*pfCurrMeanRawSegmRes_LT-*pfCurrMeanFinalSegmRes_LT)>UNSTABLE_REG_RATIO_MIN || (*pfCurrMeanRawSegmRes_ST-*pfCurrMeanFinalSegmRes_ST)>UNSTABLE_REG_RATIO_MIN)?1:0;
            //nGoodSamplesCount 即匹配背景

            size_t nGoodSamplesCount=0, nSampleIdx=0;
            size_t offset=0;
            if(!FG.empty()&&yzbxFrameNum>5){
                uchar fg=FG.at<char>(nCurrImgCoord_Y,nCurrImgCoord_X);
                if(fg>0){
                    offset=offset+5;
                }
            }


            while(nGoodSamplesCount<m_nRequiredBGSamples+offset && nSampleIdx<m_nBGSamples) {
                const ushort* const anBGIntraDesc = (ushort*)(m_voBGDescSamples[nSampleIdx].data+nDescIterRGB);
                const uchar* const anBGColor = m_voBGColorSamples[nSampleIdx].data+nPxIterRGB;
                size_t nTotDescDist = 0;
                size_t nTotSumDist = 0;
                for(size_t c=0;c<3; ++c) {
                    const size_t nColorDist = L1dist(anCurrColor[c],anBGColor[c]);
                    if(nColorDist>nCurrSCColorDistThreshold)
                        goto failedcheck3ch;
                    const size_t nIntraDescDist = hdist(anCurrIntraDesc[c],anBGIntraDesc[c]);
                    LBSP::computeSingleRGBDescriptor(oInputImg,anBGColor[c],nCurrImgCoord_X,nCurrImgCoord_Y,c,m_anLBSPThreshold_8bitLUT[anBGColor[c]],anCurrInterDesc[c]);
                    const size_t nInterDescDist = hdist(anCurrInterDesc[c],anBGIntraDesc[c]);
                    const size_t nDescDist = (nIntraDescDist+nInterDescDist)/2;
                    const size_t nSumDist = std::min((nDescDist/2)*(s_nColorMaxDataRange_1ch/s_nDescMaxDataRange_1ch)+nColorDist,s_nColorMaxDataRange_1ch);
                    if(nSumDist>nCurrSCColorDistThreshold)
                        goto failedcheck3ch;
                    nTotDescDist += nDescDist;
                    nTotSumDist += nSumDist;
                }
                if(nTotDescDist>nCurrTotDescDistThreshold || nTotSumDist>nCurrTotColorDistThreshold)
                    goto failedcheck3ch;
                if(nMinTotDescDist>nTotDescDist)
                    nMinTotDescDist = nTotDescDist;
                if(nMinTotSumDist>nTotSumDist)
                    nMinTotSumDist = nTotSumDist;
                nGoodSamplesCount++;
failedcheck3ch:
                nSampleIdx++;
            }
            const float fNormalizedLastDist = ((float)L1dist<3>(anLastColor,anCurrColor)/s_nColorMaxDataRange_3ch+(float)hdist<3>(anLastIntraDesc,anCurrIntraDesc)/s_nDescMaxDataRange_3ch)/2;
            *pfCurrMeanLastDist = (*pfCurrMeanLastDist)*(1.0f-fRollAvgFactor_ST) + fNormalizedLastDist*fRollAvgFactor_ST;
            if(nGoodSamplesCount<m_nRequiredBGSamples+offset) {
                // == foreground
                const float fNormalizedMinDist = std::min(1.0f,((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2 + (float)(m_nRequiredBGSamples-nGoodSamplesCount)/m_nRequiredBGSamples);
                *pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
                *pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
                *pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT) + fRollAvgFactor_LT;
                *pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST) + fRollAvgFactor_ST;
                //foreground mask
                oCurrFGMask.data[nPxIter] = UCHAR_MAX;
                if(m_nModelResetCooldown && (rand()%(size_t)FEEDBACK_T_LOWER)==0) {
                    const size_t s_rand = rand()%m_nBGSamples;
                    for(size_t c=0; c<3; ++c) {
                        *((ushort*)(m_voBGDescSamples[s_rand].data+nDescIterRGB+2*c)) = anCurrIntraDesc[c];
                        *(m_voBGColorSamples[s_rand].data+nPxIterRGB+c) = anCurrColor[c];
                    }
                }
            }
            else {
                // == background
                const float fNormalizedMinDist = ((float)nMinTotSumDist/s_nColorMaxDataRange_3ch+(float)nMinTotDescDist/s_nDescMaxDataRange_3ch)/2;
                *pfCurrMeanMinDist_LT = (*pfCurrMeanMinDist_LT)*(1.0f-fRollAvgFactor_LT) + fNormalizedMinDist*fRollAvgFactor_LT;
                *pfCurrMeanMinDist_ST = (*pfCurrMeanMinDist_ST)*(1.0f-fRollAvgFactor_ST) + fNormalizedMinDist*fRollAvgFactor_ST;
                *pfCurrMeanRawSegmRes_LT = (*pfCurrMeanRawSegmRes_LT)*(1.0f-fRollAvgFactor_LT);
                *pfCurrMeanRawSegmRes_ST = (*pfCurrMeanRawSegmRes_ST)*(1.0f-fRollAvgFactor_ST);
                const size_t nLearningRate = learningRateOverride>0?(size_t)ceil(learningRateOverride):(size_t)ceil(*pfCurrLearningRate);
                if((rand()%nLearningRate)==0) {
                    const size_t s_rand = rand()%m_nBGSamples;
                    for(size_t c=0; c<3; ++c) {
                        *((ushort*)(m_voBGDescSamples[s_rand].data+nDescIterRGB+2*c)) = anCurrIntraDesc[c];
                        *(m_voBGColorSamples[s_rand].data+nPxIterRGB+c) = anCurrColor[c];
                    }
                }
                int nSampleImgCoord_Y, nSampleImgCoord_X;
                const bool bCurrUsing3x3Spread = m_bUse3x3Spread && !m_oUnstableRegionMask.data[nPxIter];
                if(bCurrUsing3x3Spread)
                    getRandNeighborPosition_3x3(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                else
                    getRandNeighborPosition_5x5(nSampleImgCoord_X,nSampleImgCoord_Y,nCurrImgCoord_X,nCurrImgCoord_Y,LBSP::PATCH_SIZE/2,m_oImgSize);
                const size_t n_rand = rand();
                const size_t idx_rand_uchar = m_oImgSize.width*nSampleImgCoord_Y + nSampleImgCoord_X;
                const size_t idx_rand_flt32 = idx_rand_uchar*4;
                const float fRandMeanLastDist = *((float*)(m_oMeanLastDistFrame.data+idx_rand_flt32));
                const float fRandMeanRawSegmRes = *((float*)(m_oMeanRawSegmResFrame_ST.data+idx_rand_flt32));
                if((n_rand%(bCurrUsing3x3Spread?nLearningRate:(nLearningRate/2+1)))==0
                        || (fRandMeanRawSegmRes>GHOSTDET_S_MIN && fRandMeanLastDist<GHOSTDET_D_MAX && (n_rand%((size_t)m_fCurrLearningRateLowerCap))==0)) {
                    const size_t idx_rand_uchar_rgb = idx_rand_uchar*3;
                    const size_t idx_rand_ushrt_rgb = idx_rand_uchar_rgb*2;
                    const size_t s_rand = rand()%m_nBGSamples;
                    for(size_t c=0; c<3; ++c) {
                        *((ushort*)(m_voBGDescSamples[s_rand].data+idx_rand_ushrt_rgb+2*c)) = anCurrIntraDesc[c];
                        *(m_voBGColorSamples[s_rand].data+idx_rand_uchar_rgb+c) = anCurrColor[c];
                    }
                }
            }
            if(m_oLastFGMask.data[nPxIter] || (std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)<UNSTABLE_REG_RATIO_MIN && oCurrFGMask.data[nPxIter])) {
                if((*pfCurrLearningRate)<m_fCurrLearningRateUpperCap)
                    *pfCurrLearningRate += FEEDBACK_T_INCR/(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*(*pfCurrVariationFactor));
            }
            else if((*pfCurrLearningRate)>m_fCurrLearningRateLowerCap)
                *pfCurrLearningRate -= FEEDBACK_T_DECR*(*pfCurrVariationFactor)/std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST);
            if((*pfCurrLearningRate)<m_fCurrLearningRateLowerCap)
                *pfCurrLearningRate = m_fCurrLearningRateLowerCap;
            else if((*pfCurrLearningRate)>m_fCurrLearningRateUpperCap)
                *pfCurrLearningRate = m_fCurrLearningRateUpperCap;
            if(std::max(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)>UNSTABLE_REG_RATIO_MIN && m_oBlinksFrame.data[nPxIter])
                (*pfCurrVariationFactor) += FEEDBACK_V_INCR;
            else if((*pfCurrVariationFactor)>FEEDBACK_V_DECR) {
                (*pfCurrVariationFactor) -= m_oLastFGMask.data[nPxIter]?FEEDBACK_V_DECR/4:m_oUnstableRegionMask.data[nPxIter]?FEEDBACK_V_DECR/2:FEEDBACK_V_DECR;
                if((*pfCurrVariationFactor)<FEEDBACK_V_DECR)
                    (*pfCurrVariationFactor) = FEEDBACK_V_DECR;
            }
            if((*pfCurrDistThresholdFactor)<std::pow(1.0f+std::min(*pfCurrMeanMinDist_LT,*pfCurrMeanMinDist_ST)*2,2))
                (*pfCurrDistThresholdFactor) += FEEDBACK_R_VAR*(*pfCurrVariationFactor-FEEDBACK_V_DECR);
            else {
                (*pfCurrDistThresholdFactor) -= FEEDBACK_R_VAR/(*pfCurrVariationFactor);
                if((*pfCurrDistThresholdFactor)<1.0f)
                    (*pfCurrDistThresholdFactor) = 1.0f;
            }
            if(popcount<3>(anCurrIntraDesc)>=4)
                ++nNonZeroDescCount;
            for(size_t c=0; c<3; ++c) {
                anLastIntraDesc[c] = anCurrIntraDesc[c];
                anLastColor[c] = anCurrColor[c];
            }
        }
    }
#if DISPLAY_SUBSENSE_DEBUG_INFO
    std::cout << std::endl;
    cv::Point dbgpt(nDebugCoordX,nDebugCoordY);
    cv::Mat oMeanMinDistFrameNormalized; m_oMeanMinDistFrame_ST.copyTo(oMeanMinDistFrameNormalized);
    cv::circle(oMeanMinDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
    cv::resize(oMeanMinDistFrameNormalized,oMeanMinDistFrameNormalized,DEFAULT_FRAME_SIZE);
    cv::imshow("d_min(x)",oMeanMinDistFrameNormalized);
    std::cout << std::fixed << std::setprecision(5) << "  d_min(" << dbgpt << ") = " << m_oMeanMinDistFrame_ST.at<float>(dbgpt) << std::endl;
    cv::Mat oMeanLastDistFrameNormalized; m_oMeanLastDistFrame.copyTo(oMeanLastDistFrameNormalized);
    cv::circle(oMeanLastDistFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
    cv::resize(oMeanLastDistFrameNormalized,oMeanLastDistFrameNormalized,DEFAULT_FRAME_SIZE);
    cv::imshow("d_last(x)",oMeanLastDistFrameNormalized);
    std::cout << std::fixed << std::setprecision(5) << " d_last(" << dbgpt << ") = " << m_oMeanLastDistFrame.at<float>(dbgpt) << std::endl;
    cv::Mat oMeanRawSegmResFrameNormalized; m_oMeanRawSegmResFrame_ST.copyTo(oMeanRawSegmResFrameNormalized);
    cv::circle(oMeanRawSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
    cv::resize(oMeanRawSegmResFrameNormalized,oMeanRawSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
    cv::imshow("s_avg(x)",oMeanRawSegmResFrameNormalized);
    std::cout << std::fixed << std::setprecision(5) << "  s_avg(" << dbgpt << ") = " << m_oMeanRawSegmResFrame_ST.at<float>(dbgpt) << std::endl;
    cv::Mat oMeanFinalSegmResFrameNormalized; m_oMeanFinalSegmResFrame_ST.copyTo(oMeanFinalSegmResFrameNormalized);
    cv::circle(oMeanFinalSegmResFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
    cv::resize(oMeanFinalSegmResFrameNormalized,oMeanFinalSegmResFrameNormalized,DEFAULT_FRAME_SIZE);
    cv::imshow("z_avg(x)",oMeanFinalSegmResFrameNormalized);
    std::cout << std::fixed << std::setprecision(5) << "  z_avg(" << dbgpt << ") = " << m_oMeanFinalSegmResFrame_ST.at<float>(dbgpt) << std::endl;
    cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,0.25f,-0.25f);
    cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
    cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,DEFAULT_FRAME_SIZE);
    cv::imshow("r(x)",oDistThresholdFrameNormalized);
    std::cout << std::fixed << std::setprecision(5) << "      r(" << dbgpt << ") = " << m_oDistThresholdFrame.at<float>(dbgpt) << std::endl;
    cv::Mat oVariationModulatorFrameNormalized; cv::normalize(m_oVariationModulatorFrame,oVariationModulatorFrameNormalized,0,255,cv::NORM_MINMAX,CV_8UC1);
    cv::circle(oVariationModulatorFrameNormalized,dbgpt,5,cv::Scalar(255));
    cv::resize(oVariationModulatorFrameNormalized,oVariationModulatorFrameNormalized,DEFAULT_FRAME_SIZE);
    cv::imshow("v(x)",oVariationModulatorFrameNormalized);
    std::cout << std::fixed << std::setprecision(5) << "      v(" << dbgpt << ") = " << m_oVariationModulatorFrame.at<float>(dbgpt) << std::endl;
    cv::Mat oUpdateRateFrameNormalized; m_oUpdateRateFrame.convertTo(oUpdateRateFrameNormalized,CV_32FC1,1.0f/FEEDBACK_T_UPPER,-FEEDBACK_T_LOWER/FEEDBACK_T_UPPER);
    cv::circle(oUpdateRateFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
    cv::resize(oUpdateRateFrameNormalized,oUpdateRateFrameNormalized,DEFAULT_FRAME_SIZE);
    cv::imshow("t(x)",oUpdateRateFrameNormalized);
    std::cout << std::fixed << std::setprecision(5) << "      t(" << dbgpt << ") = " << m_oUpdateRateFrame.at<float>(dbgpt) << std::endl;
#endif //DISPLAY_SUBSENSE_DEBUG_INFO
    //oCurrFGMask is pure count by Sample distance!
    oCurrFGMask.copyTo(yzbxRawFGMask);

    cv::bitwise_xor(oCurrFGMask,m_oLastRawFGMask,m_oCurrRawFGBlinkMask);
    cv::bitwise_or(m_oCurrRawFGBlinkMask,m_oLastRawFGBlinkMask,m_oBlinksFrame);
    m_oCurrRawFGBlinkMask.copyTo(m_oLastRawFGBlinkMask);
    oCurrFGMask.copyTo(m_oLastRawFGMask);
    cv::morphologyEx(oCurrFGMask,m_oFGMask_PreFlood,cv::MORPH_CLOSE,cv::Mat());
    m_oFGMask_PreFlood.copyTo(m_oFGMask_FloodedHoles);
    cv::floodFill(m_oFGMask_FloodedHoles,cv::Point(0,0),UCHAR_MAX);
    cv::bitwise_not(m_oFGMask_FloodedHoles,m_oFGMask_FloodedHoles);
    cv::erode(m_oFGMask_PreFlood,m_oFGMask_PreFlood,cv::Mat(),cv::Point(-1,-1),3);
    cv::bitwise_or(oCurrFGMask,m_oFGMask_FloodedHoles,oCurrFGMask);
    cv::bitwise_or(oCurrFGMask,m_oFGMask_PreFlood,oCurrFGMask);
    cv::medianBlur(oCurrFGMask,m_oLastFGMask,m_nMedianBlurKernelSize);
    cv::dilate(m_oLastFGMask,m_oLastFGMask_dilated,cv::Mat(),cv::Point(-1,-1),3);
    cv::bitwise_and(m_oBlinksFrame,m_oLastFGMask_dilated_inverted,m_oBlinksFrame);
    cv::bitwise_not(m_oLastFGMask_dilated,m_oLastFGMask_dilated_inverted);
    cv::bitwise_and(m_oBlinksFrame,m_oLastFGMask_dilated_inverted,m_oBlinksFrame);
    m_oLastFGMask.copyTo(oCurrFGMask);
    cv::addWeighted(m_oMeanFinalSegmResFrame_LT,(1.0f-fRollAvgFactor_LT),m_oLastFGMask,(1.0/UCHAR_MAX)*fRollAvgFactor_LT,0,m_oMeanFinalSegmResFrame_LT,CV_32F);
    cv::addWeighted(m_oMeanFinalSegmResFrame_ST,(1.0f-fRollAvgFactor_ST),m_oLastFGMask,(1.0/UCHAR_MAX)*fRollAvgFactor_ST,0,m_oMeanFinalSegmResFrame_ST,CV_32F);

    const float fCurrNonZeroDescRatio = (float)nNonZeroDescCount/m_nTotRelevantPxCount;
    if(fCurrNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN && m_fLastNonZeroDescRatio<LBSPDESC_NONZERO_RATIO_MIN) {
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            if(m_anLBSPThreshold_8bitLUT[t]>cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+ceil(t*m_fRelLBSPThreshold/4)))
                --m_anLBSPThreshold_8bitLUT[t];
    }
    else if(fCurrNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX && m_fLastNonZeroDescRatio>LBSPDESC_NONZERO_RATIO_MAX) {
        for(size_t t=0; t<=UCHAR_MAX; ++t)
            if(m_anLBSPThreshold_8bitLUT[t]<cv::saturate_cast<uchar>(m_nLBSPThresholdOffset+UCHAR_MAX*m_fRelLBSPThreshold))
                ++m_anLBSPThreshold_8bitLUT[t];
    }
    m_fLastNonZeroDescRatio = fCurrNonZeroDescRatio;
    if(m_bLearningRateScalingEnabled) {
        cv::resize(oInputImg,m_oDownSampledFrame_MotionAnalysis,m_oDownSampledFrameSize,0,0,cv::INTER_AREA);
        cv::accumulateWeighted(m_oDownSampledFrame_MotionAnalysis,m_oMeanDownSampledLastDistFrame_LT,fRollAvgFactor_LT);
        cv::accumulateWeighted(m_oDownSampledFrame_MotionAnalysis,m_oMeanDownSampledLastDistFrame_ST,fRollAvgFactor_ST);
        size_t nTotColorDiff = 0;
        for(int i=0; i<m_oMeanDownSampledLastDistFrame_ST.rows; ++i) {
            const size_t idx1 = m_oMeanDownSampledLastDistFrame_ST.step.p[0]*i;
            for(int j=0; j<m_oMeanDownSampledLastDistFrame_ST.cols; ++j) {
                const size_t idx2 = idx1+m_oMeanDownSampledLastDistFrame_ST.step.p[1]*j;
                nTotColorDiff += (m_nImgChannels==1)?
                            (size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2)))/2
                          :  //(m_nImgChannels==3)
                            std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2))),
                                     std::max((size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+4))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+4))),
                                              (size_t)fabs((*(float*)(m_oMeanDownSampledLastDistFrame_ST.data+idx2+8))-(*(float*)(m_oMeanDownSampledLastDistFrame_LT.data+idx2+8)))));
            }
        }
        const float fCurrColorDiffRatio = (float)nTotColorDiff/(m_oMeanDownSampledLastDistFrame_ST.rows*m_oMeanDownSampledLastDistFrame_ST.cols);
        if(m_bAutoModelResetEnabled) {
            if(m_nFramesSinceLastReset>1000)
                m_bAutoModelResetEnabled = false;
            else if(fCurrColorDiffRatio>=FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD && m_nModelResetCooldown==0) {
                m_nFramesSinceLastReset = 0;
                refreshModel(0.1f); // reset 10% of the bg model
                m_nModelResetCooldown = m_nSamplesForMovingAvgs/4;
                m_oUpdateRateFrame = cv::Scalar(1.0f);
            }
            else
                ++m_nFramesSinceLastReset;
        }
        else if(fCurrColorDiffRatio>=FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD*2) {
            m_nFramesSinceLastReset = 0;
            m_bAutoModelResetEnabled = true;
        }
        if(fCurrColorDiffRatio>=FRAMELEVEL_MIN_COLOR_DIFF_THRESHOLD/2) {
            m_fCurrLearningRateLowerCap = (float)std::max((int)FEEDBACK_T_LOWER>>(int)(fCurrColorDiffRatio/2),1);
            m_fCurrLearningRateUpperCap = (float)std::max((int)FEEDBACK_T_UPPER>>(int)(fCurrColorDiffRatio/2),1);
        }
        else {
            m_fCurrLearningRateLowerCap = FEEDBACK_T_LOWER;
            m_fCurrLearningRateUpperCap = FEEDBACK_T_UPPER;
        }
        if(m_nModelResetCooldown>0)
            --m_nModelResetCooldown;
    }
    yzbxFrameNum=yzbxFrameNum+1;

}

Mat subsenseShrink::getNoiseImg(){
    Mat kernel=getStructuringElement(MORPH_ELLIPSE,Size(5,5));
    Mat a;
    dilate(m_oBlinksFrame,a,kernel);
    //unused!
    //    return m_oCurrRawFGBlinkMask;
    //    cout<<"m_oCurrRawFGBlinkMask sum="<<sum(m_oCurrRawFGBlinkMask)<<endl;
    //    cout<<m_oCurrRawFGBlinkMask.channels()<<" "<<m_oCurrRawFGBlinkMask.depth()<<" "<<m_oCurrRawFGBlinkMask.size<<endl;
    //used!
    //    return m_oBlinksFrame;
    //    cout<<"m_oBlinksFrame sum="<<sum(m_oBlinksFrame)<<endl;
    //used!
    //    cout<<"m_oUnstableRegionMask sum="<<m_oUnstableRegionMask<<endl;
    return a;

}

Mat Yzbx::getSingleShrinkFGMask(Mat input,Mat m_oLastFGMask,Mat subsense_R){
    if(BoxDown.empty()){
        cout<<"BoxDown is empty"<<endl;
        //        BoxDown=input-15;
        //        BoxUp=input+15;
        BoxUp=input.clone();
        BoxDown=input.clone();
        vector<Mat> inputMats,BoxUpMats,BoxDownMats;
        split(input,inputMats);
        split(BoxUp,BoxUpMats);
        split(BoxDown,BoxDownMats);

        for(int i=0;i<3;i++){
            add(inputMats[i],10,BoxUpMats[i]);
            subtract(inputMats[i],10,BoxDownMats[i]);
        }
        cv::merge(BoxUpMats,BoxUp);
        cv::merge(BoxDownMats,BoxDown);

        yzbxNoiseRate=0.2;
        int row=input.rows;
        int col=input.cols;
        Mat mask;
        mask.create(row,col,CV_8U);
        mask.setTo(0);
        unStableArea=mask.clone();
        return mask;
    }
    else{
        int minUnstableGap=10;
        vector<Mat> inputMats,BoxUpMats,BoxDownMats;
        split(input,inputMats);
        split(BoxUp,BoxUpMats);
        split(BoxDown,BoxDownMats);

        //only use boxmax, not to use boxmin to resist shadow and light change!
        BoxGap=(BoxUp-BoxDown)/2;
        vector<Mat> BoxGapMats;
        split(BoxGap,BoxGapMats);

        Mat mask=inputMats[0].clone();
        mask.setTo(1);
        Mat bgMask=mask.clone();

        Mat dif3=input.clone();
        vector<Mat> dif;
        split(dif3,dif);
        for(int i=0;i<3;i++){
            bgMask=bgMask&(inputMats[i]<=BoxUpMats[i])&(inputMats[i]>=BoxDownMats[i]);
            dif[i]=(inputMats[i]-BoxUpMats[i])+(BoxDownMats[i]-inputMats[i]);
        }
        merge(dif,dif3);
        Mat gray;
//        cvtColor(dif3,gray,CV_RGB2HLS_FULL)
        cvtColor(dif3,gray,CV_RGB2GRAY);
        imshow("HSL dif3 to gray",gray);

        mask=bgMask<1;
        rawFG=mask.clone();

        int m_nMedianBlurKernelSize=9;
        //        Mat oCurrFGMask=rawFG|m_oLastFGMask;
//        Mat oCurrFGMask=rawFG;
        Mat kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3));
        cv::erode(rawFG,FG,kernel);
        cv::medianBlur(FG,FG,m_nMedianBlurKernelSize);
        cv::dilate(FG,FG,kernel);
        cv::morphologyEx(FG,FG,MORPH_CLOSE,kernel);

        Mat m_oLastFGMask_dilated=mask.clone();
        kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(30,30));
        dilate(m_oLastFGMask,m_oLastFGMask_dilated,kernel);

        imshow("dilate fg",m_oLastFGMask_dilated);
        Mat randMat=mask.clone();
        Mat unStableArea=mask.clone();
        unStableArea.setTo(0);
        for(int i=0;i<3;i++){
            unStableArea=unStableArea|(BoxGapMats[i]>(minUnstableGap+20));
        }
        medianBlur(unStableArea,unStableArea,9);
        imshow("unstable area",unStableArea);

        if(yzbxNoiseRate<0.2){
            for(int i=0;i<3;i++){
                randu(randMat,0,20);
                subtract(BoxUpMats[i],1,BoxUpMats[i],randMat<5&(unStableArea)&(BoxGapMats[i]>minUnstableGap));
                add(BoxDownMats[i],1,BoxDownMats[i],randMat<5&(unStableArea)&(BoxGapMats[i]>minUnstableGap));
            }
        }

        imshow("rawFG",rawFG);
        imshow("FG",FG);
        //set BoxUp and BoxDown by input and FGMask!!!
        //容易出现历史累积错误，最终输出全前景。。。因此随机赋一定新值
        //        cout<<"set BoxUp and BoxDown by input and FGMask"<<endl;

        Mat tmp;
        for(int i=0;i<3;i++){
            //            rgb.fill(randMat,RNG::UNIFORM,0,20,false);
            randu(randMat,0,20);
            cv::max(BoxUpMats[i],inputMats[i],tmp);
            add(tmp,0,BoxUpMats[i],(m_oLastFGMask_dilated<1));

            //             rgb.fill(randMat,RNG::UNIFORM,0,20,false);
            randu(randMat,0,20);
            cv::min(BoxDownMats[i],inputMats[i],tmp);
            add(tmp,0,BoxDownMats[i],(m_oLastFGMask_dilated<1));
        }

        for(int i=0;i<3;i++){
            randu(randMat,0,20);
            cv::max(BoxUpMats[i],inputMats[i]-learnStep,tmp);
            add(tmp,0,BoxUpMats[i],(m_oLastFGMask_dilated<1)&(randMat<1));

            randu(randMat,0,20);
            cv::min(BoxDownMats[i],inputMats[i]+learnStep,tmp);
            add(tmp,0,BoxDownMats[i],(m_oLastFGMask_dilated<1)&(randMat<1));
        }

        cv::merge(BoxUpMats,BoxUp);
        cv::merge(BoxDownMats,BoxDown);

        Scalar raw=sum(rawFG&unStableArea);
        Scalar pure=sum(FG&unStableArea);
        Scalar area=sum(unStableArea);
        yzbxNoiseRate=(raw[0]-pure[0])/(area[0]-pure[0]);


         cout<<"FG.cols "<<FG.cols<<" type"<<FG.type()<<endl;
        add(gray,0,mask,FG&unStableArea);
        cout<<"FG.cols "<<mask.cols<<" type"<<mask.type()<<endl;
        FG=mask;
        cout<<"FG.cols "<<FG.cols<<" type"<<FG.type()<<endl;
        cout<<"yzbxNoiseRate="<<yzbxNoiseRate<<endl;
        return FG;
    }

}

Mat subsenseShrink::getRandShrinkFGMask(Mat _input){
    Mat input=_input.clone();
//    cvtColor(input,input,CV_RGB2HLS_FULL);
    if(yzbxs.empty()){
        for(int i=0;i<randMaskNum;i++){
            Yzbx yzbx;
            yzbxs.push_back(yzbx);
        }
    }
    Mat mask=Mat::zeros(input.rows,input.cols,CV_8U);

    //    int divFactor=256/randMaskNum;
    cv::Mat oDistThresholdFrameNormalized; m_oDistThresholdFrame.convertTo(oDistThresholdFrameNormalized,CV_32FC1,0.25f,-0.25f);
    //    cv::circle(oDistThresholdFrameNormalized,dbgpt,5,cv::Scalar(1.0f));
    //    cv::resize(oDistThresholdFrameNormalized,oDistThresholdFrameNormalized,DEFAULT_FRAME_SIZE);
    for(int i=0;i<randMaskNum;i++){
        //        Mat ret=yzbxs[i].getSingleShrinkFGMask(input,m_oLastFGMask);
        Mat ret=yzbxs[i].getSingleShrinkFGMask(input,m_oLastFGMask,oDistThresholdFrameNormalized);
        mask=mask+ret/randMaskNum;
    }

    return mask;
}

double subsenseShrink::colorDistance(Vec3b a,Vec3b b){
    int na=0,nb=0,cos=0;
    for(int i=0;i<3;i++){
        na=na+a[i]*a[i];
        nb=nb+b[i]*b[i];
        cos=cos+a[i]*b[i];
    }

    double lambda=cos/(sqrt(na*nb+1));
    Vec3d L,d;
    double nl=0,nd=0;
    for(int i=0;i<3;i++){
        L[i]=lambda*b[i];
        d[i]=a[i]-L[i];
        nl=nl+L[i];
        nd=d[i]+nd;
    }

    double dist=0;
    if(nb>nl){
//        dist=sqrt(nd)+sqrt(nl/(bl+1));
        dist=cos+sqrt(nl/(nb+1));
    }
    else{
//        dist=sqrt(nd)+sqrt(bl/(nl+1));
        dist=cos+sqrt(nb/(nl+1));
    }

    return dist;
}
