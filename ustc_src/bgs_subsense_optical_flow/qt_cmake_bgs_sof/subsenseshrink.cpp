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
//    getShrinkFGMask(_image);
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

Mat subsenseShrink::getShrinkFGMask(Mat input){
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

        hitCountDown=input.clone();
        hitCountDown.setTo(1);
        hitCountUp=input.clone();
        hitCountUp.setTo(1);

        yzbxNoiseRate=0.2;
        int row=input.rows;
        int col=input.cols;
        Mat mask;
        mask.create(row,col,CV_8U);
        mask.setTo(0);
        return mask;
    }
    else{
        vector<Mat> inputMats,BoxUpMats,BoxDownMats;
        split(input,inputMats);
        split(BoxUp,BoxUpMats);
        split(BoxDown,BoxDownMats);

        Mat mask=inputMats[0].clone();
        mask.setTo(1);
        Mat bgMask=mask.clone();

        uchar a,b,c,d,e,f;
        for(int i=0;i<3;i++){
            e=bgMask.at<uchar>(150,150);
            bgMask=bgMask&(inputMats[i]<=BoxUpMats[i]+learnStep)&(inputMats[i]>=BoxDownMats[i]-learnStep);
            f=bgMask.at<uchar>(150,150);
            a=inputMats[i].at<uchar>(150,150);
            b=BoxUpMats[i].at<uchar>(150,150);
            c=BoxDownMats[i].at<uchar>(150,150);
//            imshow("bgmask",bgMask);
        }
        e=bgMask.at<uchar>(150,150);
//        bitwise_not(bgMask,mask); //1->254
        mask=bgMask<1;

        f=mask.at<uchar>(150,150);

        rawFG=mask.clone();
        difImage=input.clone();
        vector<Mat> difImages;
        split(difImage,difImages);
        Mat dif1,dif2;
        dif1=difImages[0].clone();
        dif2=difImages[0].clone();
        for(int i=0;i<3;i++){
            dif1=inputMats[i]-BoxUpMats[i];
            dif2=BoxDownMats[i]-inputMats[i];
           max(dif1,dif2,difImages[i]);
        }
        cv::merge(difImages,difImage);

        Mat oCurrFGMask=mask|m_oLastFGMask;
        Mat kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3));
        cv::erode(rawFG,FG,kernel);
        cv::medianBlur(FG,FG,m_nMedianBlurKernelSize);
        cv::dilate(FG,FG,kernel);
//        FG=FG&m_oLastFGMask_dilated;
//        Mat invFG=(mask<1);
//        cv::dilate(invFG,invFG,kernel);
//        imshow("addtion",invFG);

        //
        vector<vector<Point> > contours,contours0;
        vector<Vec4i> hierarchy;
        Mat img=FG.clone();
        findContours( img, contours0, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
        contours.resize(contours0.size());
        for( size_t k = 0; k < contours0.size(); k++ )
            approxPolyDP(Mat(contours0[k]), contours[k], 3, true);

        if(contours0.size()>0){
            Mat cnt_img = Mat::zeros(mask.rows, mask.cols, CV_8UC3);
            int _levels = 4 - 3;
            drawContours( cnt_img, contours, _levels <= 0 ? 3 : -1, Scalar(128,255,255),
                          3, CV_AA, hierarchy, std::abs(_levels) );


            //tracking...
            vector<Moments> mu(contours0.size());
            vector<Vec3f> trackingStatus(contours0.size());
             for( int i = 0; i < contours0.size(); i++ )
                {
                 mu[i] = moments( contours0[i], false );
                 Vec3f v3;
                 v3[0]=mu[i].m10/mu[i].m00;
                 v3[1]=mu[i].m01/mu[i].m00;
                 v3[2]=mu[i].m00;
                 trackingStatus[i]=v3;
                 //void circle(Mat&img, Point center, intradius, const Scalar&color,intthickness=1, intlineType=8, intshift=0)
                cv::circle(cnt_img,cv::Point(v3[0],v3[1]),5,cv::Scalar(0,0,255),-1);
             }
//             imshow("addtion",cnt_img);

            for(int i=0;i<contours0.size();i++){
                double area=contourArea(contours[i]);
            }
        }

        Mat HitMask=mask.clone();
        Mat tmp=inputMats[0].clone();
        vector<Mat> hitCountDownMats,hitCountUpMats;
        split(hitCountDown,hitCountDownMats);
        split(hitCountUp,hitCountUpMats);

        //count the hit
        Mat notTmp=tmp.clone();
//        bitwise_not(oCurrFGMask,notTmp);
        notTmp=oCurrFGMask<1;
        for(int i=0;i<3;i++){
            //up
//            HitMask=(notTmp)&(inputMats[i]-BoxUpMats[i]>1)&(BoxUpMats[i]-inputMats[i]<5);
            HitMask=(inputMats[i]-BoxUpMats[i]<1)&(BoxUpMats[i]-inputMats[i]<5);
            add(hitCountUpMats[i],1,hitCountUpMats[i],HitMask);
            //down
//            HitMask=(notTmp)&(inputMats[i]-BoxDownMats[i]<5)&(BoxDownMats[i]-inputMats[i]<1);
            HitMask=(inputMats[i]-BoxDownMats[i]<5)&(BoxDownMats[i]-inputMats[i]<1);
            add(hitCountDownMats[i],1,hitCountDownMats[i],HitMask);
        }


//        RNG rgb;// rng.fill 产生的矩阵一样。
        Mat randMat=HitMask.clone();

        cout<<"smooth decrease"<<endl;
        //smooth decrease hitCount by 10%
        for(int i=0;i<3;i++){
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
            randu(randMat,0,50);
            subtract(hitCountUpMats[i],1,hitCountUpMats[i],randMat<1);
//             rgb.fill(randMat,RNG::UNIFORM,0,50,false);
            randu(randMat,0,50);
            subtract(hitCountDownMats[i],1,hitCountDownMats[i],randMat<1);
        }

        //smooth shrink by 5%
//        if(yzbxNoiseRate<0.25){
//            BoxUp=BoxUp-1;
//            BoxDown=BoxDown+1;
//        }

        int randK;
        if(yzbxNoiseRate<0.1){
            randK=1;
        }
        else{
            randK=yzbxNoiseRate*50;
        }

        cout<<"smooth shrink "<<endl;
        for(int i=0;i<3;i++){
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
            randu(randMat,0,randK);
            subtract(BoxUpMats[i],1,BoxUpMats[i],randMat<1&(hitCountUpMats[i]<1));
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
            randu(randMat,0,randK);
            add(BoxDownMats[i],1,BoxDownMats[i],randMat<1&(hitCountDownMats[i]<1));

            //no too much improve!
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
//            add(BoxUpMats[i],1,BoxUpMats[i],randMat<1&(hitCountUpMats[i]>1));
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
//            subtract(BoxDownMats[i],1,BoxDownMats[i],randMat<1&(hitCountDownMats[i]>1));
        }

        //set BoxUp and BoxDown by input and FGMask!!!
        //容易出现历史累积错误，最终输出全前景。。。因此随机赋一定新值
        cout<<"set BoxUp and BoxDown by input and FGMask"<<endl;

        for(int i=0;i<3;i++){
//            rgb.fill(randMat,RNG::UNIFORM,0,20,false);
            randu(randMat,0,20);
            HitMask=(BoxUpMats[i]-inputMats[i]<1)&(inputMats[i]-BoxUpMats[i]<learnStep);
            cv::max(BoxUpMats[i],inputMats[i],tmp);
            add(tmp,0,BoxUpMats[i],(m_oLastFGMask_dilated<1)&HitMask);

//             rgb.fill(randMat,RNG::UNIFORM,0,20,false);
            randu(randMat,0,20);
             HitMask=(BoxDownMats[i]-inputMats[i]<learnStep)&(inputMats[i]-BoxDownMats[i]<1);
            cv::min(BoxDownMats[i],inputMats[i],tmp);
            add(tmp,0,BoxDownMats[i],(m_oLastFGMask_dilated<1)&HitMask);
        }
//        randu(randMat,0,20);
//        imshow("addtion",randMat);

        BoxGap=(BoxUp-BoxDown)/2;
        vector<Mat> BoxGapMats;
        split(BoxGap,BoxGapMats);
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

        for(int idx=0;idx<3;idx++){
            a=BoxUpMats[idx].at<uchar>(150,150);
            b=BoxDownMats[idx].at<uchar>(150,150);
            c=hitCountDownMats[idx].at<uchar>(150,150);
            d=hitCountUpMats[idx].at<uchar>(150,150);
            e=hitCountDownMats[idx].at<uchar>(150,150);
            f=inputMats[idx].at<uchar>(150,150);
        }

        Scalar raw=sum(rawFG);
        Scalar pure=sum(FG);
        int rows=FG.rows;
        int cols=FG.cols;
        yzbxNoiseRate=(raw[0]-pure[0])/(rows*cols*256-pure[0]);

        cout<<"yzbxNoiseRate="<<yzbxNoiseRate<<endl;
        return oCurrFGMask;
    }


}
Mat Yzbx::getSingleShrinkFGMask(Mat input,Mat m_oLastFGMask){
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

        hitCountDown=input.clone();
        hitCountDown.setTo(1);
        hitCountUp=input.clone();
        hitCountUp.setTo(1);

        yzbxNoiseRate=0.2;
        int row=input.rows;
        int col=input.cols;
        Mat mask;
        mask.create(row,col,CV_8U);
        mask.setTo(0);
        return mask;
    }
    else{
        vector<Mat> inputMats,BoxUpMats,BoxDownMats;
        split(input,inputMats);
        split(BoxUp,BoxUpMats);
        split(BoxDown,BoxDownMats);

        BoxGap=(BoxUp-BoxDown)/2;
        vector<Mat> BoxGapMats;
        split(BoxGap,BoxGapMats);

        Mat mask=inputMats[0].clone();
        mask.setTo(1);
        Mat bgMask=mask.clone();

        uchar a,b,c,d,e,f;
        for(int i=0;i<3;i++){
            bgMask=bgMask&(inputMats[i]<=BoxUpMats[i])&(inputMats[i]>=BoxDownMats[i]);
        }

        mask=bgMask<1;
        rawFG=mask.clone();

        int m_nMedianBlurKernelSize=9;
//        Mat oCurrFGMask=rawFG|m_oLastFGMask;
        Mat oCurrFGMask=rawFG;
        Mat kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(3,3));
        cv::erode(rawFG,FG,kernel);
        cv::medianBlur(FG,FG,m_nMedianBlurKernelSize);
        cv::dilate(FG,FG,kernel);
        cv::morphologyEx(FG,FG,MORPH_CLOSE,kernel);


        Mat m_oLastFGMask_dilated=mask.clone();
        kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(30,30));
        dilate(m_oLastFGMask,m_oLastFGMask_dilated,kernel);
//        dilate(FG,m_oLastFGMask_dilated,kernel);

//        FG=FG&m_oLastFGMask_dilated;
//        Mat invRawFG=(rawFG<1);
//        cv::dilate(invRawFG,invRawFG,kernel);
//        imshow("addtion",invRawFG);

        //count the hit
        Mat HitMask=mask.clone();
        Mat tmp=inputMats[0].clone();
        vector<Mat> hitCountDownMats,hitCountUpMats;
        split(hitCountDown,hitCountDownMats);
        split(hitCountUp,hitCountUpMats);

        Mat notTmp=tmp.clone();
        notTmp=oCurrFGMask<1;
        for(int i=0;i<3;i++){
            //up
//            HitMask=(notTmp)&(inputMats[i]-BoxUpMats[i]>1)&(BoxUpMats[i]-inputMats[i]<5);
            HitMask=(inputMats[i]<BoxUpMats[i])&(BoxUpMats[i]<inputMats[i]+5)&(m_oLastFGMask_dilated<1);
            add(hitCountUpMats[i],1,hitCountUpMats[i],HitMask);
            //down
//            HitMask=(notTmp)&(inputMats[i]-BoxDownMats[i]<5)&(BoxDownMats[i]-inputMats[i]<1);
            HitMask=(inputMats[i]<BoxDownMats[i]+5)&(BoxDownMats[i]<inputMats[i])&(m_oLastFGMask_dilated<1);
            add(hitCountDownMats[i],1,hitCountDownMats[i],HitMask);
        }


//        RNG rgb;// rng.fill 产生的矩阵一样。
        Mat randMat=mask.clone();

        //smooth decrease hitCount by 10%
        for(int i=0;i<3;i++){
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
            randu(randMat,0,20);
            subtract(hitCountUpMats[i],1,hitCountUpMats[i],randMat<1&(m_oLastFGMask_dilated<1));
//             rgb.fill(randMat,RNG::UNIFORM,0,50,false);
            randu(randMat,0,20);
            subtract(hitCountDownMats[i],1,hitCountDownMats[i],randMat<1&(m_oLastFGMask_dilated<1));
        }
        cv::merge(hitCountUpMats,hitCountUp);
        cv::merge(hitCountDownMats,hitCountDown);

        int randK;
        if(yzbxNoiseRate<0.1){
            randK=1;
        }
        else{
            randK=yzbxNoiseRate*50;
        }

//        cout<<"smooth shrink BoxUp and BoxDown"<<endl;
        for(int i=0;i<3;i++){
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
            randu(randMat,0,randK);
            subtract(BoxUpMats[i],1,BoxUpMats[i],randMat<1&(hitCountUpMats[i]<1)&(m_oLastFGMask_dilated<1));
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
            randu(randMat,0,randK);
            add(BoxDownMats[i],1,BoxDownMats[i],randMat<1&(hitCountDownMats[i]<1)&(m_oLastFGMask_dilated<1));

            //no too much improve!
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
//            add(BoxUpMats[i],1,BoxUpMats[i],randMat<1&(hitCountUpMats[i]>1));
//            rgb.fill(randMat,RNG::UNIFORM,0,50,false);
//            subtract(BoxDownMats[i],1,BoxDownMats[i],randMat<1&(hitCountDownMats[i]>1));
        }

        if(yzbxNoiseRate<0.2){
            for(int i=0;i<3;i++){
                randu(randMat,0,20);
                subtract(BoxUpMats[i],1,BoxUpMats[i],randMat<5&(hitCountUpMats[i]>2)&(BoxGapMats[i]>10));
                add(BoxDownMats[i],1,BoxDownMats[i],randMat<5&(hitCountDownMats[i]>2)&(BoxGapMats[i]>10));
            }

        }
        imshow("addtion",hitCountUp);
        imshow("boxGap",BoxGap);
        imshow("FG",FG);

        //set BoxUp and BoxDown by input and FGMask!!!
        //容易出现历史累积错误，最终输出全前景。。。因此随机赋一定新值
//        cout<<"set BoxUp and BoxDown by input and FGMask"<<endl;


        for(int i=0;i<3;i++){
//            rgb.fill(randMat,RNG::UNIFORM,0,20,false);
            randu(randMat,0,20);
            HitMask=(BoxUpMats[i]-inputMats[i]<1)&(inputMats[i]-BoxUpMats[i]<learnStep);
            cv::max(BoxUpMats[i],inputMats[i],tmp);
            add(tmp,0,BoxUpMats[i],(m_oLastFGMask_dilated<1)&HitMask);

//             rgb.fill(randMat,RNG::UNIFORM,0,20,false);
            randu(randMat,0,20);
             HitMask=(BoxDownMats[i]-inputMats[i]<learnStep)&(inputMats[i]-BoxDownMats[i]<1);
            cv::min(BoxDownMats[i],inputMats[i],tmp);
            add(tmp,0,BoxDownMats[i],(m_oLastFGMask_dilated<1)&HitMask);
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

        for(int idx=0;idx<3;idx++){
            a=BoxUpMats[idx].at<uchar>(150,150);
            b=BoxDownMats[idx].at<uchar>(150,150);
            c=hitCountDownMats[idx].at<uchar>(150,150);
            d=hitCountUpMats[idx].at<uchar>(150,150);
            e=hitCountDownMats[idx].at<uchar>(150,150);
            f=inputMats[idx].at<uchar>(150,150);
        }

        Scalar raw=sum(rawFG);
        Scalar pure=sum(FG);
        int rows=FG.rows;
        int cols=FG.cols;
        yzbxNoiseRate=(raw[0]-pure[0])/(rows*cols*256-pure[0]);
        imshow("rawFG",rawFG);
        imshow("FG",FG);

//        cout<<"yzbxNoiseRate="<<yzbxNoiseRate<<endl;
        return oCurrFGMask;
    }

}

Mat subsenseShrink::getRandShrinkFGMask(Mat input){
    if(yzbxs.empty()){
        for(int i=0;i<randMaskNum;i++){
            Yzbx yzbx;
            yzbxs.push_back(yzbx);
        }
    }
    Mat mask=Mat::zeros(input.rows,input.cols,CV_8U);

//    int divFactor=256/randMaskNum;
    for(int i=0;i<randMaskNum;i++){
//        Mat ret=yzbxs[i].getSingleShrinkFGMask(input,m_oLastFGMask);
        Mat ret=yzbxs[i].getSingleShrinkFGMask(input,m_oLastFGMask);
        mask=mask+ret/randMaskNum;
    }

    return mask;
}

Mat Yzbx::getSingleShrinkFGMask2(Mat input, Mat m_oLastFGMask)
{
    if(mean.empty()){
        cout<<"start init yzbx"<<endl;
       mean=input.clone();


       //work well for 3-channels mat
       difmax=input.clone();
       difmax.setTo(5);

       difmaxCount=input.clone();
       difmaxCount.setTo(1);

       rawFG=Mat::zeros(input.rows,input.cols,CV_8U);
       FG=Mat::zeros(input.rows,input.cols,CV_8U);
       cout<<"init yzbx end"<<endl;
       return rawFG;
    }
    else{
        uchar a,b,c,d,e,f;
        Mat dif=(input-mean)+(mean-input);

        //work well for 3-channels mat
        Mat rawFG3=dif>difmax;

        //compute rawfg
        vector<Mat> rawFG3Mats;
        split(rawFG3,rawFG3Mats);
        rawFG.setTo(255);
        for(int i=0;i<3;i++){
            rawFG=rawFG|rawFG3Mats[i];
        }

        //count hit
        Mat hitMask3=(difmax>dif)&(difmax<dif+learnStep);
        Mat m_oLastFGMask_dilated=m_oLastFGMask.clone();
        Mat kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(10,10));
        dilate(m_oLastFGMask,m_oLastFGMask_dilated,kernel);

        Mat outBgMask3=input.clone();
        vector<Mat> mats;
        split(outBgMask3,mats);
        for(int i=0;i<3;i++){
            mats[i]=(m_oLastFGMask_dilated<1);
        }
        cv::merge(mats,outBgMask3);

//        add(difmaxCount,1,difmaxCount,hitMask3&outBgMask3);

        Mat mat=hitMask3&outBgMask3;
        split(mat,mats);
        vector<Mat> difmaxCountMats;
        split(difmaxCount,difmaxCountMats);
        for(int i=0;i<3;i++){
            add(difmaxCountMats[i],1,difmaxCountMats[i],mats[i]);
        }

        int learnRate=10;
        //shrink hit
        Mat randMat3=input.clone();
        randu(randMat3,0,learnRate);
        mat=randMat3<1&outBgMask3;
        split(mat,mats);
//        subtract(difmaxCount,1,difmaxCount,randMat3<1&outBgMask3);

        for(int i=0;i<3;i++){
            subtract(difmaxCountMats[i],1,difmaxCountMats[i],mats[i]);
        }
        cv::merge(difmaxCountMats,difmaxCount);

        //shrink difmax
        randu(randMat3,0,learnRate);
        mat=randMat3<1&outBgMask3&difmaxCount<1;
        split(mat,mats);
        vector<Mat> difmaxMats;
        split(difmax,difmaxMats);
        //        subtract(difmax,1,difmax,difmaxCount>1&outBgMask3&randMat3);
        for(int i=0;i<3;i++){
            subtract(difmaxMats[i],1,difmaxMats[i],mats[i]);
        }
//        cv::merge(difmaxMats,difmax);


        //rand set difmax through outBgMask by 20%
        randu(randMat3,0,learnRate);
        mat=randMat3<4&outBgMask3;
        split(mat,mats);
        vector<Mat> difMats;
        split(dif,difMats);
        Mat tmp;
        for(int i=0;i<3;i++){
            tmp=max(difmaxMats[i],difMats[i]);
            subtract(tmp,0,difmaxMats[i],mats[i]);
        }
        cv::merge(difmaxMats,difmax);

        //rand update mean through outBgMask by 20%
        Mat doubleMean,doubleInput;
        mean.convertTo(doubleMean,CV_32FC3,0.95);
        input.convertTo(doubleInput,CV_32FC3,0.05);
        doubleMean=doubleMean+doubleInput;
        Mat charMean;
        doubleMean.convertTo(charMean,CV_8UC3);
        randu(randMat3,0,20);
        mat=randMat3<4&outBgMask3;
        split(mat,mats);
        vector<Mat> meanMats,charMeanMats;
        split(mean,meanMats);
        split(charMean,charMeanMats);
        for(int i=0;i<3;i++){
            add(charMeanMats[i],0,meanMats[i],mats[i]);
        }
        cv::merge(meanMats,mean);

        for(int i=0;i<3;i++){
            a=difMats[i].at<char>(150,150);
            b=difmaxMats[i].at<char>(150,150);
            c=difmaxCountMats[i].at<char>(150,150);
            d=mats[i].at<char>(150,150);
            e=rawFG3Mats[i].at<char>(150,150);
            f=charMeanMats[i].at<char>(150,150);
        }

        return rawFG3;
    }
}

Mat subsenseShrink::getRandShrinkFGMask2(Mat input){
    if(yzbxs.empty()){
        for(int i=0;i<randMaskNum;i++){
            Yzbx yzbx;
            yzbxs.push_back(yzbx);
        }
    }

    //    Mat mask=Mat::zeros(input.rows,input.cols,CV_8U);
    Mat mask3=Mat::zeros(input.rows,input.cols,CV_8UC3);

//    int divFactor=256/randMaskNum;
    for(int i=0;i<randMaskNum;i++){
        Mat ret=yzbxs[i].getSingleShrinkFGMask2(input,m_oLastFGMask);
        if(ret.channels()==1){
            cout<<"ret's channels is 1"<<endl;
            break;
        }
        else{
            mask3=mask3+ret/randMaskNum;
        }

    }

    return mask3;
}

Mat subsenseShrink::getRandShrinkFGMask3(Mat input){
    if(yzbxs.empty()){
        for(int i=0;i<randMaskNum;i++){
            Yzbx yzbx;
            yzbxs.push_back(yzbx);
        }
    }

    Mat dif;
    if(mean.empty()){
        dif=Mat::zeros(input.rows,input.cols,CV_8UC3);
        mean=input.clone();
    }
    else{
        dif=(input-mean)+(mean-input);

        Mat m_oLastFGMask_dilated=m_oLastFGMask.clone();
        Mat kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(10,10));
        dilate(m_oLastFGMask,m_oLastFGMask_dilated,kernel);

        Mat outBgMask3=input.clone();
        vector<Mat> mats;
        split(outBgMask3,mats);
        for(int i=0;i<3;i++){
            mats[i]=(m_oLastFGMask_dilated<1);
        }
        cv::merge(mats,outBgMask3);

        Mat doubleMean,doubleInput;
        mean.convertTo(doubleMean,CV_32FC3,0.95);
        input.convertTo(doubleInput,CV_32FC3,0.05);
        doubleMean=doubleMean+doubleInput;
        Mat charMean,mat,randMat3;
        doubleMean.convertTo(charMean,CV_8UC3);
        randMat3=input.clone();
        randu(randMat3,0,20);
        mat=input.clone();
        mat=randMat3<4&outBgMask3;
        split(mat,mats);
        vector<Mat> meanMats,charMeanMats;
        split(mean,meanMats);
        split(charMean,charMeanMats);
        for(int i=0;i<3;i++){
            add(charMeanMats[i],0,meanMats[i],mats[i]);
        }
        cv::merge(meanMats,mean);
    }


//    Mat mask=Mat::zeros(input.rows,input.cols,CV_8U);
    Mat mask3=Mat::zeros(input.rows,input.cols,CV_8UC3);

//    int divFactor=256/randMaskNum;
    for(int i=0;i<randMaskNum;i++){
        Mat ret=yzbxs[i].getSingleShrinkFGMask(dif,m_oLastFGMask);
        if(ret.channels()==1){
            cout<<"ret's channels is 1"<<endl;
            break;
        }
        else{
            mask3=mask3+ret/randMaskNum;
        }

    }

    return mask3;
}
