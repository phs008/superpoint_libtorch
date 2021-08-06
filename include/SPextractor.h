//
// Created by phs on 21. 6. 14..
//

#ifndef SUPERPOINT_CPLUS14_SPEXTRACTOR_H
#define SUPERPOINT_CPLUS14_SPEXTRACTOR_H

#include <opencv2/opencv.hpp>
#include "SuperPoint.h"
#include "optional"


using namespace std;

namespace SuperPoint
{
    class SPextractor
    {
    public:
        SPextractor(string weight_path, float _keypoint_threshold, bool _nms, int _nms_minDistance, float _matching_threshold, bool _show_matchingview = false);

        vector<pair<cv::KeyPoint, cv::KeyPoint>> Run(cv::InputArray image, std::string img_name);

        void Show_MatchingPoint(cv::Mat leftImg, cv::Mat rightImg, vector<pair<cv::KeyPoint, cv::KeyPoint>> matching_point);

    public:
        ImageFrame_CV *currentFrame = nullptr;
        ImageFrame_CV *lastFrame = nullptr;


    public:
        bool showMatchingView;
    private:
        std::shared_ptr<SuperPoint> model;
        std::shared_ptr<SPDetector> detector;
    };
}

#endif //SUPERPOINT_CPLUS14_SPEXTRACTOR_H
