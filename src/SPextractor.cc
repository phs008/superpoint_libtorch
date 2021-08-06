//
// Created by phs on 21. 6. 14..
//

#include <torch/serialize.h>
#include "SPextractor.h"
#include "TimeClock.h"


namespace SuperPoint
{
    SPextractor::SPextractor(string weight_path, float _keypoint_threshold, bool _nms, int _nms_minDistance, float _matching_threshold, bool _show_matchingview) : showMatchingView(
            _show_matchingview)
    {
        model = make_shared<SuperPoint>();
        torch::load(model , weight_path);
        detector = make_shared<SPDetector>(model, true, _keypoint_threshold, _nms, _nms_minDistance, _matching_threshold);
        torch::autograd::GradMode::set_enabled(false);
    }
    /// 이미지를 한장씩 받아서 lastFrame 과 현재 Frame 간의 keypoint matching 을 수행한다.
    vector<pair<cv::KeyPoint, cv::KeyPoint>> SPextractor::Run(cv::InputArray image, string img_name)
    {
        /// lastFrame 과 current Frame 간에 매칭된 포인트들
        vector<pair<cv::KeyPoint, cv::KeyPoint>> matchingPoints;
        TimeClock::GetInstance().SetStart("----------SuperPoint----------");
        /// Superpoint KeyPoint extractor
        ImageFrame_CV frame = detector->Detect_CV(image.getMat());
        if (lastFrame == nullptr)
        {
            lastFrame = new ImageFrame_CV(frame);
            currentFrame = lastFrame;
        } else
        {
            lastFrame = currentFrame;
            currentFrame = new ImageFrame_CV(frame);
            auto matching_index = detector->Matching(*lastFrame, *currentFrame);
            cout << "Matching cnt is " << matching_index.size() << endl;\

            for (int i = 0; i < matching_index.size(); i++)
            {
                tuple<int, int, float> matcher = matching_index.at(i);
                int lv_indexr = get<0>(matcher);
                int rv_indexr = get<1>(matcher);
                float distance = get<2>(matcher);
                cv::KeyPoint leftPoint(lastFrame->keypoints[lv_indexr].pt, 1);
                cv::KeyPoint rightPoint(currentFrame->keypoints[rv_indexr].pt, 1);
                matchingPoints.push_back(pair<cv::KeyPoint, cv::KeyPoint>(leftPoint, rightPoint));
            }
        }
        TimeClock::GetInstance().SetEnd();
        Show_MatchingPoint(lastFrame->img, currentFrame->img, matchingPoints);
        return matchingPoints;
    }

    /// Matching point 를 보기위한 Show 함수
    void SPextractor::Show_MatchingPoint(cv::Mat leftImg, cv::Mat rightImg, vector<pair<cv::KeyPoint, cv::KeyPoint>> matching_point)
    {
        if (!matching_point.empty())
        {
            int rows = leftImg.rows + leftImg.rows;
            int cols = rightImg.cols + rightImg.cols;
            cv::Mat combine_img(rows, cols, CV_32F);
            leftImg.copyTo(combine_img(cv::Rect(0, 0, leftImg.cols, leftImg.rows)));
            rightImg.copyTo(combine_img(cv::Rect(rightImg.cols, 0, rightImg.cols, rightImg.rows)));

            /// Matching point text drawing
            string matching_text = "Matching point : " + to_string(matching_point.size()) + "";
            cv::Point myPoint;
            myPoint.x = 10;
            myPoint.y = 40;

            /// Font Face
            int myFontFace = 2;

            /// Font Scale
            double myFontScale = 1.2;

            cv::putText( combine_img, matching_text, myPoint, myFontFace, myFontScale, cv::Scalar::all(255) );


            for (int i = 0; i < matching_point.size(); i++)
            {
                auto matching = matching_point.at(i);
                cv::KeyPoint left = matching.first;
                cv::KeyPoint right = matching.second;
                right.pt.x += leftImg.cols;
                cv::circle(combine_img, left.pt, 2, 0, 5, cv::LINE_AA);
                cv::circle(combine_img, right.pt, 2, 0, 5, cv::LINE_AA);
                cv::line(combine_img, left.pt, right.pt, 255, 1, cv::LINE_AA);
            }
            cv::imshow("TOTAL", combine_img);
            cv::waitKey(1);
        }
    }
}