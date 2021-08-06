//
// Created by phs on 21. 6. 14..
//

#ifndef SUPERPOINT_CPLUS14_SUPERPOINT_H
#define SUPERPOINT_CPLUS14_SUPERPOINT_H

#include <torch/nn/module.h>
#include <torch/nn/modules/conv.h>
#include <opencv2/opencv.hpp>

using namespace std;

namespace SuperPoint
{
    struct SuperPoint : torch::nn::Module
    {
        SuperPoint();

        std::vector<torch::Tensor> forward(torch::Tensor x);


        torch::nn::Conv2d conv1a;
        torch::nn::Conv2d conv1b;

        torch::nn::Conv2d conv2a;
        torch::nn::Conv2d conv2b;

        torch::nn::Conv2d conv3a;
        torch::nn::Conv2d conv3b;

        torch::nn::Conv2d conv4a;
        torch::nn::Conv2d conv4b;

        torch::nn::Conv2d convPa;
        torch::nn::Conv2d convPb;

        // descriptor
        torch::nn::Conv2d convDa;
        torch::nn::Conv2d convDb;
    };

    class ImageFrame_TORCH
    {
    public:
        torch::Tensor keypoints; // [keypoints cnt , {x,y}]
        torch::Tensor desc; // [256 , keypoints cnt]
        cv::Mat img;
    public:
        ImageFrame_TORCH(torch::Tensor _keypoints, torch::Tensor _desc, cv::Mat _img) : keypoints(_keypoints), desc(_desc), img(_img)
        {}
    };

    class ImageFrame_CV
    {
    public:
        std::vector<cv::KeyPoint> keypoints;
        torch::Tensor desc;
        cv::Mat img;
    };


    class SPDetector
    {
    public:
        SPDetector(std::shared_ptr<SuperPoint> model, bool cuda, float keypoint_threshold, bool nms, int minDistance, float matching_threshold);

        ImageFrame_TORCH Detect(cv::Mat image);

        ImageFrame_CV Detect_CV(cv::Mat image);

        vector<tuple<int, int, float>> Matching(ImageFrame_TORCH lv, ImageFrame_TORCH rv);

        vector<tuple<int, int, float>> Matching(ImageFrame_CV lv, ImageFrame_CV rv);

    private:
        ImageFrame_CV NMS_CV(torch::Tensor xs, torch::Tensor ys, torch::Tensor prob, torch::Tensor coarse_desc, int H, int W, int dist_thresh);

        std::pair<torch::Tensor, torch::Tensor> NMS_TORCH(torch::Tensor in_corners, int H, int W, int dist_thresh);

        torch::Tensor Descriptor_TORCH(int W, int H, torch::Tensor sampX, torch::Tensor sampY, torch::Tensor coarse_desc);

    private:
        float keypoint_threshold;
        bool nms;
        int minDistance;
        float matching_threshold;
    private:
        int best_kpoint = 1000;
    private:
        double mt = 1e3 / 30;
    private:
        torch::Device mTorchDevice = torch::kCPU;
        std::shared_ptr<SuperPoint> model;
        cv::BFMatcher matcher;
//        cv::BFMatcher matcher(cv::NORM_L2);
    };

}

#endif //SUPERPOINT_CPLUS14_SUPERPOINT_H
