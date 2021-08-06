//
// Created by phs on 21. 6. 14..
//


//#include <torch/nn/options/conv.h>
#include <torch/csrc/api/include/torch/all.h>
#include <geos_c.h>

#include "SuperPoint.h"
#include "TimeClock.h"

using namespace std;
namespace SuperPoint
{
    const int c1 = 64;
    const int c2 = 64;
    const int c3 = 128;
    const int c4 = 128;
    const int c5 = 256;
    const int d1 = 256;

    SuperPoint::SuperPoint()
            :
            conv1a(torch::nn::Conv2dOptions(1, c1, 3).stride(1).padding(1)),
            conv1b(torch::nn::Conv2dOptions(c1, c1, 3).stride(1).padding(1)),

            conv2a(torch::nn::Conv2dOptions(c1, c2, 3).stride(1).padding(1)),
            conv2b(torch::nn::Conv2dOptions(c2, c2, 3).stride(1).padding(1)),

            conv3a(torch::nn::Conv2dOptions(c2, c3, 3).stride(1).padding(1)),
            conv3b(torch::nn::Conv2dOptions(c3, c3, 3).stride(1).padding(1)),

            conv4a(torch::nn::Conv2dOptions(c3, c4, 3).stride(1).padding(1)),
            conv4b(torch::nn::Conv2dOptions(c4, c4, 3).stride(1).padding(1)),

            convPa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
            convPb(torch::nn::Conv2dOptions(c5, 65, 1).stride(1).padding(0)),

            convDa(torch::nn::Conv2dOptions(c4, c5, 3).stride(1).padding(1)),
            convDb(torch::nn::Conv2dOptions(c5, d1, 1).stride(1).padding(0))
    {
        register_module("conv1a", conv1a);
        register_module("conv1b", conv1b);

        register_module("conv2a", conv2a);
        register_module("conv2b", conv2b);

        register_module("conv3a", conv3a);
        register_module("conv3b", conv3b);

        register_module("conv4a", conv4a);
        register_module("conv4b", conv4b);

        register_module("convPa", convPa);
        register_module("convPb", convPb);

        register_module("convDa", convDa);
        register_module("convDb", convDb);
    }


    std::vector<torch::Tensor> SuperPoint::forward(torch::Tensor x)
    {
        x = torch::relu(conv1a->forward(x));
        x = torch::relu(conv1b->forward(x));
        x = torch::max_pool2d(x, 2, 2);

        x = torch::relu(conv2a->forward(x));
        x = torch::relu(conv2b->forward(x));
        x = torch::max_pool2d(x, 2, 2);

        x = torch::relu(conv3a->forward(x));
        x = torch::relu(conv3b->forward(x));
        x = torch::max_pool2d(x, 2, 2);

        x = torch::relu(conv4a->forward(x));
        x = torch::relu(conv4b->forward(x));

        auto cPa = torch::relu(convPa->forward(x));
        auto semi = convPb->forward(cPa);  // [B, 65, H/8, W/8]


        auto cDa = torch::relu(convDa->forward(x));
        auto desc = convDb->forward(cDa);  // [B, d1, H/8, W/8]

        auto dn = torch::norm(desc, 2, 1);
        desc = desc.div(torch::unsqueeze(dn, 1));

        std::vector<torch::Tensor> ret;
        ret.push_back(semi);
        ret.push_back(desc);

        return ret;
    }

    SPDetector::SPDetector(std::shared_ptr<SuperPoint> _model, bool _cuda, float keypoint_threshold, bool _nms, int _minDistance,
                           float matching_threshold) :
            model(_model), keypoint_threshold(keypoint_threshold), nms(_nms), minDistance(_minDistance), matching_threshold(matching_threshold)
    {
        bool use_cuda = _cuda && torch::hasCUDA();
        matcher = cv::BFMatcher(cv::NORM_L2, true);

        if (use_cuda)
            mTorchDevice = torch::kCUDA;
        model->to(mTorchDevice);
        model->eval();
    }

    ImageFrame_CV SPDetector::Detect_CV(cv::Mat image)
    {
        int W = image.cols;
        int H = image.rows;
        auto x = torch::from_blob(image.clone().data, {1, 1, H, W}); /// [SPEED] IO 병목현상 발생
        x = x.clone();
        auto outs = model->forward(x.to(torch::kCUDA));
        torch::Tensor semi = outs[0];
        torch::Tensor coarse_desc = outs[1];

        semi = semi.squeeze(); // [65 , H , W]
        torch::Tensor dense = semi.exp(); // softmax
        dense = dense / (dense.sum(0) + .0001); // sum to 1
        torch::Tensor nodust = dense.slice(0, 0, 64); // remove dustbin
        int Hc = int(H / 8);
        int Wc = int(W / 8);
        nodust = nodust.permute({1, 2, 0});
        torch::Tensor heatmap = nodust.reshape({Hc, Wc, 8, 8});
        heatmap = heatmap.permute({0, 2, 1, 3});
        heatmap = heatmap.reshape({Hc * 8, Wc * 8});
        auto kpts = (heatmap >= keypoint_threshold);
        auto kpts_nonzero = torch::nonzero(kpts);

        kpts_nonzero = kpts_nonzero.transpose(1, 0);

        auto ys = kpts_nonzero[0];
        auto xs = kpts_nonzero[1];
        auto _prob = torch::zeros({xs.size(0)});

        for (int i = 0; i < xs.size(0); i++)
        {
            torch::Scalar x = xs[i].item();
            torch::Scalar y = ys[i].item();
            _prob.slice(0, i, i + 1) = heatmap[y][x];
        }

        torch::Tensor pts = torch::zeros({3, xs.size(0)});
        pts[0] = ys;
        pts[1] = xs;
        pts[2] = _prob;
        auto return_Value = NMS_CV(xs, ys, _prob, coarse_desc, H, W, minDistance);
        return_Value.img = image;
        return return_Value;
    }

    ImageFrame_TORCH SPDetector::Detect(cv::Mat image)
    {
        TimeClock::GetInstance().SetStart("TORCH_DETECT");
        int W = image.cols;
        int H = image.rows;
        auto x = torch::from_blob(image.clone().data, {1, 1, H, W}); /// [SPEED] IO 병목현상 발생
        x = x.clone();
//        x = x.set_requires_grad(false);

        auto outs = model->forward(x.to(torch::kCUDA));
        torch::Tensor semi = outs[0];
        torch::Tensor coarse_desc = outs[1];

        semi = semi.squeeze(); // [65 , H , W]
        torch::Tensor dense = semi.exp(); // softmax
        dense = dense / (dense.sum(0) + .0001); // sum to 1
        torch::Tensor nodust = dense.slice(0, 0, 64); // remove dustbin
        int Hc = int(H / 8);
        int Wc = int(W / 8);
        nodust = nodust.permute({1, 2, 0});
        torch::Tensor heatmap = nodust.reshape({Hc, Wc, 8, 8});
        heatmap = heatmap.permute({0, 2, 1, 3});
        heatmap = heatmap.reshape({Hc * 8, Wc * 8});
        auto kpts = (heatmap >= keypoint_threshold);
        auto kpts_nonzero = torch::nonzero(kpts);


        kpts_nonzero = kpts_nonzero.transpose(1, 0);

        auto ys = kpts_nonzero[0];
        auto xs = kpts_nonzero[1];
        auto _prob = torch::zeros({xs.size(0)});

        for (int i = 0; i < xs.size(0); i++)
        {
            torch::Scalar x = xs[i].item();
            torch::Scalar y = ys[i].item();
            _prob.slice(0, i, i + 1) = heatmap[y][x];
        }

        torch::Tensor pts = torch::zeros({3, xs.size(0)});
        pts[0] = ys;
        pts[1] = xs;
        pts[2] = _prob;

        auto nms_pts = NMS_TORCH(pts, H, W, minDistance);
        auto desc = Descriptor_TORCH(W, H, nms_pts.first, nms_pts.second, coarse_desc);

        torch::Tensor samp_pts = torch::stack({nms_pts.first, nms_pts.second});
        ImageFrame_TORCH rv(samp_pts, desc, image);
        TimeClock::GetInstance().SetEnd();
        return rv;
    }

    ImageFrame_CV SPDetector::NMS_CV(torch::Tensor xs, torch::Tensor ys, torch::Tensor prob, torch::Tensor coarse_desc, int H, int W, int dist_thresh)
    {
        auto scale_factor = std::vector<double>();
        scale_factor.push_back(dist_thresh);
        scale_factor.push_back(dist_thresh);

/// torch1_3
        auto desc_bicubic = torch::upsample_bicubic2d(coarse_desc, {coarse_desc.size(2) * dist_thresh, coarse_desc.size(3) * dist_thresh}, false);
        desc_bicubic = desc_bicubic[0].permute({1, 2, 0});

/// torch1_8
//        torch::nn::Upsample model(torch::nn::UpsampleOptions().scale_factor(scale_factor).mode(torch::kBicubic).align_corners(false));
//        auto desc_bicubic = model->forward(coarse_desc);
//        desc_bicubic = desc_bicubic.index({0}).permute({1, 2, 0});


        cv::Mat grid = cv::Mat::zeros(H, W, CV_32F);
        for (int i = 0; i < xs.size(0); i++)
        {
            auto x = xs[i].item<int>();
            auto y = ys[i].item<int>();
            auto pro = prob[i].item<float>();
            grid.ptr<float>(y)[x] = pro;
        }

        for (int x = 0; x < grid.cols - (dist_thresh - 1); x++)
        {
            for (int y = 0; y < grid.rows - (dist_thresh - 1); y++)
            {
                cv::Mat src = grid(CvRect(x, y, dist_thresh, dist_thresh));
                int min[2], max[2];
                double minVal, maxVal;
                cv::minMaxIdx(src, &minVal, &maxVal, min, max);
                if (maxVal > 0.0)
                {
                    int max_x = max[0], max_y = max[1];
                    for (int inner_c = 0; inner_c < src.cols; inner_c++)
                        for (int inner_r = 0; inner_r < src.rows; inner_r++)
                            grid.ptr<float>(y + inner_c)[x + inner_r] = 0.0;
                    grid.ptr<float>(y + max_y)[x + max_x] = maxVal;
                }
            }
        }

        torch::Tensor descriptors;
        std::vector<cv::KeyPoint> keyPoints;
        for (int col = 0; col < grid.cols; col++)
            for (int row = 0; row < grid.rows; row++)
            {
                float value = grid.ptr<float>(row)[col];
                if (value > 0.0)
                {
                    cv::KeyPoint keypoint(col, row, 0., -1, 0, 0, -1);
                    keyPoints.push_back(keypoint);
                    torch::Tensor des = desc_bicubic[row][col].reshape({1, 256});
                    if (descriptors.numel() == 0)
                        descriptors = des;
                    else
                        descriptors = torch::cat({descriptors, des}, 0);
                }
            }
        ImageFrame_CV returnVal;
        returnVal.keypoints = keyPoints;
        returnVal.desc = descriptors;
        return returnVal;
    };


    std::pair<torch::Tensor, torch::Tensor> SPDetector::NMS_TORCH(torch::Tensor in_corners, int H, int W, int dist_thresh)
    {
        torch::Tensor grid = torch::zeros({H, W}, torch::kFloat);
        torch::Tensor padded_grid;
        torch::Tensor inds = torch::zeros({H, W}, torch::kFloat);
        torch::Tensor inds1 = -in_corners[2];
        inds1 = torch::argsort(inds1);
        auto sorted_ys = in_corners[0].index({inds1}).to(torch::kInt);
        auto sorted_xs = in_corners[1].index({inds1}).to(torch::kInt);
        auto sorted_prob = in_corners[2].index({inds1}).to(torch::kFloat);

        for (int i = 0; i < sorted_xs.size(0); i++)
        {
            auto x = sorted_xs[i].item();
            auto y = sorted_ys[i].item();
            grid[y][x] = torch::Scalar(1.0);
            inds[y][x] = sorted_prob[i];
        }

        grid = torch::constant_pad_nd(grid, {dist_thresh, dist_thresh, dist_thresh, dist_thresh}, 0);

        int count = 0;

        for (int i = 0; i < sorted_xs.size(0); i++)
        {
            auto x = sorted_xs[i].item().toInt() + dist_thresh;
            auto y = sorted_ys[i].item().toInt() + dist_thresh;
            if (grid[y][x].item().toInt() == 1)
            {
                /// y - dist_thresh ~ y + dist_thresh + 1 = 0
                /// x - dist_thresh ~ x + dist_thresh + 1 = 0
                /// y, x = -1
                ///grid.slice(0, y - dist_thresh, y + dist_thresh + 1).slice(1, x - dist_thresh, x + dist_thresh + 1) = torch::Scalar(0);

                grid.slice(0, y - dist_thresh, y + dist_thresh + 1) = torch::Scalar(0);
                grid.slice(0, y - dist_thresh, y + dist_thresh + 1).slice(1, x - dist_thresh, x + dist_thresh + 1) = torch::Scalar(0);

                grid[y][x] = -1;
                count += 1;
            }
        }

        torch::Tensor grid_idx = (grid == -1);

        vector<float> keepx, keepy;
        vector<float> keepidx;


        for (int i = 0; i < grid_idx.size(0); i++)
            for (int j = 0; j < grid_idx.size(1); j++)
                if (grid_idx[i][j].item().toInt() == 1)
                {
                    float x = j - dist_thresh;
                    float y = i - dist_thresh;
                    keepx.push_back(x);
                    keepy.push_back(y);
                    keepidx.push_back(-inds[y][x].item().toFloat());
                }

        torch::Tensor keep_tensorX = torch::from_blob(keepx.data(), {static_cast<int>(keepx.size())}, torch::kFloat);
        torch::Tensor keep_tensorY = torch::from_blob(keepy.data(), {static_cast<int>(keepy.size())}, torch::kFloat);
        torch::Tensor keep_tensorIdx = torch::from_blob(keepidx.data(), {static_cast<int>(keepidx.size())}, torch::kFloat);

        torch::Tensor sorted_tensorProbIdx = torch::argsort(keep_tensorIdx);

        auto sorted_keep_tensorX = keep_tensorX.index({sorted_tensorProbIdx});
        auto sorted_keep_tensorY = keep_tensorY.index({sorted_tensorProbIdx});

        return std::pair<torch::Tensor, torch::Tensor>(sorted_keep_tensorX, sorted_keep_tensorY);
    }

    torch::Tensor SPDetector::Descriptor_TORCH(int W, int H, torch::Tensor sampX, torch::Tensor sampY, torch::Tensor coarse_desc)
    {
        torch::Tensor samp_pts = torch::stack({sampX, sampY});

        /// gird sample 을 위한 처리
        sampX = (sampX / float(W) / 2.) - 1.;
        sampY = (sampY / float(H) / 2.) - 1.;
        torch::Tensor _temp_samp_pts = torch::stack({sampX, sampY});
        _temp_samp_pts = _temp_samp_pts.transpose(0, 1);
        _temp_samp_pts = _temp_samp_pts.view({1, 1, -1, 2});
        coarse_desc = coarse_desc.to(torch::kCPU);
        torch::Tensor desc = torch::grid_sampler(coarse_desc, _temp_samp_pts, 0, 0, false);
        desc = desc.reshape({256, desc.size(3)});
        desc /= torch::norm(desc, 2, {0});
        samp_pts = samp_pts.t().detach();
        desc = desc.detach();
        return desc;
    }

    vector<tuple<int, int, float>> SPDetector::Matching(ImageFrame_CV lv, ImageFrame_CV rv)
    {
        auto des1 = lv.desc; // [des1_cnt , 256]
        auto des2 = rv.desc; // [des2_cnt , 256]
        auto dmat = torch::matmul(des1, des2.t()); // [des1_cnt , des2_cnt]
        dmat = torch::sqrt(2 - 2 * torch::clamp(dmat, -1.0, 1.0));
        auto idx = torch::argmin(dmat, 1);
        auto score = dmat.index_select(1, idx);

        vector<tuple<int, int, float>> matching_index;
        for (int i = 0; i < score.size(0); i++)
        {
            auto minIndex = idx[i].item().toInt();
            auto value = dmat[i][minIndex].item().toFloat();
            if (value < matching_threshold)
            {
                matching_index.push_back(tuple<int, int, float>(i, minIndex, value));
            }
        }
        return matching_index;
    }
}
