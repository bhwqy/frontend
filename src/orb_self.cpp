#include "iostream"
#include <chrono>
#include "inc/orb.h"

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cout << "usage: frontend img1 img2 ...\n";
        return 1;
    }

    std::vector<cv::Mat> imgs;
    for (int i = 1; i < argc; ++i) {
        imgs.push_back(cv::imread(argv[i], cv::IMREAD_COLOR));
        assert(!imgs.back().empty());
    }

    int n = imgs.size();
    std::vector<std::vector<cv::KeyPoint>> keypoints(n);
    std::vector<cv::Mat> descriptors(n);
    ORBextractor orbextractor(1000, 1.2, 8, 100, 20);
    auto matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; ++i)
        orbextractor(imgs[i], keypoints[i], descriptors[i]);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "extract ORB cost = " << time_used.count() << " seconds.\n";

    for (int i = 0; i < n; ++i) {
        cv::Mat outimg;
        cv::drawKeypoints(imgs[i], keypoints[i], outimg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        std::string save_path = "../result/self_orb" + std::to_string(i) + ".bmp";
        cv::imwrite(save_path, outimg);
    }

    std::vector<std::vector<cv::DMatch>> matches(n - 1);
    t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < n - 1; ++i)
        matcher->match(descriptors[i], descriptors[i + 1], matches[i]);
    t2 = std::chrono::steady_clock::now();
    time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    std::cout << "match ORB cost = " << time_used.count() << " seconds.\n";

    //-- 第四步:匹配点对筛选
    // 计算最小距离和最大距离
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < n - 1; ++i) {
        auto min_max = minmax_element(matches[i].begin(), matches[i].end(),
            [](const cv::DMatch &m1, const cv::DMatch &m2) { return m1.distance < m2.distance; });
        double min_dist = min_max.first->distance;
        double max_dist = min_max.second->distance;
        std::cout << "-- Max dist : " << max_dist << "\n";
        std::cout << "-- Min dist : " << min_dist << "\n";
        
        std::vector<cv::DMatch> good_matches;
        for (int j = 0; j < descriptors[i].rows; ++j)
            if (matches[i][j].distance <= std::max(2 * min_dist, 30.0))
                good_matches.push_back(matches[i][j]);
        cv::Mat img_match, img_goodmatch;
        cv::drawMatches(imgs[i], keypoints[i], imgs[i + 1], keypoints[i + 1], matches[i], img_match);
        cv::drawMatches(imgs[i], keypoints[i], imgs[i + 1], keypoints[i + 1], good_matches, img_goodmatch);

        std::string s1 = "../result/self_match" + std::to_string(i) + ".bmp";
        std::string s2 = "../result/self_goodmatch" + std::to_string(i) + ".bmp";
        cv::imwrite(s1, img_match);
        cv::imwrite(s2, img_goodmatch);
    }

    return 0;
}
