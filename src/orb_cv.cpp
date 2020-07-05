#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if (argc < 3) {
        cout << "usage: orb_cv img1 img2 ..." << endl;
        return 1;
    }
    //-- 读取图像
    std::vector<cv::Mat> imgs;
    for (int i = 1; i < argc; ++i) {
        imgs.push_back(cv::imread(argv[i], cv::IMREAD_COLOR));
        assert(!imgs.back().empty());
    }

    //-- 初始化
    int n = imgs.size();
    std::vector<std::vector<cv::KeyPoint>> keypoints(n);
    std::vector<cv::Mat> descriptors(n);
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

    //-- 第一步:检测 Oriented FAST 角点位置
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    for (int i = 0; i < n; ++i)
        detector->detect(imgs[i], keypoints[i]);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    for (int i = 0; i < n; ++i)
        descriptor->compute(imgs[i], keypoints[i], descriptors[i]);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "extract ORB cost = " << time_used.count() << " seconds.\n";

    for (int i = 0; i < n; ++i) {
        cv::Mat outimg;
        cv::drawKeypoints(imgs[i], keypoints[i], outimg, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
        string save_path = "../result/cv_orb" + to_string(i) + ".bmp";
        cv::imwrite(save_path, outimg);
    }

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<std::vector<cv::DMatch>> matches(n - 1);
    t1 = chrono::steady_clock::now();
    for (int i = 0; i < n - 1; ++i)
        matcher->match(descriptors[i], descriptors[i + 1], matches[i]);
    t2 = chrono::steady_clock::now();
    time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "match ORB cost = " << time_used.count() << " seconds.\n";

    //-- 第四步:匹配点对筛选
    // 计算最小距离和最大距离
    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < n - 1; ++i) {
        auto min_max = minmax_element(matches[i].begin(), matches[i].end(),
            [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
        double min_dist = min_max.first->distance;
        double max_dist = min_max.second->distance;
        std::cout << "-- Max dist : " << max_dist << "\n";
        std::cout << "-- Min dist : " << min_dist << "\n";
        
        std::vector<DMatch> good_matches;
        for (int j = 0; j < descriptors[i].rows; ++j)
            if (matches[i][j].distance <= max(2 * min_dist, 30.0))
                good_matches.push_back(matches[i][j]);
        cv::Mat img_match, img_goodmatch;
        cv::drawMatches(imgs[i], keypoints[i], imgs[i + 1], keypoints[i + 1], matches[i], img_match);
        cv::drawMatches(imgs[i], keypoints[i], imgs[i + 1], keypoints[i + 1], good_matches, img_goodmatch);

        string s1 = "../result/cv_match" + to_string(i) + ".bmp";
        string s2 = "../result/cv_goodmatch" + to_string(i) + ".bmp";
        cv::imwrite(s1, img_match);
        cv::imwrite(s2, img_goodmatch);
    }

    return 0;
}
