#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {

    if (argc < 3) {
        std::cout << "usage: lk_cv img1 img2 ...\n";
        return 1;
    }

    std::vector<cv::Mat> imgs;
    for (int i = 1; i < argc; ++i) {
        imgs.push_back(cv::imread(argv[i], cv::IMREAD_GRAYSCALE));
        assert(!imgs.back().empty());
    }

    int n = imgs.size();
    std::vector<cv::KeyPoint> keypoints;
    Ptr<GFTTDetector> detector = GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(imgs[0], keypoints);

    std::vector<std::vector<cv::Point2f>> pts(n);
    for (auto pt : keypoints) pts[0].push_back(pt.pt);
    std::vector<std::vector<uchar>> status(n - 1);
    std::vector<std::vector<float>> error(n - 1);

    auto t1 = chrono::steady_clock::now();
    for (int i = 0; i < n - 1; ++i)
        cv::calcOpticalFlowPyrLK(imgs[i], imgs[i + 1], pts[i], pts[i + 1], status[i], error[i]);
    auto t2 = chrono::steady_clock::now();
    auto time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "optical flow by opencv: " << time_used.count() << endl;

    for (int i = 0; i < n; ++i) {
        cv::Mat img_save;
        cv::cvtColor(imgs[i], img_save, cv::COLOR_GRAY2BGR);
        if (i == 0) {
            for (auto pt : pts[0])
                cv::circle(img_save, pt, 2, cv::Scalar(0, 250, 0), 2);
        }
        else {
            for (int j = 0; j < pts[i].size(); ++j) {
                if (status[i - 1][j]) {
                    cv::circle(img_save, pts[i][j], 2, cv::Scalar(0, 250, 0), 2);
                    cv::line(img_save, pts[i - 1][j], pts[i][j], cv::Scalar(0, 250, 0));
                }
            }
        }
        string s = "../result/cv_lk" + std::to_string(i) + ".bmp";
        cv::imwrite(s, img_save);
    }
    return 0;
}
