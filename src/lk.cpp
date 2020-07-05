#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

#include "inc/lk.h"

using namespace std;
using namespace cv;

static float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols) x = img.cols - 1;
    if (y >= img.rows) y = img.rows - 1;
    uchar *data = &img.data[int(y) * img.step + int(x)];
    float xx = x - floor(x);
    float yy = y - floor(y);
    return float(
        (1 - xx) * (1 - yy) * data[0] +
        xx * (1 - yy) * data[1] +
        (1 - xx) * yy * data[img.step] +
        xx * yy * data[img.step + 1]
    );
}

void OpticalFlowSingleLevel(
    const Mat &img1, const Mat &img2, const vector<cv::Point2f>& pts1, 
    vector<cv::Point2f>& pts2, vector<bool> &success, bool inverse, bool has_initial
) {
    pts2.resize(pts1.size());
    success.resize(pts1.size());
    OpticalFlowTracker tracker(img1, img2, pts1, pts2, success, inverse, has_initial);
    cv::parallel_for_(cv::Range(0, pts1.size()),
        std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const Range &range) {
    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    for (size_t i = range.start; i < range.end; i++) {
        auto pt = pts1[i];
        double dx = 0, dy = 0; // dx,dy need to be estimated
        if (has_initial) {
            dx = pts2[i].x - pt.x;
            dy = pts2[i].y - pt.y;
        }

        double cost = 0, lastCost = 0;
        bool succ = true; // indicate if this point succeeded

        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias
        Eigen::Vector2d J;  // jacobian
        for (int iter = 0; iter < iterations; iter++) {
            if (inverse == false) {
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            } else {
                // only reset b
                b = Eigen::Vector2d::Zero();
            }

            cost = 0;

            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, pt.x + x, pt.y + y) -
                                   GetPixelValue(img2, pt.x + x + dx, pt.y + y + dy);  // Jacobian
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img2, pt.x + dx + x + 1, pt.y + dy + y) -
                                   GetPixelValue(img2, pt.x + dx + x - 1, pt.y + dy + y)),
                            0.5 * (GetPixelValue(img2, pt.x + dx + x, pt.y + dy + y + 1) -
                                   GetPixelValue(img2, pt.x + dx + x, pt.y + dy + y - 1))
                        );
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                            0.5 * (GetPixelValue(img1, pt.x + x + 1, pt.y + y) -
                                   GetPixelValue(img1, pt.x + x - 1, pt.y + y)),
                            0.5 * (GetPixelValue(img1, pt.x + x, pt.y + y + 1) -
                                   GetPixelValue(img1, pt.x + x, pt.y + y - 1))
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }

            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // converge
                break;
            }
        }

        success[i] = succ;

        // set pts2
        pts2[i] = pt + Point2f(dx, dy);
    }
}

void OpticalFlowMultiLevel(
    const Mat &img1, const Mat &img2, const vector<cv::Point2f> &pts1,
    vector<cv::Point2f> &pts2, vector<bool> &success, bool inverse
) {
    // parameters
    int pyramids = 4;
    double pyramid_scale = 0.5;
    double scales[] = {1.0, 0.5, 0.25, 0.125};

    // create pyramids
    vector<Mat> pyr1, pyr2; // image pyramids
    for (int i = 0; i < pyramids; i++) {
        if (i == 0) {
            pyr1.push_back(img1);
            pyr2.push_back(img2);
        } else {
            Mat img1_pyr, img2_pyr;
            cv::resize(pyr1[i - 1], img1_pyr,
                       cv::Size(pyr1[i - 1].cols * pyramid_scale, pyr1[i - 1].rows * pyramid_scale));
            cv::resize(pyr2[i - 1], img2_pyr,
                       cv::Size(pyr2[i - 1].cols * pyramid_scale, pyr2[i - 1].rows * pyramid_scale));
            pyr1.push_back(img1_pyr);
            pyr2.push_back(img2_pyr);
        }
    }

    // coarse-to-fine LK tracking in pyramids
    vector<Point2f> pts1_pyr, pts2_pyr;
    for (auto &pt : pts1) {
        auto pt_top = pt;
        pt_top *= scales[pyramids - 1];
        pts1_pyr.push_back(pt_top);
        pts2_pyr.push_back(pt_top);
    }

    for (int level = pyramids - 1; level >= 0; level--) {
        // from coarse to fine
        success.clear();
        OpticalFlowSingleLevel(pyr1[level], pyr2[level], pts1_pyr, pts2_pyr, success, inverse, true);
        if (level > 0) {
            for (auto &pt: pts1_pyr)
                pt /= pyramid_scale;
            for (auto &pt: pts2_pyr)
                pt /= pyramid_scale;
        }
    }

    for (auto &kp: pts2_pyr)
        pts2.push_back(kp);
}
