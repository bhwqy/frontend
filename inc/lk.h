#include <vector>
#include <opencv2/opencv.hpp>

class OpticalFlowTracker {
public:
    OpticalFlowTracker(const cv::Mat &img1_, const cv::Mat &img2_, const std::vector<cv::Point2f>& pts1_,
        std::vector<cv::Point2f>& pts2_, std::vector<bool> &success_, bool inverse_ = true, bool has_initial_ = false) :
        img1(img1_), img2(img2_), pts1(pts1_), pts2(pts2_), success(success_), inverse(inverse_), has_initial(has_initial_) {}
    void calculateOpticalFlow(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const std::vector<cv::Point2f> &pts1;
    std::vector<cv::Point2f> &pts2;
    std::vector<bool> &success;
    bool inverse;
    bool has_initial;
};

/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] pts1 keypoints in img1
 * @param [in|out] pts2 keypoints in img2, if empty, use initial guess in pts1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
void OpticalFlowSingleLevel(
    const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &pts1,
    std::vector<cv::Point2f> &pts2, std::vector<bool> &success, bool inverse = false, bool has_initial_guess = false
);

/**
 * multi level optical flow, scale of pyramid is set to 2 by default
 * the image pyramid will be create inside the function
 * @param [in] img1 the first pyramid
 * @param [in] img2 the second pyramid
 * @param [in] pts1 keypoints in img1
 * @param [out] pts2 keypoints in img2
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse set true to enable inverse formulation
 */
void OpticalFlowMultiLevel(
    const cv::Mat &img1, const cv::Mat &img2, const std::vector<cv::Point2f> &pts1,
    std::vector<cv::Point2f> &pts2, std::vector<bool> &success, bool inverse = false
);
