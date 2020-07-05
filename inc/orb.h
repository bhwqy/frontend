#include <vector>
#include <opencv2/opencv.hpp>

class ExtractorNode {
public:

    ExtractorNode():bNoMore(false){}
    void DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4);

    std::vector<cv::KeyPoint> vKeys;
    cv::Point2i UL, UR, BL, BR;
    std::list<ExtractorNode>::iterator lit;
    bool bNoMore;
    
};

class ORBextractor {
public:
    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);
    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    void operator()(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors);

    int GetLevels() const { return nlevels; }
    float GetScaleFactor() const { return scaleFactor; }
    std::vector<float> GetScaleFactors() const { return mvScaleFactor; }
    std::vector<float> GetInverseScaleFactors() const { return mvInvScaleFactor; }
    std::vector<float> GetScaleSigmaSquares() const { return mvLevelSigma2; }
    std::vector<float> GetInverseScaleSigmaSquares() const { return mvInvLevelSigma2; }
    std::vector<cv::Mat> mvImagePyramid;

protected:

    void ComputePyramid(cv::Mat image);
    void ComputeKeyPointsOctTree(std::vector<std::vector<cv::KeyPoint> >& allKeypoints);    
    std::vector<cv::KeyPoint> DistributeOctTree(const std::vector<cv::KeyPoint>& vToDistributeKeys, const int &minX,
        const int &maxX, const int &minY, const int &maxY, const int &nFeatures, const int &level);

    std::vector<cv::Point> pattern;

    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;

    std::vector<int> mnFeaturesPerLevel;
    std::vector<int> umax;
    std::vector<float> mvScaleFactor;
    std::vector<float> mvInvScaleFactor;
    std::vector<float> mvLevelSigma2;
    std::vector<float> mvInvLevelSigma2;

};
