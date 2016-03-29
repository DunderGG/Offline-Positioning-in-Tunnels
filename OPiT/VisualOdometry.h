#ifndef _VISUAL_ODOMETRY_HEADER
#define _VISUAL_ODOMETRY_HEADER

#include <opencv2/features2d.hpp>
#include <initializer_list>
#include <iostream>

enum
{
    VO_METHOD_FAST = 0,
    VO_METHOD_SIFT = 1,
    VO_METHOD_SURF = 2,
    VO_METHOD_LUKASKANADE = 3,
    VO_METHOD_AKAZE = 4
};

class VO
{
public:
    VO();

    void initParameter();
    void setImagePath(char *pathname);
    void visualodometry();
    
private:
    cv::Mat cameraMatrix;

	cv::Mat rVec, tVec;
	cv::Mat rMat, tMat;
	cv::Mat cameraPose, cameraPosition;
    cv::Mat E, mask;
    
    char imagePath[100];
    
    void featureDetection(cv::Mat img1, std::vector<cv::Point2f>& points1, cv::Mat img2, std::vector<cv::Point2f>& points2, int vo_method);
    void fastDetection(cv::Mat img1, std::vector<cv::Point2f>& points1, cv::Mat img2, std::vector<cv::Point2f>& points2);
    void siftDetection(cv::Mat img1, std::vector<cv::Point2f>& points1, cv::Mat img2, std::vector<cv::Point2f>& points2);
    void surfDetection();
    void akazeDetection(cv::Mat img1, std::vector<cv::Point2f>& points1, cv::Mat img2, std::vector<cv::Point2f>& points2);
    void findPoses(cv::Mat prevImg, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
    void drawPoints(cv::Mat img, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2);
    
    // param
    int iterationsCount;
    float reprojectionError;
    double confidence;
    int method;
    
    // fast param
    int fast_threshold;
    bool nonMaxSuppression;
    
    // surf param
    int min_hessian;
    int octave_layer;
    double contrast_threshold;
    double edge_threshold;
    double sigma;
    
    // sift param
    float sift_matching_ratio;
    
    // matching param
    float inlier_threshold; // Distance threshold to identify inliers
    float nn_match_ratio;   // Nearest neighbor matching ratio
};

#endif
