#ifndef FEATUREDETECTION_H
#define FEATUREDETECTION_H

#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <initializer_list>
#include <iostream>

#include "Frame.h"

// TODO: future mod on the class naming
class FeatureDetection
{
public:
    // constructor
	FeatureDetection();

	// destructor
    virtual ~FeatureDetection();

	// set SIFT param
	void setSiftParam(int octave, double contrastThreshold, double edgeThreshold, double sigma, double ratio);


    // the wrapper
    void computeKeypointsAndDraw(char *pathname);

    // methods for feature detections, SIFT, FAST and SURF
    void fastDetector(cv::Mat img, std::vector<cv::KeyPoint>& detectedPoints);
    void siftDetector(cv::Mat img, std::vector<cv::KeyPoint>& detectedPoints);
	void siftDetector(cv::Mat img, std::vector<cv::KeyPoint>& detectedPoints, cv::Mat mask);
    void surfDetector(cv::Mat img, std::vector<cv::KeyPoint>& detectedPoints);

    // methods for extracting the features
    void siftExtraction (cv::Mat img, std::vector<cv::KeyPoint> detectedPoints, cv::Mat &descriptor);
    void surfExtraction (cv::Mat img, std::vector<cv::KeyPoint> detectedPoints, cv::Mat &descriptor);

	// methods for matching descriptors
    void bfMatcher (cv::Mat trainDesc, cv::Mat queryDesc, std::vector<std::vector<cv::DMatch> > &matches);
	void ratioTest (std::vector<std::vector<cv::DMatch> > &, std::vector<int> &, std::vector<int> &);
    void ratioTestRansac (vector<vector<DMatch> > &, Frame &, Frame &, bool);
    // draw keypoints
    void drawKeypoints (cv::Mat img, std::vector<cv::KeyPoint> detectedPoints, cv::Mat &output);

    // return some val
    float getSiftMatchingRatio ();

private:
    // pointer to the feature point detector object
    cv::Ptr<cv::FeatureDetector> siftdetect_;
    cv::Ptr<cv::FeatureDetector> fastdetect_;
    cv::Ptr<cv::FeatureDetector> surfdetect_;

    // pointer to the feature descriptor extractor object
    cv::Ptr<cv::DescriptorExtractor> siftextract_;
    cv::Ptr<cv::DescriptorExtractor> surfextract_;

    // pointer to the matcher object
    cv::Ptr<cv::DescriptorMatcher> matcher_;

    // fast param
    int fast_threshold;
    bool nonMaxSuppression;

    // surf param
    int min_hessian;

	// sift param
    int octave_layer;
    double contrast_threshold;
    double edge_threshold;
    double sigma;
    double sift_matching_ratio;
};

#endif
