#include "FeatureDetection.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>


#include <stdlib.h>     //for using the function sleep
#include <iostream>
#include <thread>
#include <iomanip>

#define IMG_ENTRANCE    433
#define IMG_EARLY       503
#define IMG_INSIDE      563

using namespace cv;
using namespace std;

//Handy function for finding type of image
string type2str(int type);

FeatureDetection::FeatureDetection()
{
	// fast
	fast_threshold = 20;
	nonMaxSuppression = true;

	// surf
	min_hessian = 200;
	octave_layer = 3;
	contrast_threshold = 0.01;			// default 0.04, lower value more features
	contrast_threshold = 0.02;			// default 0.04, lower value more features
	edge_threshold = 10;				// default 10, higher value more features
	sigma = 1.6;

	// sift
	sift_matching_ratio = 0.6;

	//detector
	fastdetect_ = FastFeatureDetector::create(
		10,			// int		threshold on difference between intensity of the central pixel and pixels of a circle around this pixel.
		true,		// bool		nonmaxSuppression. If true, non-maximum suppression is applied to detected corners (keypoints).
		2			// int		type, one of the three neighborhoods as defined in the paper:	FastFeatureDetector::TYPE_9_16,
					//																			FastFeatureDetector::TYPE_7_12,
					//																			FastFeatureDetector::TYPE_5_8
		);

	siftdetect_ = cv::xfeatures2d::SIFT::create(0, octave_layer, contrast_threshold, edge_threshold, sigma);


	surfdetect_ = xfeatures2d::SURF::create(
		100,	// double	hessianThreshold	for hessian keypoint detector used in SURF
		4,		// int		nOctaves			number of pyramid octaves the keypoint detector will use
		3,		// int		nOctaveLayers		number of octave layers within each octave
		false,	// bool		extended			extended descriptor flag (true - use extended 128-element descriptors; false - use 64-element descriptors)
		false	// bool		upright				up-right or rotated features flag (true - do not compute orientation of features; false - compute orientation)
		);

	//extractor
	siftextract_ = cv::xfeatures2d::SIFT::create(0, octave_layer, contrast_threshold, edge_threshold, sigma);
	surfextract_ = cv::xfeatures2d::SURF::create(min_hessian);

	//matcher
	matcher_ 	 = cv::BFMatcher::create("BruteForce");
}


// destructor
FeatureDetection::~FeatureDetection()
{
    // TODO Auto-generated destructor stub
}

void FeatureDetection::computeKeypointsAndDraw(char *pathname)
{
    // load image
    char imagenameentr[100];
    char imagenameearl[100];
    char imagenameinsi[100];

    sprintf(imagenameentr, "%simg_%05d.png", pathname, IMG_ENTRANCE);
    sprintf(imagenameearl, "%simg_%05d.png", pathname, IMG_EARLY);
    sprintf(imagenameinsi, "%simg_%05d.png", pathname, IMG_INSIDE);

    Mat img0 = imread(imagenameentr);
    Mat img1 = imread(imagenameearl);
    Mat img2 = imread(imagenameinsi);

    // work with grayscale only
    cvtColor(img0, img0, COLOR_BGR2GRAY);

    vector<KeyPoint> fastpoints, surfpoints, siftpoints;
    vector<KeyPoint> fastpoints1, surfpoints1, siftpoints1;
    vector<KeyPoint> fastpoints2, surfpoints2, siftpoints2;

    // fast detection
    fastdetect_->detect(img0, fastpoints);
    fastdetect_->detect(img1, fastpoints1);
    fastdetect_->detect(img2, fastpoints2);

    cout << "FAST: " << endl;
    cout << "entra keypoints: " << fastpoints.size() << endl;
    cout << "early keypoints: " << fastpoints1.size() << endl;
    cout << "insid keypoints: " << fastpoints2.size() << endl;

    // surf detection
    surfdetect_->detect(img0, surfpoints);
    surfdetect_->detect(img1, surfpoints1);
    surfdetect_->detect(img2, surfpoints2);

    cout << "SURF: " << endl;
    cout << "entra keypoints: " << surfpoints.size() << endl;
    cout << "early keypoints: " << surfpoints1.size() << endl;
    cout << "insid keypoints: " << surfpoints2.size() << endl;

    // sift detection
    siftdetect_->detect(img0, siftpoints);
    siftdetect_->detect(img1, siftpoints1);
    siftdetect_->detect(img2, siftpoints2);

    cout << "SIFT: " << endl;
    cout << "entra keypoints: " << siftpoints.size() << endl;
    cout << "early keypoints: " << siftpoints1.size() << endl;
    cout << "insid keypoints: " << siftpoints2.size() << endl;

    // draw
    namedWindow("fast-entrance", WINDOW_NORMAL);
    namedWindow("fast-early", WINDOW_NORMAL);
    namedWindow("fast-inside", WINDOW_NORMAL);

    namedWindow("surf-entrance", WINDOW_NORMAL);
    namedWindow("surf-early", WINDOW_NORMAL);
    namedWindow("surf-inside", WINDOW_NORMAL);

    namedWindow("sift-entrance", WINDOW_NORMAL);
    namedWindow("sift-early", WINDOW_NORMAL);
    namedWindow("sift-inside", WINDOW_NORMAL);

    Mat fast0, fast1, fast2;
    Mat surf0, surf1, surf2;
    Mat sift0, sift1, sift2;

    drawKeypoints(img0, fastpoints,  fast0);
    drawKeypoints(img1, fastpoints1, fast1);
    drawKeypoints(img2, fastpoints2, fast2);

    drawKeypoints(img0, surfpoints, surf0);
    drawKeypoints(img1, surfpoints1, surf1);
    drawKeypoints(img2, surfpoints2, surf2);

    drawKeypoints(img0, siftpoints, sift0);
    drawKeypoints(img1, siftpoints1, sift1);
    drawKeypoints(img2, siftpoints2, sift2);

    imshow("fast-entrance", fast0);
    imshow("fast-early", fast1);
    imshow("fast-inside", fast2);

    imshow("surf-entrance", surf0);
    imshow("surf-early", surf1);
    imshow("surf-inside", surf2);

    imshow("sift-entrance", sift0);
    imshow("sift-early", sift1);
    imshow("sift-inside", sift2);

    // write to file
    vector<int> compression_params;
    compression_params.push_back(IMWRITE_PNG_COMPRESSION);
    compression_params.push_back(9);

    imwrite("fast-entrance.png", fast0, compression_params);
    imwrite("fast-early.png",    fast1, compression_params);
    imwrite("fast-inside.png",   fast2, compression_params);

    imwrite("surf-entrance.png", surf0, compression_params);
    imwrite("surf-early.png",    surf1, compression_params);
    imwrite("surf-inside.png",   surf2, compression_params);

    imwrite("sift-entrance.png", sift0, compression_params);
    imwrite("sift-early.png",    sift1, compression_params);
    imwrite("sift-inside.png",   sift2, compression_params);

    cv::waitKey(1);
}

void FeatureDetection::fastDetector(cv::Mat img, std::vector<cv::KeyPoint> &detectedPoints)
{
    // detect keypoints using fast
    fastdetect_->detect(img, detectedPoints);
}

void FeatureDetection::surfDetector(cv::Mat img, std::vector<cv::KeyPoint> &detectedPoints)
{
    // detect keypoints using surf
    surfdetect_->detect(img, detectedPoints);
}

void FeatureDetection::siftDetector(cv::Mat img, std::vector<cv::KeyPoint> &detectedPoints)
{
    // detect keypoints using sift
    siftdetect_->detect(img, detectedPoints);
}

void FeatureDetection::siftDetector(cv::Mat img, std::vector<cv::KeyPoint> &detectedPoints, cv::Mat mask)
{
    // detect keypoints using sift
    siftdetect_->detect(img, detectedPoints, mask);
}

void FeatureDetection::surfExtraction(cv::Mat img, std::vector<cv::KeyPoint> detectedPoints, cv::Mat &descriptor)
{
    // extract descriptor from keypoints using surf
    surfdetect_->compute(img, detectedPoints, descriptor);
}

void FeatureDetection::siftExtraction (cv::Mat img, std::vector<cv::KeyPoint> detectedPoints, cv::Mat &descriptor)
{
    // extract descriptor from keypoints using sift
    siftdetect_->compute(img, detectedPoints, descriptor);
}

void FeatureDetection::bfMatcher (cv::Mat queryDesc, cv::Mat trainDesc, std::vector<std::vector<DMatch> > &matches)
{
	// matching using BF L2
	matcher_->knnMatch(queryDesc, trainDesc, matches, 10);
}

void FeatureDetection::drawKeypoints (cv::Mat img, std::vector<cv::KeyPoint> detectedPoints, cv::Mat &output)
{
    // copy original image
    img.copyTo(output);

    // draw the keypoints using rich method (draw the radius)
    cv::drawKeypoints(img, detectedPoints, output, Scalar(249, 205, 47), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

float FeatureDetection::getSiftMatchingRatio ()
{
    return sift_matching_ratio;
}

void FeatureDetection::computeFeaturesAndMatching(Mat img, vector<Point2d> tunnel2D, vector<Point3d> tunnel3D, Mat tunnelDescriptor, int frameCounter,
                                                  vector<KeyPoint> *detectedkpts, Mat *descriptor,
                                                  vector<Point2d> *retrieved2D, vector<Point3d> *retrieved3D)
{
    Mat img_maskUpperPart = Mat::zeros(img.size(), CV_8U);
    Mat img_roiUpperPart (img_maskUpperPart, Rect(0, 0, img.cols, img.rows/2));
    img_roiUpperPart = Scalar(255, 255, 255);
	Mat img_maskRightPart = Mat::zeros(img.size(), CV_8U);
	Mat img_roiRightPart (img_maskRightPart, Rect(img.cols*3/5, img.rows/2, img.cols*2/5, img.rows*2/5));
	img_roiRightPart = Scalar(255, 255, 255);
	Mat img_combinedMask = img_maskUpperPart | img_maskRightPart;
    this->siftDetector(img, *detectedkpts, img_combinedMask);
    this->siftExtraction(img, *detectedkpts, *descriptor);
    vector<vector<DMatch> > matches;
    this->bfMatcher(tunnelDescriptor, *descriptor, matches);
    vector<int> matchedIndices;
    vector<int> matchedXYZ;
    vector<Point2d> lutPt, framePt;
    vector<DMatch> goodMatches,goodMatches2;
    for (int i = 0; i < matches.size(); ++i)
    {
        DMatch first  = matches[i][0];
        auto dist1 = matches[i][0].distance;
        auto dist2 = matches[i][1].distance;
        if(dist1 < this->getSiftMatchingRatio() * dist2)
        {
            goodMatches.push_back(first);
            matchedIndices.push_back(first.trainIdx);
            matchedXYZ.push_back(first.queryIdx);
            framePt.push_back((*detectedkpts)[first.trainIdx].pt);
            lutPt.push_back(tunnel2D[first.queryIdx]);
        }
    }

    // --------------------------------------------------------ransac
    if (frameCounter > 0)
    {
        vector<uchar> state;
        findFundamentalMat(lutPt, framePt, FM_RANSAC, 5, 0.99, state);
        vector<Point2d> lutPtRefined, framePtRefined;
        for (size_t i = 0; i < state.size(); ++i)
        {
            if (state[i] != 0)
            {
                goodMatches2.push_back(goodMatches[i]);
                matchedIndices.push_back(goodMatches[i].trainIdx);
                matchedXYZ.push_back(goodMatches[i].queryIdx);
                lutPtRefined.push_back(lutPt[i]);
                framePtRefined.push_back(framePt[i]);
            }
        }

        cout << "num of RAW matches (SIFT ratio)    : " << goodMatches.size() << endl;
        cout << "num of REF matches (RANSAC)        : " << goodMatches2.size() << endl;
        cout << "num of removed outliers            : " << goodMatches.size()-goodMatches2.size() << endl;
        cout << "outliers in percent (%)            : " << (double)(goodMatches.size()-goodMatches2.size())/(double)goodMatches.size()*100 << endl;
        cout << "inliers in percent (%)             : " << 100 - (double)(goodMatches.size()-goodMatches2.size())/(double)goodMatches.size()*100 << endl;
    }
    // --------------------------------------------------------ransac

    (*retrieved2D).clear(); (*retrieved3D).clear();
    for (int i = 0; i< matchedIndices.size(); i++)
    {
        (*retrieved2D).push_back(Point2d((*detectedkpts)[matchedIndices[i]].pt.x,(*detectedkpts)[matchedIndices[i]].pt.y));
        (*retrieved3D).push_back(Point3d(tunnel3D[matchedXYZ[i]].x,tunnel3D[matchedXYZ[i]].y,tunnel3D[matchedXYZ[i]].z));
    }
}
