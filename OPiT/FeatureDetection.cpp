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
	contrast_threshold = 0.04;
	edge_threshold = 10;
	sigma = 1.6;

	// sift
	sift_matching_ratio = 0.8;

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

void FeatureDetection::drawKeypoints (cv::Mat img, std::vector<cv::KeyPoint> detectedPoints, cv::Mat &output)
{
    // copy original image
    img.copyTo(output);

    // draw the keypoints using rich method (draw the radius)
    cv::drawKeypoints(img, detectedPoints, output, Scalar(249, 205, 47), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
}

/*
	Each detect() + compute() of SIFT takes ~3.5 seconds, possible to improve?
*/
void FeatureDetection::foo()
{
	BFMatcher matcher;
	std::vector<DMatch> matches, goodMatches;
//	double max_dist = 0, min_dist = 100;

	Ptr<Feature2D> sfd = xfeatures2d::SIFT::create(
		0,			// int 		nfeatures
		3,			// int 		nOctaveLayers
		0.04,		// double 	contrastThreshold
		10,			// double 	edgeThreshold
		1.6			// double 	sigma
		);


	vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2, imgKeypoints1, imgKeypoints2;

	// Detect keypoints in the frame
	//sfd->detect(frame2, keypoints2);

	// Compute the 128 dimension SIFT descriptor at each keypoint.
	// Each row in "descriptors" correspond to the SIFT descriptor for each keypoint
	//sfd->compute(frame2, keypoints2, descriptors2);

	//drawKeypoints(frame2, keypoints2, imgKeypoints2, Scalar::all(-1), DrawMatchesFlags::DEFAULT);

	//sfd->detect(frame1, keypoints1);
	//sfd->compute(frame1, keypoints1, descriptors1);
}
