#include "Frame.h"
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
	edge_threshold = 10;				// default 10, higher value more features
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

	//matcher
	matcher_ 	 = cv::BFMatcher::create("BruteForce");
}


// destructor
FeatureDetection::~FeatureDetection()
{
    // TODO Auto-generated destructor stub
}

void FeatureDetection::setSiftParam(int octave, double contrastThreshold, double edgeThreshold, double sigma, double ratio)
{
	this->octave_layer				= octave;
    this->contrast_threshold		= contrastThreshold;
    this->edge_threshold 			= edgeThreshold;
    this->sigma						= sigma;
    this->sift_matching_ratio		= ratio;

	// reupdate the SIFT
	siftdetect_ = cv::xfeatures2d::SIFT::create(0, octave_layer, contrast_threshold, edge_threshold, sigma);
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
	matcher_->knnMatch(queryDesc, trainDesc, matches, 2);
}

void FeatureDetection::ratioTest (vector<vector<DMatch> > &matches, vector<int> &retrieved3D, vector<int> &retrieved2D)
{
	for (int i=0; i<matches.size(); i++)
	{
		// get the closest neighbour (index 0 always gives the closest descriptor)
		DMatch first = matches[i][0];

		// compare the L2 distance between the closest and the 2nd-closest
		auto dist1 = matches[i][0].distance;
		auto dist2 = matches[i][1].distance;

		// if the 2nd-closest distance is less than the Lowe's ratio (default 0.8)
		if (dist1 < this->getSiftMatchingRatio() * dist2)
		{
			// the 3D points are from the query (lookup table)
			retrieved3D.push_back(first.queryIdx);
			// the 2D points are from current frame keypoints
			retrieved2D.push_back(first.trainIdx);
		}
	}
}

void FeatureDetection::ratioTestRansac (vector<vector<DMatch> > &matches, Frame &prev, Frame &curr, bool verbose)
{
    vector<DMatch> goodMatches, goodMatches2;
    vector<Point2d> prev2D, curr2D;
    vector<int> prev2Drefined, curr2Drefined;

    for (int i=0; i<matches.size(); i++)
    {
        // get the closest neighbour (index 0 always gives the closest descriptor)
        DMatch first = matches[i][0];

        // compare the L2 distance between the closest and the 2nd-closest
        auto dist1 = matches[i][0].distance;
        auto dist2 = matches[i][1].distance;

        if (dist1 < this->getSiftMatchingRatio() * dist2)
        {
            goodMatches.push_back(first);
            prev2D.push_back(prev.reprojectedImagePoints[first.queryIdx]);
            curr2D.push_back(curr.keypoints[first.trainIdx].pt);
        }
    }

    // RANSAC using find homogeneus
    vector<uchar> state;
    findFundamentalMat(prev2D, curr2D, FM_RANSAC, 5, 0.99, state);

    for (size_t i = 0; i<state.size(); ++i)
    {
        // discards outliers (mask == 0)
        if (state[i] != 0)
        {
            goodMatches2.push_back(goodMatches[i]);
            prev2Drefined.push_back(goodMatches[i].queryIdx);
            curr2Drefined.push_back(goodMatches[i].trainIdx);
        }
    }

    if (verbose)
    {
   cout << "    num of RAW matches (SIFT ratio)    : " << goodMatches.size() << endl;
   cout << "    num of REF matches (RANSAC)        : " << goodMatches2.size() << endl;
   cout << "    num of removed outliers            : " << goodMatches.size()-goodMatches2.size() << endl;
   cout << "    outliers in percent (%)            : " << (double)(goodMatches.size()-goodMatches2.size())/(double)goodMatches.size()*100 << endl;
   cout << "    inliers in percent (%)             : " << 100 - (double)(goodMatches.size()-goodMatches2.size())/(double)goodMatches.size()*100 << endl;
   cout << endl;
    }

    // get the world-image correspondences
    for (int i=0; i<goodMatches2.size(); i++)
    {
        // current world points are obtained from the
        curr.matchedWorldPoints.push_back(prev.reprojectedWorldPoints[prev2Drefined[i]]);
        // current image points are obtained from the keypoints
        curr.matchedImagePoints.push_back(curr.keypoints[curr2Drefined[i]].pt);
    }
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
