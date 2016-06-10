#include "Calibration.h"

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>

using namespace cv;
using namespace std;



Calibration::Calibration()										// WE HAVE TO TAKE DISTORTION INTO ACCOUNT!
{																//							    Camera matrix
	cameraMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0));			//								___		  ___
    cameraMatrix.at<double>(0, 0) = 1432;						// Focal length X				| fx  0  cx |
    cameraMatrix.at<double>(1, 1) = 1432;						// Focal length Y				| 0  fy  cy |
    cameraMatrix.at<double>(0, 2) = 640;						// Principal point X			| 0   0   1 |
    cameraMatrix.at<double>(1, 2) = 481;						// Principal point Y			---		  ---
    cameraMatrix.at<double>(2, 2) = 1.0;						// Just a 1 cause why not
//	cameraMatrix.at<double>(0, 0) = 1693.87882;						// Focal length X				| fx  0  cx |
//	cameraMatrix.at<double>(1, 1) = 1695.69754;						// Focal length Y				| 0  fy  cy |
//	cameraMatrix.at<double>(0, 2) = 1009.89572;						// Principal point X			| 0   0   1 |
//	cameraMatrix.at<double>(1, 2) = 507.91942;						// Principal point Y			---		  ---
//	cameraMatrix.at<double>(2, 2) = 1.0;						// Just a 1 cause why not
//	distortionCoeffs = (Mat1d(5,1) << -0.28197, 0.11386, 0.00131, 0.0013, 0.0000);

	setVoVImagePoints();
	setVoVWorldPoints();
}



Mat Calibration::getCameraMatrix()
{
	return Calibration::cameraMatrix;
}

Mat Calibration::foo( vector<vector<Point2f> > vovIP, vector<vector<Point3f> > vovWP)
{

	// Before anything, we try to calibrate the camera
	Mat distCoeffs = Mat::zeros(8, 1, CV_64F);

	// Make a copy of the camera matrix so as to not overwrite the original
	Mat cm; cameraMatrix.copyTo(cm);
	Mat rvec, tvec;

	calibrateCamera(
		vovWP,							// A vector of vectors with the world points
		vovIP,							// A vector of vectors with the image points
		Size(1280, 960),				// The image size
		cm,								// Output camera matrix, but we have already defined ours
		distCoeffs,						// The distortion coeffiecients for the camera
		rvec, tvec,						//
		CV_CALIB_USE_INTRINSIC_GUESS);	// Optional flags


	return distCoeffs;
}

Mat Calibration::foo()
{
	// Before anything, we try to calibrate the camera
	Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
	Mat cm; cameraMatrix.copyTo(cm);
	Mat rvec, tvec;

	// Get the distortion coefficients by looking at image + world points, and the camera matrix
	calibrateCamera(
		Calibration::VoVWorldPoints,	// A vector of vectors with the world points
		Calibration::VoVImagePoints,	// A vector of vectors with the image points
		Size(1280, 960),				// The image size
		cameraMatrix,					// Output camera matrix, but we have already defined ours
		Calibration::distortionCoeffs,	// The distortion coeffiecients for the camera
		rvec, tvec,						//
		CV_CALIB_USE_INTRINSIC_GUESS);	// Optional flags

										// Calculate time


	return distortionCoeffs;
}

Mat Calibration::getDistortionCoeffs()
{
	return Calibration::distortionCoeffs;
}

void Calibration::pushToVoVWorldPoints(vector<Point3f> WP)
{
	Calibration::VoVWorldPoints.push_back(WP);
}

void Calibration::pushToVoVImagePoints(vector<Point2f> IP)
{
	Calibration::VoVImagePoints.push_back(IP);
}

// Both initial frames use the same 3D points
void Calibration::setVoVWorldPoints()
{
	vector<Point3f> worldPoints;
	worldPoints.push_back(Point3d(143430.318, 6394363.127, 39.797));
	worldPoints.push_back(Point3d(143434.166, 6394361.662, 39.842));
	worldPoints.push_back(Point3d(143432.108, 6394362.287, 39.617));
	worldPoints.push_back(Point3d(143432.948, 6394364.927, 37.656));
	worldPoints.push_back(Point3d(143427.658, 6394362.027, 38.376));
	worldPoints.push_back(Point3d(143436.316, 6394359.472, 38.452));
	worldPoints.push_back(Point3d(143427.048, 6394361.520, 33.577));
	worldPoints.push_back(Point3d(143430.465, 6394361.396, 38.098));
	worldPoints.push_back(Point3d(143437.223, 6394361.204, 39.037));
	worldPoints.push_back(Point3d(143432.753, 6394362.541, 39.446));

	Calibration::VoVWorldPoints.push_back(worldPoints);
	Calibration::VoVWorldPoints.push_back(worldPoints);
}

// The initial frames have different 2D coordinates for the points
void Calibration::setVoVImagePoints()
{
	vector<cv::Point2f> imagepoints;
	imagepoints.push_back(Point2d(397.210571, 145.146866));
	imagepoints.push_back(Point2d(650.494934, 129.172379));
	imagepoints.push_back(Point2d(519.567688, 131.898239));
	imagepoints.push_back(Point2d(531.834473, 267.480103));
	imagepoints.push_back(Point2d(239.835358, 207.141220));
	imagepoints.push_back(Point2d(834.740051, 174.580566));
	imagepoints.push_back(Point2d(211.190155, 510.402740));
	imagepoints.push_back(Point2d(437.319458, 218.244186));
	imagepoints.push_back(Point2d(845.259948, 160.41391));
	imagepoints.push_back(Point2d(559.729248, 170.678528));

	vector<cv::Point2f> imagepoints2;
	imagepoints2.push_back(Point2d(490, 250));
	imagepoints2.push_back(Point2d(668, 242));
	imagepoints2.push_back(Point2d(578, 242));
	imagepoints2.push_back(Point2d(582, 335));
	imagepoints2.push_back(Point2d(380, 294));
	imagepoints2.push_back(Point2d(793, 278));
	imagepoints2.push_back(Point2d(368, 503));
	imagepoints2.push_back(Point2d(521, 306));
	imagepoints2.push_back(Point2d(806, 262));
	imagepoints2.push_back(Point2d(604, 272));

	Calibration::VoVImagePoints.push_back(imagepoints);
	Calibration::VoVImagePoints.push_back(imagepoints2);


}
