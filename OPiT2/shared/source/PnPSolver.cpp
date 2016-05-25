#include "PnPSolver.h"
#include "Converter.h"
#include "Calibration.h"

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

/*
	A class for solving the Perspective-n-Point (PnP) problem.

	solvePnP defined here: http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

	solvePnP outputs the translation in the same unit as we specify world points in (currently the sweref 99).


	Some good stuff
		 http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
		http://ksimek.github.io/2012/08/22/extrinsic/
		http://joelgranados.com/2010/07/30/opencv-camera-extrinsics/
		http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html


*/
PnPSolver::PnPSolver()
{
	setWorldPoints();
	setVoVImagePoints();
}

PnPSolver::PnPSolver(int iterCount, int repError, double confidence)
{
	setWorldPoints();
	setVoVImagePoints();
	paramIterCount  = iterCount;
	paramRepError   = repError;
	paramConfidence = confidence;
}

void PnPSolver::setPnPParam(int iterCount, int repError, double confidence)
{
	paramIterCount  = iterCount;
	paramRepError   = repError;
	paramConfidence = confidence;
}

int PnPSolver::run(int verbalOutput)
{

	Calibration calib;

	Mat cameraMatrix = calib.getCameraMatrix();




	/*
	recoverPose() described here: http://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html#recoverpose
	*/
	// recoverPose(
	// 	essentialMatrix,
	// 	VoVImagePoints[0],
	// 	VoVImagePoints[1],
	// 	R,
	// 	t,
	// 	cameraMatrix.at<double>(0, 0),
	// 	Point2d(cameraMatrix.at<double>(0, 2), cameraMatrix.at<double>(1, 2)),
	// 	mask
	// 	);


	Mat inliers;
	/*
	solvePnPRansac(): Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.

	The function estimates an object pose given a set of object points, their corresponding image projections,
	as well as the camera matrix and the distortion coefficients. This function finds such a pose that minimizes reprojection error,
	that is, the sum of squared distances between the observed projections imagePoints and the projected (using projectPoints() ) objectPoints.
	The use of RANSAC makes the function resistant to outliers.
	The function is parallelized with the TBB library.

	Basically it estimates how the "world" is rotated relative to camera.
	*/
	/*
	Use zero distortion, or the distcoeff calculated by calibrateCamera()?
	*/
	solvePnPRansac(
		Mat(worldPoints),			// Array of world points in the world coordinate space, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel, where N is the number of points.
		Mat(imagePoints),			// Array of corresponding image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, where N is the number of points.
		calib.getCameraMatrix(),				// Self-explanatory...
		calib.getDistortionCoeffs(),// DIST COEFFS, Input vector of distortion coefficients. If null, zero distortion coefficients
		rVec,						// Output rotation vector.   Together with tvec, brings points from the model coordinate system to the camera coordinate system.
		tVec,						// Output translation vector
		false,						// USE EXTRINSIC GUESS, if true (1), the function uses the provided rvec and tvec values as initial approximations
									//						of the rotation and translation vectors, respectively, and further optimizes them.
		paramIterCount,				// ITERATIONS COUNT, number of iterations
		paramRepError,				// REPROJECTION ERROR, inlier threshold value used by the RANSAC procedure.
		paramConfidence,			// CONFIDENCE, The probability that the algorithm produces a useful result. default 0.99;
									//100,				// INLIERS, number of inliers. If the algorithm at some stage finds more inliers than minInliersCount , it finishes.
		inliers,					// INLIERS, output vector that contains indices of inliers in worldPoints and imagePoints.
		SOLVEPNP_ITERATIVE);				// FLAGS, method for solving a PnP problem.

		//Create the rotation matrix from the vector created above, by using the "Rodrigues"
		rMat.create(3, 3, DataType<double>::type);
		Rodrigues(rVec, rMat);
		//Create the translation matrix from the vector created above, by using the "Rodrigues"
		tMat.create(3, 3, DataType<double>::type);
		Rodrigues(tVec, tMat);

		//Instead of only keeping the 4x4 pose matrix, also keep the 3x4
		cameraPose34.create(3, 4, rMat.type());
		// Copies the rotation matrix into the camera pose
		cameraPose34(Range(0, 3), Range(0, 3)) = rMat.t();
		//Copies tvec into the camera pose
		cameraPose34(Range(0, 3), Range(3, 4)) = -(rMat.t()) * tVec;

		/*
		*  Create the camera pose matrix
		*
		*		The [R | t] (3x4) matrix is the extrinsic matrix,
		*			 it describes how the world is transformed relative to the camera
		*				(how to go from world to image, or image to world coordinates).
		*/
		cameraPose.create(4, 4, rMat.type());
		//Copies the rotation matrix into the camera pose
		cameraPose(Range(0, 3), Range(0, 3)) = rMat.t();
		//Copies tvec into the camera pose
		cameraPose(Range(0, 3), Range(3, 4)) = -(rMat.t()) * tVec;
		// Fill the last row of the camera pose matrix with [0, 0, 0, 1]
		double *p = cameraPose.ptr<double>(3);
		p[0] = p[1] = p[2] = 0; p[3] = 1;



		/* undistortPoints giving some funny results, not sure how to use it

		vector<Point2f> uImagePoints;

		undistortPoints(projectedImagePoints, uImagePoints, cameraMatrix, distCoeffs);

		cout << endl << endl << "******************************************" << endl;
		cout << "image points \t\t undistorted points" << endl;
		for (int i = 0; i < uImagePoints.size(); i++)
		{
			cout << imagePoints.at(i) << "\t" << uImagePoints.at(i) << endl;
		}
		cout << "******************************************" << endl;
		*/




		/*	Equation taken from: http://stackoverflow.com/questions/22389896/finding-the-real-world-coordinates-of-an-image-point

		P = Position in world coordinates (assume this is (x, y, z, 1))
		p = Position in image coordinates (assume this is (0, 0, 1))
		R = Rotation matrix    (R^t = R transposed)
		t = translation vector

										      position = R.t * (K^-1 * (u, v, 1)) - t)

		P = R^t (p-t)		Previously:	cameraPosition = rMat.t() * ((cameraMatrix.inv() * coords2D) - tVec);

		Note that transposed rotation (R^t) does the inverse operation to original rotation but is much faster to calculate than  the inverse (R^-1).
		*/
		PnPSolver::cameraPosition.create(3, 1, DataType<double>::type);
		Mat coords2D = (Mat_<double>(3, 1) << 0, 0, 0);

		//cameraPosition = -1 * rMat.t() * tVec;
		PnPSolver::cameraPosition = rMat.t() * ((calib.getCameraMatrix().inv() * coords2D) - tVec);

		camPositions.push_back(PnPSolver::cameraPosition.clone());

		/*
			Taken from: https://en.wikipedia.org/wiki/Essential_matrix#3D_points_from_corresponding_image_points

		Mat rMatRow1 = rMat.row(1), rMatRow3 = rMat.row(2);

		Mat y = (Mat_<float>(1, 3) << VoVImagePoints[0][0].x, VoVImagePoints[0][0].y, 1.0f );
		y.convertTo(y, CV_64FC1);
		*/


		/*
			From: http://www.nature.com/nature/journal/v293/n5828/pdf/293133a0.pdf

			SUPER WEIRD RESULTS
				Guess some of the matrices are used incorrectly.

	 	Mat Z1 = ( (rMatRow1 - (VoVImagePoints[1][0].x * rMatRow3))  * tVec);
		//cout << "Z1 = " << Z1 << endl;
		Mat Z2 = (Mat_<float>(1, 3)); Z2 = ((rMatRow1 - (VoVImagePoints[1][0].x * rMatRow3))  * y.t());
		//cout << "Z2  = " << Z2 << endl;

		Mat Z = Z1 / Z2;


		Mat Y = Z * VoVImagePoints[0][0].y;
		Mat X = Z * VoVImagePoints[0][0].x;


		cout << "Unprimed X Y Z = " << endl << X << endl << Y << endl << Z << endl << endl;

		Mat XX = rMat * (X - tVec);
		Mat YY = rMat * (Y - tVec);
		Mat ZZ = rMat * (Z - tVec);

		cout << "Primed X Y Z = " << endl << XX << endl << endl << YY << endl << endl << ZZ << endl << endl;
		*/


		if(verbalOutput)
		{
	//		cout << endl << "***********************************************" << endl << endl;
	//
	//		cout << "Essential Matrix = " << endl << essentialMatrix << endl << endl;
	//
	//		cout << "Fundamental Matrix = " << endl << fundamentalMatrix << endl << endl;
	//
	//		cout << "CM =" << endl << calib.getCameraMatrix() << endl << endl;
	//
	//		cout << "R =" << endl << rMat << endl << endl;
	//
	//		cout << "T =" << endl << tMat << endl << endl;
	//
	//		cout << "t =" << endl << tVec << endl << endl;
	//
			// cout << "Camera Pose = [" << cameraPose.at<double>(0,3) << ", "
	        //                           << cameraPose.at<double>(1,3) << ", "
	        //                           << cameraPose.at<double>(2,3) << "]" << endl;

			cout << "[" << std::fixed << setprecision(10)
									  << cameraPose.at<double>(0,3) << ", "
									  << cameraPose.at<double>(1,3) << ", "
									  << cameraPose.at<double>(2,3) << "]" << endl;

			// cout << "  camera rcal at frame-["  << cameraPosition.at<double>(0) << ", "
            //                                     << cameraPosition.at<double>(1) << ", "
            //                                     << cameraPosition.at<double>(2) << "]" << endl;

	#include <iomanip>



//		cout << endl << "***********************************************" << endl << endl;
	}

	return 0;
}



// Use default image points, not recommended
void PnPSolver::setImagePoints()
{
	/*
	float	W = 4.8,
		w = 1280.0,
		H = 3.6,
		h = 960,
		f = 1432,
		F = Converter::ImageToWorldF(f, W, w);

	// Converted from pixels to meters
	PnPSolver::imagePoints.push_back(Point2d(Converter::ImageToWorldX(384.3331f, W, w), Converter::ImageToWorldY(162.23618f, H, h)));
	PnPSolver::imagePoints.push_back(Point2d(Converter::ImageToWorldX(385.27521f, W, w), Converter::ImageToWorldY(135.21503f, H, h)));
	PnPSolver::imagePoints.push_back(Point2d(Converter::ImageToWorldX(409.36746f, W, w), Converter::ImageToWorldY(139.30315f, H, h)));
	PnPSolver::imagePoints.push_back(Point2d(Converter::ImageToWorldX(407.43854f, W, w), Converter::ImageToWorldY(165.64435f, H, h)));
	*/
}

void PnPSolver::setImagePoints(vector<Point2d> IP)
{

	PnPSolver::imagePoints = IP;
}

vector<cv::Point2d> PnPSolver::getImagePoints()
{
	return PnPSolver::imagePoints;
}

// Use default world points, not recommended
void PnPSolver::setWorldPoints()
{
	// 3D scene
	PnPSolver::worldPoints.push_back(Point3d(143430.318, 6394363.127, 39.797));
	PnPSolver::worldPoints.push_back(Point3d(143434.166, 6394361.662, 39.842));
	PnPSolver::worldPoints.push_back(Point3d(143432.108, 6394362.287, 39.617));
	PnPSolver::worldPoints.push_back(Point3d(143432.948, 6394364.927, 37.656));
	PnPSolver::worldPoints.push_back(Point3d(143427.658, 6394362.027, 38.376));
	PnPSolver::worldPoints.push_back(Point3d(143436.316, 6394359.472, 38.452));
	PnPSolver::worldPoints.push_back(Point3d(143427.048, 6394361.520, 33.577));
	PnPSolver::worldPoints.push_back(Point3d(143430.465, 6394361.396, 38.098));
	PnPSolver::worldPoints.push_back(Point3d(143437.223, 6394361.204, 39.037));
	PnPSolver::worldPoints.push_back(Point3d(143432.753, 6394362.541, 39.446));

	/*
	PnPSolver::worldPoints.push_back(Point3d(143469.613, 6394456.418, 38.800));
	PnPSolver::worldPoints.push_back(Point3d(143468.953, 6394441.302, 37.281));
	PnPSolver::worldPoints.push_back(Point3d(143466.173, 6394449.469, 38.671));
	PnPSolver::worldPoints.push_back(Point3d(143467.283, 6394451.589, 38.711));
	PnPSolver::worldPoints.push_back(Point3d(143468.983, 6394441.362, 36.431));
	PnPSolver::worldPoints.push_back(Point3d(143427.048, 6394361.520, 33.577));
	*/


	/*
	PnPSolver::worldPoints.push_back(Point3d(143436.316, 6394359.472, 38.452));
	PnPSolver::worldPoints.push_back(Point3d(143427.658, 6394362.027, 38.376));
	PnPSolver::worldPoints.push_back(Point3d(143468.983, 6394441.362, 36.431));
	PnPSolver::worldPoints.push_back(Point3d(143468.953, 6394441.302, 37.281));
	PnPSolver::worldPoints.push_back(Point3d(143462.114, 6394450.099, 38.451));
	PnPSolver::worldPoints.push_back(Point3d(143467.283, 6394451.589, 38.711));
	PnPSolver::worldPoints.push_back(Point3d(143466.173, 6394449.469, 38.671));
	PnPSolver::worldPoints.push_back(Point3d(143469.613, 6394456.418, 38.800));
	*/
}

void PnPSolver::setWorldPoints(vector<Point3d> WP)
{
	PnPSolver::worldPoints = WP;
}

vector<cv::Point3d> PnPSolver::getWorldPoints()
{
	return PnPSolver::worldPoints;
}

Mat PnPSolver::getRotationVector()
{
	return PnPSolver::rVec;
}
Mat PnPSolver::getRotationMatrix()
{
	return PnPSolver::rMat;
}
Mat PnPSolver::getTranslationVector()
{
	return PnPSolver::tVec;
}
Mat PnPSolver::getTranslationMatrix()
{
	return PnPSolver::tMat;
}
cv::Mat PnPSolver::getCameraPose()
{
	return PnPSolver::cameraPose;
}
cv::Mat PnPSolver::getCameraPosition()
{
	return PnPSolver::cameraPosition;
}

cv::Mat PnPSolver::getEssentialMatrix()
{
	Calibration calib;
	Mat mask;
	Mat cameraMatrix = calib.getCameraMatrix();
	/*
	findEssentialMat() declared in: https://github.com/Itseez/opencv/blob/master/modules/calib3d/src/five-point.cpp
	*/
	PnPSolver::essentialMatrix = findEssentialMat(
		VoVImagePoints[0],					// Array of N (N >= 5) 2D points from the first image.The point coordinates should be floating - point(single or double precision).
		VoVImagePoints[1],					// Array of the second image points of the same size and format as points1 .
		cameraMatrix,
		RANSAC,								// Method for computing a fundamental matrix.

		0.99,								// Parameter used for the RANSAC. It specifies a desirable level of confidence (probability) that the estimated matrix is correct.

		3,									// RANSAC threshold, the maximum distance from a point to an epipolar line in pixels,
											//  beyond which the point is considered an outlier and is not used for computing the final fundamental matrix.
											//   It can be set to something like 1 - 3, depending on the accuracy of the point localization, image resolution, and the image noise.

		mask								// Output array of N elements, every element of which is set to 0 for outliers and to 1 for the other points.The array is computed only in the RANSAC and LMedS methods.
	);

	return PnPSolver::essentialMatrix;
}

cv::Mat PnPSolver::getFundamentalMatrix()
{
	PnPSolver::fundamentalMatrix = findFundamentalMat(
		VoVImagePoints[0],					// Take the first vector of points (the first image)
		VoVImagePoints[1],					// ...And the second
		CV_FM_RANSAC						// Use RANSAC
	);

	return PnPSolver::fundamentalMatrix;
}

// The initial frames have different 2D coordinates for the points
void PnPSolver::setVoVImagePoints()
{
	vector<Point2d> imagepoints;
	imagepoints.push_back(Point2d(397.210571, 145.146866));
	imagepoints.push_back(Point2d(650.494934, 129.172379));
	imagepoints.push_back(Point2d(519.567688, 131.898239));
	imagepoints.push_back(Point2d(531.834473, 267.480103));
	imagepoints.push_back(Point2d(239.835358, 207.141220));
	imagepoints.push_back(Point2d(834.740051, 174.580566));
	imagepoints.push_back(Point2d(211.190155, 510.402740));
	imagepoints.push_back(Point2d(437.319458, 218.244186));
	imagepoints.push_back(Point2d(845.259948, 160.413910));
	imagepoints.push_back(Point2d(559.729248, 170.678528));

	vector<Point2d> imagepoints2;
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

	PnPSolver::VoVImagePoints.push_back(imagepoints);
	PnPSolver::VoVImagePoints.push_back(imagepoints2);
}

vector< vector<cv::Point2d> > PnPSolver::getVoVImagePoints()
{
	return PnPSolver::VoVImagePoints;
}
