#include "PnPSolver.h"
#include "Converter.h"

#include "Mesh.h"
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

/*
	A class for solving the Perspective-n-Point (PnP) problem.

	solvePnP defined here: http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html

	solvePnP outputs the translation in the same unit as we specify world points in (currently the sweref 99?).

	
	Some good stuff
		 http://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
		http://ksimek.github.io/2012/08/22/extrinsic/
		http://joelgranados.com/2010/07/30/opencv-camera-extrinsics/
		http://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_calib3d/py_pose/py_pose.html

	
*/


int PnPSolver::foo(int verbalOutput)
{
	//MEASURE THE TIME
	int64 now, then;
	double elapsedSecondsCALIB, elapsedSecondsPNP, elapsedSecondsPROJ, elapsedSecondsCPOS, ticksPerSecond = cvGetTickFrequency()*1.0e6;

	then = cvGetTickCount();

	
	// Before anything, we try to calibrate the camera
	vector< vector<Point3f> > VoVWorldPoints{ worldPoints, worldPoints };
	distCoeffs = Mat::zeros(8, 1, CV_64F);
	Mat rvec, tvec;
	calibrateCamera(	
						VoVWorldPoints,					// A vector of vectors with the world points
						VoVImagePoints,					// A vector of vectors with the image points
						Size(1280,960),					// The image size
						cameraMatrix,					// obvious...
						distCoeffs,						// The distortion coeffiecients for the camera
						rvec, tvec,						//
						CV_CALIB_USE_INTRINSIC_GUESS);	// Optional flags	

	// Calculate time
	now = cvGetTickCount();
	elapsedSecondsCALIB = (double)(now - then) / ticksPerSecond;

	cout << "Distcoeffs = " << endl << distCoeffs << endl << "**********" << endl;



	// Keep track of time
	then = cvGetTickCount();
		
	Mat inliers;
	cout << "Size of worldPoints = " << worldPoints.size() << endl << "Size of imagePoints = " << PnPSolver::imagePoints.size() << endl << endl;
	/*
		solvePnPRansac(): Finds an object pose from 3D-2D point correspondences using the RANSAC scheme.

		The function estimates an object pose given a set of object points, their corresponding image projections, 
		as well as the camera matrix and the distortion coefficients. This function finds such a pose that minimizes reprojection error, 
		that is, the sum of squared distances between the observed projections imagePoints and the projected (using projectPoints() ) objectPoints. 
		The use of RANSAC makes the function resistant to outliers. 
		The function is parallelized with the TBB library.

		Basically it estimates how the "world" is rotated relative to camera.
	*/
	solvePnPRansac(
		Mat(worldPoints),	// Array of world points in the world coordinate space, 3xN/Nx3 1-channel or 1xN/Nx1 3-channel, where N is the number of points.
		Mat(imagePoints),	// Array of corresponding image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel, where N is the number of points.
		cameraMatrix,		// Self-explanatory...
		distCoeffs,			// DIST COEFFS, Input vector of distortion coefficients. If null, zero distortion coefficients 
		rVecIter,			// Output rotation vector.   Together with tvec, brings points from the model coordinate system to the camera coordinate system.
		tVecIter,			// Output translation vector
		false,				// USE EXTRINSIC GUESS, if true (1), the function uses the provided rvec and tvec values as initial approximations
							//						of the rotation and translation vectors, respectively, and further optimizes them.
		100,				// ITERATIONS COUNT, number of iterations
		8,					// REPROJECTION ERROR, inlier threshold value used by the RANSAC procedure.
		0.95,				// CONFIDENCE, The probability that the algorithm produces a useful result. default 0.99;
		//100,				// INLIERS, number of inliers. If the algorithm at some stage finds more inliers than minInliersCount , it finishes.
		inliers,			// INLIERS, output vector that contains indices of inliers in worldPoints and imagePoints.
		CV_ITERATIVE);		// FLAGS, method for solving a PnP problem.

	// Calculate time
	now = cvGetTickCount();
	elapsedSecondsPNP = (double)(now - then) / ticksPerSecond;
		
	//Create the rotation matrix from the vector created above, by using the "Rodrigues"
	rMatIter.create(3, 3, DataType<double>::type);
	Rodrigues(rVecIter, rMatIter);
	//Create the translation matrix from the vector created above, by using the "Rodrigues"
	tMatIter.create(3, 3, DataType<double>::type);
	Rodrigues(tVecIter, tMatIter);


	/*
	*  Create the camera pose matrix
	*
	*		The [R | t] (3x4) matrix is the extrinsic matrix,
	*			 it describes how the world is transformed relative to the camera
	*				(how to go from world to image, or image to world coordinates).
	*/
	cameraPose.create(4, 4, rMatIter.type());
	//Copies the rotation matrix into the camera pose
	cameraPose(Range(0, 3), Range(0, 3)) = rMatIter.t();
	//Copies tvec into the camera pose
	cameraPose(Range(0, 3), Range(3, 4)) = -(rMatIter.t()) * tVecIter;
	// Fill the last row of the camera pose matrix with [0, 0, 0, 1]
	double *p = cameraPose.ptr<double>(3);
	p[0] = p[1] = p[2] = 0; p[3] = 1;

	then = cvGetTickCount();

	projectPoints(						// Projects 3D points to an image plane.
		worldPoints,					//
		rVecIter,						// Rotation vector
		tVecIter,						// Translation vector
		cameraMatrix,					//
		distCoeffs,						// Input vector of distortion coefficients. If the vector is NULL/empty, the zero distortion coefficients are assumed.
		projectedImagePoints,			// Output array of image points, 2xN/Nx2 1-channel or 1xN/Nx1 2-channel
		projectionJacobian				// Jacobian, jacobian matrix of derivatives of image points with respect to components of the rotation vector, 
										// translation vector, focal lengths, coordinates of the principal point and the distortion coefficients.

										// Aspect ratio, optional (fx/fy)
		);
	
	now = cvGetTickCount();
	elapsedSecondsPROJ = (double)(now - then) / ticksPerSecond;



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


	/*
	NOT BEING USED

	Assume:
	m = camera position in image coordinates
	A = camera matrix
	M = camera position in world coordinates
	then:
	m = A [R | t] M

	Thus, if we want the world coordinates of the camera:
	M = m....
	*/


	cameraPosition.create(3, 1, DataType<double>::type);

	then = cvGetTickCount();

	/*	Equation taken from: http://stackoverflow.com/questions/22389896/finding-the-real-world-coordinates-of-an-image-point

		P = Position in world coordinates (assume this is (x, y, z, 1))
		p = Position in image coordinates (assume this is (0, 0, 1))
		R = Rotation matrix    (R^t = R transposed)
		t = translation vector

		P = R^t (p-t)

		Note that transposed rotation (R^t) does the inverse operation to original rotation but is much faster to calculate than  the inverse (R^-1).
	*/
	cameraPosition = rMatIter.t() * ((cameraMatrix.inv() * (Mat_<double>(3, 1) << 0,0,1)) - tVecIter);
	
	now = cvGetTickCount();
	elapsedSecondsCPOS = (double)(now - then) / ticksPerSecond;
	
	//Normalisation of 2D coords
	Mat cameraPositionNorm = Mat(3, 1, DataType<double>::type);
	cameraPositionNorm.at<double>(0, 0) = cameraPosition.at<double>(0, 0) / cameraPosition.at<double>(2, 0);
	cameraPositionNorm.at<double>(1, 0) = cameraPosition.at<double>(1, 0) / cameraPosition.at<double>(2, 0);
	cameraPositionNorm.at<double>(2, 0) = 1;

	

	if(verbalOutput)
	{
		cout << endl << "***********************************************" << endl << endl;

		cout << "CM =" << endl << cameraMatrix << endl << endl;

		cout << "R =" << endl << rMatIter << endl << endl;

		cout << "T =" << endl << tMatIter << endl << endl;

		cout << "t =" << endl << tVecIter << endl << endl;

		cout << "Camera pose: " << endl << cameraPose << endl << endl;
		
		cout << "cPos = \t" << endl << cameraPosition << endl << endl;
		cout << "cPos norm = \t" << endl << cameraPositionNorm << endl << endl;
		
		cout << "Given image point\tProjected image point" << endl;
		for (int i = 0; i < projectedImagePoints.size(); i++)
		{
			cout << imagePoints.at(i) << "\t" << projectedImagePoints.at(i) << endl;
		}
		cout << endl;
		cout << "Calibrating camera position took " << elapsedSecondsCALIB << " seconds" << endl;
		cout << "PnP (ITERATIVE) took " << elapsedSecondsPNP << " seconds" << endl;
		cout << "Projection of image points took " << elapsedSecondsPROJ << " seconds" << endl;
		cout << "Calculating camera position took " << elapsedSecondsCPOS << " seconds" << endl;
		


		cout << endl << "***********************************************" << endl << endl;
	}

	

	return 0;
}

PnPSolver::PnPSolver()											// WE HAVE TO TAKE DISTORTION INTO ACCOUNT!
{																//							    Camera matrix
	cameraMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0));			//								___		  ___
	cameraMatrix.at<double>(0, 0) = 1432;						// Focal length X				| fx  0  cx |
	cameraMatrix.at<double>(1, 1) = 1432;						// Focal length Y				| 0  fy  cy |
	cameraMatrix.at<double>(0, 2) = 640;						// Principal point X			| 0   0   1 |
	cameraMatrix.at<double>(1, 2) = 481;						// Principal point Y			---		  ---
	cameraMatrix.at<double>(2, 2) = 1.0;						// Just a 1 cause why not	

}

PnPSolver::PnPSolver(Mat CM)
{
	if (cameraMatrix.rows != 3)
		cerr << "WRONG NR OF ROWS, MUST BE 3" << endl;
	else if (cameraMatrix.cols != 3)
		cerr << "WRONG NR OF COLUMNS, MUST BE 3" << endl;

	cameraMatrix.copyTo(PnPSolver::cameraMatrix);

	
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

void PnPSolver::setImagePoints(vector<Point2f> IP)
{
	
	PnPSolver::imagePoints = IP;
}

vector<cv::Point2f> PnPSolver::getImagePoints()
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
	PnPSolver::worldPoints.push_back(Point3d(143462.114, 6394450.099, 38.451));
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

void PnPSolver::setWorldPoints(vector<Point3f> WP)
{
	PnPSolver::worldPoints = WP;
}

vector<cv::Point3f> PnPSolver::getWorldPoints()
{
	return PnPSolver::worldPoints;
}


Mat PnPSolver::getRotationVector()
{
	return PnPSolver::rVecIter;
}
Mat PnPSolver::getRotationMatrix()
{
	return PnPSolver::rMatIter;
}
Mat PnPSolver::getTranslationVector()
{
	return PnPSolver::tVecIter;
}
Mat PnPSolver::getTranslationMatrix()
{
	return PnPSolver::tMatIter;
}
cv::Mat PnPSolver::getCameraPose()
{
	return PnPSolver::cameraPose;
}
cv::Mat PnPSolver::getCameraPosition()
{
	return PnPSolver::cameraPosition;
}
vector<cv::Point2f> PnPSolver::getProjectedImagePoints()
{
	return PnPSolver::projectedImagePoints;
}
cv::Mat PnPSolver::getProjectionJacobian()
{
	return PnPSolver::projectionJacobian;
}
Mat PnPSolver::getCameraMatrix()
{
	return PnPSolver::cameraMatrix;
}

void PnPSolver::setVoVImagePoints()
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

	VoVImagePoints.push_back(imagepoints);
	VoVImagePoints.push_back(imagepoints2);

	
}