/*
	AUTHOR: DAVID BENNEHAG

	Requires OpenCV (tested on 3.1)

		
*/
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

#include "SIFTdetector.h"
#include "Calibration.h"
#include "PnPSolver.h"
#include "PointProjection.h"
#include "Converter.h"
#include "VisualOdometry.h"

#include <iostream>

using namespace std;
using namespace cv;

Mat LinearLSTriangulation(Point3d u, Matx34d P,	Point3d u1,	Matx34d P1);
Mat_<double> IterativeLinearLSTriangulation(Point3d u, Matx34d P, Point3d u1, Matx34d P1);

// initiate the vector input

vector<Point2f> setInputOne();
vector<Point2f> setInputTwo();

int main(int argc, char** argv)
{

	//unsigned cts = thread::hardware_concurrency();
	//cout << cts << " concurrent threads are supported\n";

    //VISUAL ODOMETRY
    VO visodo;
    
    char pathname[100];
    sprintf(pathname, "/Users/januaditya/Workspace/_opencv/ocv-voup/images/raw/");
    
    visodo.setImagePath(pathname);
    visodo.initParameter();
    visodo.visualodometry();
    
	//MEASURE THE TIME
	int64 now, then;
	double  ticksPerSecond = cvGetTickFrequency()*1.0e6;

	PnPSolver solver1, solver2;
	
	Calibration calib;
	calib.setVoVImagePoints();
	calib.setVoVWorldPoints();
	// Get the distortion coefficients by looking at image + world points, and the camera matrix initialized in the pose solver
    
    Mat distcoeffs;
#ifdef DISTORT_ENABLE
    distcoeffs = calib.foo(solver1.getCameraMatrix());
#else
    distcoeffs = Mat(1,5, CV_64FC1, 0.0);
#endif

	cout << "Starting first solver............." << endl << endl << endl;
    
    solver1.setImagePoints(setInputOne());
	
	solver1.setWorldPoints();
	// Run the PnP Solver. All matrices and stuff will be set up after this.
	solver1.foo(1, distcoeffs);

	cout << endl << endl << endl << endl << endl << endl;
	cout << "Starting second solver............." << endl << endl << endl;
	// Take another image showing the same scene.
	//  So the 2D image points will be different of course...
    
	solver2.setImagePoints(setInputTwo());
	//  But the 3D world points will be the same.
	solver2.setWorldPoints();
	solver2.foo(1, distcoeffs);

	cout << endl << endl << endl << endl << endl << endl;
	cout << endl << endl << "camera 1 position: " << endl << solver1.getCameraPosition() << endl << "camera 2 position: " << endl << solver2.getCameraPosition() << endl;
	
	// Should use undistortPoints() and then pass into correctMatches() to refine the coordinates as they were handpicked?
	//   Discussed here: http://www.answers.opencv.org/question/3275/is-triangulatepoints-outputing-rubish/

	// Is triangulatePoints() the next step?   (Points need to be "undistorted" first. Either remove distortion before picking points, or undistort the points)

	// The triangulation in OpenCV is apparently crap, should use equations from H&Z instead,
	//    as described here: http://www.morethantechnical.com/2012/01/04/simple-triangulation-with-opencv-from-harley-zisserman-w-code/#more-1023
	// Or use the example here: http://stackoverflow.com/questions/16295551/how-to-correctly-use-cvtriangulatepoints


	/*
	triangulatePoints: Reconstructs points by triangulation.
	*/
	
	Mat cam1 = solver1.getCameraMatrix().inv() * solver1.cameraPose34;
	Mat cam2 = solver2.getCameraMatrix().inv() * solver2.cameraPose34;
	
	vector<Point2f> cam1Points = solver1.getImagePoints();
	vector<Point2f> cam2Points = solver2.getImagePoints();
	/*
	Mat points3D;

	triangulatePoints(
		cam1,			// 3x4 projection matrix of the first camera.
		cam2,			// 3x4 projection matrix of the second camera.
		cam1Points,		// 2xN array of feature points in the first image. In case of c++ version it can be also a vector of feature points or two - channel matrix of size 1xN or Nx1.
		cam2Points,		// 2xN array of corresponding points in the second image. In case of c++ version it can be also a vector of feature points or two - channel matrix of size 1xN or Nx1.
		points3D		// 4xN array of reconstructed points in homogeneous coordinates.
		);

	cout << endl << "size of points3D: " << points3D.size() << endl;
	cout << endl << points3D << endl;
	cout << endl << "****************************" <<  endl;
	*/
	
	//An alternative to the OpenCV triangulatePoints()
	Point3d u; u.x = cam1Points[0].x; u.y = cam1Points[0].y; u.z = 1;
	Point3d u1; u1.x = cam1Points[0].x; u1.y = cam1Points[0].y; u1.z = 1;
	Mat_<double> X_ = IterativeLinearLSTriangulation(u, cam1, u1, cam2);
	cout << X_ << endl;

	
	cout << endl << endl << endl << endl << endl << endl;
	cout << "Fundamental Mat = " << endl << findFundamentalMat(solver1.getImagePoints(), solver2.getImagePoints(), CV_FM_RANSAC) << endl << endl;
	
	
	SIFTdetector::foo();

	int counter = 0;

	string filename = "CADS4_OWG740_20140422_162231_005.avi";
	VideoCapture vc(filename);
	if (!vc.isOpened())
		exit(EXIT_FAILURE);

	Mat frame1, frame2;

	while (vc.read(frame2))
	{
		// Keep track of time
		then = cvGetTickCount();
		
		// Change brightness
		//frame2 = frame2 + Scalar(10,10,10);

		cout << "Iteration # " << counter << endl;

		//resize(frame2, frame2, Size(frame2.cols / 2, frame2.rows / 2));
		//In the first iteration, only frame2 will contain a frame, so skip this
		if (counter == 0)
		{
			counter++;
			frame1 = frame2.clone();
			continue;
		}


		Mat frame1g, frame2g;
		// For StereoBM, images have to be CV_8UC1, original was CV_8UC3
		cvtColor(frame2, frame2g, CV_BGRA2GRAY);
		cvtColor(frame1, frame1g, CV_BGRA2GRAY);

		//printf("Matrix: %s \n", type2str(frame1g.type()).c_str());
		//printf("Matrix: %s \n", type2str(frame2g.type()).c_str());

		Mat disparity;

		Ptr<StereoBM> sbm = StereoBM::create(
			5 * 16,		//ndisparities, the disparity search range. For each pixel, the algorithm will find the best disparity from 0 (default minimum disparity) to ndisparities. 
						//	The search range can then be shifted by changing the minimum disparity.

			5		//SADWindowSize, the linear size of the blocks compared by the algorithm. The size should be odd (as the block is centered at the current pixel). 
					//	Larger block size implies smoother, though less accurate disparity map. Smaller block size gives more detailed disparity map, 
					//		but there is higher chance for algorithm to find a wrong correspondence.

			);

		sbm->compute(frame1g, frame2g, disparity);

		imshow("frame1", frame1);
		imshow("frame2", frame2);
		imshow("disparity", disparity);

		Mat Q, R1, R2, P1, P2;

		stereoRectify(
			solver1.getCameraMatrix(),
			distcoeffs,
			solver2.getCameraMatrix(),
			distcoeffs,
			Size(1280, 960),
			solver2.getRotationMatrix(),
			solver2.getTranslationVector(),
			R1,
			R1,
			P1,
			P2,
			Q
			);
		
		Mat _3dImage;
		reprojectImageTo3D(disparity, _3dImage, Q);

		cout << "size of _3dImage is " << _3dImage.size() << endl;
		cout << "First item is " << endl << _3dImage.row(0).col(0) << endl << endl;

		counter++;

		// frame1 will hold the previous frame, and in the next iteration we will read a new frame into frame2
		//    These two will thus be continuously cycled.
		frame1 = frame2.clone();

		// Calculate time
		now = cvGetTickCount();
		cout << "Iteration took " << (double)(now - then) / ticksPerSecond << " seconds" << endl;



		waitKey(1);
	}
	
	return 0;
}

/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat LinearLSTriangulation(Point3d u,       //homogenous image point (u,v,1)
						  Matx34d P,       //camera 1 matrix
						  Point3d u1,      //homogenous image point in 2nd camera
						  Matx34d P1       //camera 2 matrix
	)
{
	cout << "Creating A" << endl;
	//build matrix A for homogenous equation system Ax = 0
	//assume X = (x,y,z,1), for Linear-LS method
	//which turns it into a AX = B system, where A is 4x3, X is 3x1 and B is 4x1
	Matx43d A(
		u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
		u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
		u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
		u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
		);
	cout << "Creating B" << endl;
	
	Mat temp; temp = -(u.x*P(2, 3) - P(0, 3));
	Matx41d B;
		B = (temp,
		-(u.y*P(2, 3) - P(1, 3)),
		-(u1.x*P1(2, 3) - P1(0, 3)),
		-(u1.y*P1(2, 3) - P1(1, 3)));

	cout << "Creating X" << endl;
	Mat X;
	solve(A, B, X, DECOMP_SVD);

	return X;
}

/**
From "Triangulation", Hartley, R.I. and Sturm, P., Computer vision and image understanding, 1997
*/
Mat_<double> IterativeLinearLSTriangulation(
	Point3d u,		    //homogenous image point (u,v,1)
	Matx34d P,          //camera 1 matrix
	Point3d u1,         //homogenous image point in 2nd camera
	Matx34d P1          //camera 2 matrix
	) {
	double wi = 1, wi1 = 1;
	Mat_<double> X(4, 1);
	for (int i = 0; i<10; i++) { //Hartley suggests 10 iterations at most
		Mat_<double> X_ = LinearLSTriangulation(u, P, u1, P1);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;

		//recalculate weights
		double p2x = Mat_<double>(Mat_<double>(P).row(2)*X)(0);
		double p2x1 = Mat_<double>(Mat_<double>(P1).row(2)*X)(0);

		//breaking point
		//if (fabsf(wi - p2x) <= EPSILON && fabsf(wi1 - p2x1) <= EPSILON) break;

		wi = p2x;
		wi1 = p2x1;

		//reweight equations and solve
		Matx43d A((u.x*P(2, 0) - P(0, 0)) / wi, (u.x*P(2, 1) - P(0, 1)) / wi, (u.x*P(2, 2) - P(0, 2)) / wi,
			(u.y*P(2, 0) - P(1, 0)) / wi, (u.y*P(2, 1) - P(1, 1)) / wi, (u.y*P(2, 2) - P(1, 2)) / wi,
			(u1.x*P1(2, 0) - P1(0, 0)) / wi1, (u1.x*P1(2, 1) - P1(0, 1)) / wi1, (u1.x*P1(2, 2) - P1(0, 2)) / wi1,
			(u1.y*P1(2, 0) - P1(1, 0)) / wi1, (u1.y*P1(2, 1) - P1(1, 1)) / wi1, (u1.y*P1(2, 2) - P1(1, 2)) / wi1
			);
		Mat_<double> B = (Mat_<double>(4, 1) << -(u.x*P(2, 3) - P(0, 3)) / wi,
			-(u.y*P(2, 3) - P(1, 3)) / wi,
			-(u1.x*P1(2, 3) - P1(0, 3)) / wi1,
			-(u1.y*P1(2, 3) - P1(1, 3)) / wi1
			);

		solve(A, B, X_, DECOMP_SVD);
		X(0) = X_(0); X(1) = X_(1); X(2) = X_(2); X_(3) = 1.0;
	}
	return X;
}

static string type2str(int type)
{
	string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

vector<Point2f> setInputOne()
{
    vector<Point2f> input1;
    input1.push_back(Point2d(397.210571, 145.146866));
    input1.push_back(Point2d(650.494934, 129.172379));
    input1.push_back(Point2d(519.567688, 131.898239));
    input1.push_back(Point2d(531.834473, 267.480103));
    input1.push_back(Point2d(239.835358, 207.141220));
    input1.push_back(Point2d(834.740051, 174.580566));
    input1.push_back(Point2d(211.190155, 510.402740));
    input1.push_back(Point2d(437.319458, 218.244186));
    input1.push_back(Point2d(845.259948, 160.413910));
    input1.push_back(Point2d(559.729248, 170.678528));
    
    return input1;
}

vector<Point2f> setInputTwo()
{
    vector<Point2f> input2;
    input2.push_back(Point2d(490, 250));
    input2.push_back(Point2d(668, 242));
    input2.push_back(Point2d(578, 242));
    input2.push_back(Point2d(582, 335));
    input2.push_back(Point2d(380, 294));
    input2.push_back(Point2d(793, 278));
    input2.push_back(Point2d(367, 499));
    input2.push_back(Point2d(521, 306));
    input2.push_back(Point2d(806, 262));
    input2.push_back(Point2d(604, 272));
    
    return input2;
}