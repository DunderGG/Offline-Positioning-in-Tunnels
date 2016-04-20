/*
	AUTHOR: DAVID BENNEHAG and YANUAR T. A. NUGRAHA

	Requires OpenCV (tested on 3.1) AND PCL


*/
//OPENCV
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

//OUR STUFF
#include "Triangulation.h"
#include "FeatureDetection.h"
#include "Calibration.h"
#include "PnPSolver.h"
#include "PointProjection.h"
#include "Converter.h"
#include "VisualOdometry.h"
#include "PCLCloudSearch.h"

#include <iostream>
#include <string>

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	const float THRESHOLD = 0.01f;

	//Calibration moved to its own class.
    Calibration calib;
	PnPSolver solver1, solver2;
	VO vodometry;
    FeatureDetection fdetect;

    Mat distCoeffs = calib.getDistortionCoeffs();

#ifndef KITTI_DATASET
    char pathname[100] = "/Users/januaditya/Thesis/exjobb-data/volvo/out0/";
#else
    char pathname[100] = "/Users/januaditya/Thesis/exjobb-data/kitti-odometry/dataset/sequences/00/image_0/";
#endif

    // CHECKING THE IMAGE QUALITY PER DIFFERENT FEATURE DETECTION
    // fdetect.computeKeypointsAndDraw(pathname);

    // VISUAL ODOMETRY
    //vodometry.initParameter();
    //vodometry.setImagePath(pathname);
    //vodometry.visualodometry();

    // MANUAL CORRESPONDENCES PNP SOLVER
	cout << "Starting first solver............." << endl << endl << endl;

	vector<Point2d> imageOne;
	imageOne.push_back(Point2d(397.210571, 145.146866));	imageOne.push_back(Point2d(650.494934, 129.172379));
	imageOne.push_back(Point2d(519.567688, 131.898239));	imageOne.push_back(Point2d(531.834473, 267.480103));
	imageOne.push_back(Point2d(239.835358, 207.141220));	imageOne.push_back(Point2d(834.740051, 174.580566));
	imageOne.push_back(Point2d(211.190155, 510.402740));	imageOne.push_back(Point2d(437.319458, 218.244186));
	imageOne.push_back(Point2d(845.259948, 160.413910));	imageOne.push_back(Point2d(559.729248, 170.678528));

	solver1.setImagePoints(imageOne);
	solver1.foo(1);

	cout << endl << endl << endl << endl << endl << endl;
	cout << "Starting second solver............." << endl << endl << endl;

	// Take another image showing the same scene.
	//  So the 2D image points will be different of course...

	vector<Point2d> imageTwo;
	imageTwo.push_back(Point2d(490, 250));	imageTwo.push_back(Point2d(668, 242));
	imageTwo.push_back(Point2d(578, 242));	imageTwo.push_back(Point2d(582, 335));
	imageTwo.push_back(Point2d(380, 294));	imageTwo.push_back(Point2d(793, 278));
	imageTwo.push_back(Point2d(367, 499));	imageTwo.push_back(Point2d(521, 306));
	imageTwo.push_back(Point2d(806, 262));	imageTwo.push_back(Point2d(604, 272));

	solver2.setImagePoints(imageTwo);
	solver2.foo(1);

	cout << endl << endl << "camera 1 position: " << endl << solver1.getCameraPosition() << endl << "camera 2 position: " << endl << solver2.getCameraPosition() << endl;

    //
    //// IDK WHAT IS THIS
	//PCLTest::foo();
	//FeatureDetection::foo();
	//
	//int counter = 0;
	//
	//string filename = "CADS4_OWG740_20140422_162231_005.avi";
	//VideoCapture vc(filename);
	//if (!vc.isOpened())
	//	exit(EXIT_FAILURE);
	//
	//Mat frame1, frame2;

	cout << endl << "Loading point cloud... ";
	// Load the point cloud only once, as it is very slow
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("cloud.pcd", *cloud);
	cout << "Done!" << endl << endl;

	/*
		Projection algorithm from: http://stackoverflow.com/questions/13957150/opencv-computing-camera-position-rotation

		Projection details: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html

		v' is camera space coordinate, v is world space coordinate
		v = R^T * v' - R^T * t

		x = K * [R|t] * X

	*/
	Mat T = solver1.getCameraPose().clone();
	Mat K = calib.getCameraMatrix();

	Mat p, p_, p3d;
	double newX, newY, newZ;

	vector<double> bestPoint{ 0, 0, 0, 1000 };

	vector<Point2d> imagepoints = solver1.getVoVImagePoints()[0];

	for(Point2d point : imagepoints)
	{
		for (double i = 20; i < 30; i+=.1)
		{
			/*
			*	Take the image coordinates (X and Y) of the feature point,
			*		together with i which represents going "one step further" on the ray.
			*/
			p = (Mat_<double>(3, 1) << i*point.x, i*point.y, i);


			/*
			*	We use the inverse camera matrix (K) to bring the image coordinate from the
			*		Image coordinate system
			*	to
			*		Camera coordinate system
			*
			*
			*/
			p = K.inv() * p;


			/*
			*	To represent a 3D point in the world coordinate system as a homogeneous point,
			*		we need a 4x1 vector, containing the X, Y, Z and a 1.
			*/
			p3d = (Mat_<double>(4, 1) << p.at<double>(0,0), p.at<double>(1, 0), p.at<double>(2, 0), 1);


			/*
			*	We use our 4x4 transformation matrix (T), which is equal to our camera pose matrix,
			*		to bring the point from the
			*			Camera coordinate system
			*		to
			*			World coordinate system
			*
			*	If we have a point in camera coordinates, we can transform it to world coordinates.
			*		p' = T * (x, y, z, 1)^T
			*/
			p_ = T * p3d;


			/*
			*	We use the calculated point inside the world coordinate system as the search point
			*		for finding the closest neighbour (point) in the point cloud.
			*/
			newX = p_.at<double>(0, 0);	newY = p_.at<double>(1, 0); newZ = p_.at<double>(2, 0);
			vector<double> newPoint = PCLCloudSearch::FindClosestPoint(newX, newY, newZ, cloud);

			/*
			*	We want to save the neighbour only if it's better than our previously best.
			*
			*/
			if (newPoint[3] < THRESHOLD)
			{
				bestPoint = newPoint;

				break;
			}
		}

		cout << setprecision(15);
		cout << "*****************************" << endl;
		cout << "Seaching for image point\t" << point << endl << endl;
		cout << "The best point found:" << endl
			<< "X = \t"		<< bestPoint[0] << endl
			<< "Y = \t"		<< bestPoint[1] << endl
			<< "Z = \t"		<< bestPoint[2] << endl
			<< "DIST = \t"	<< bestPoint[3] << endl;
		cout << "*****************************\n\n\n\n\n";
	}
    
	/*
	// Skeleton code for iterating through the image sequence
	while (vc.read(frame2))
	{

		// Change brightness
		//frame2 = frame2 + Scalar(10,10,10);


		//resize(frame2, frame2, Size(frame2.cols / 2, frame2.rows / 2));

		//In the first iteration, only frame2 will contain a frame, so skip this
		if (counter == 0)
		{
			counter++;
			frame1 = frame2.clone();
			continue;
		}

		counter++;

		// frame1 will hold the previous frame, and in the next iteration we will read a new frame into frame2
		//    These two will thus be continuously cycled.
		frame1 = frame2.clone();

		if (waitKey(5000) == 'k')
			break;
	}
	*/



	return 0;
}
