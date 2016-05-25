#include "Reprojection.h"
#include "Calibration.h"
#include "PCLCloudSearch.h"

//PCL
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <cmath>

using namespace std;
using namespace cv;

Reprojection::Reprojection(double thresh, double mind, double maxd, double delta)
{
	this->threshold = 0.01;
	this->mindist   = 5;
	this->maxdist  	= 150;
	this->deltaz   	= 0.01;
}

Mat Reprojection::foo(Mat frame1, Mat frame2, Mat rMat1, Mat rMat2, cv::Mat tVec1, cv::Mat tVec2)
{
	Calibration calib;
	Mat distCoeffs = calib.getDistortionCoeffs();

	Mat frame1g, frame2g;
	// For StereoBM, images have to be CV_8UC1, original was CV_8UC3
	cvtColor(frame2, frame2g, CV_BGRA2GRAY);
	cvtColor(frame1, frame1g, CV_BGRA2GRAY);

	Mat disparity;

	Ptr<StereoBM> sbm = StereoBM::create(
		5 * 16,		// ndisparities, the disparity search range. For each pixel, the algorithm will find the best disparity from 0 (default minimum disparity) to ndisparities.
					//	The search range can then be shifted by changing the minimum disparity.

		5			// SADWindowSize, the linear size of the blocks compared by the algorithm. The size should be odd (as the block is centered at the current pixel).
					//	Larger block size implies smoother, though less accurate disparity map. Smaller block size gives more detailed disparity map,
					//		but there is higher chance for algorithm to find a wrong correspondence.

		);

	sbm->compute(frame1g, frame2g, disparity);

	//imshow("frame1", frame1);
	//imshow("frame2", frame2);
	//imshow("Disparity Map", disparity);

	Mat Q = Mat(4, 4, CV_64F),
	R1 = Mat(3, 3, CV_64F),
	R2 = Mat(3, 3, CV_64F),
	P1 = Mat(3, 4, CV_64F),
	P2 = Mat(3, 4, CV_64F);
	/*
	The rotation and translation we give stereoRectify() has to be the rotation and translation between the two cameras,
	not between the two cameras and the world separately.

	From: http://stackoverflow.com/questions/5055864/finding-rotation-matrices-between-two-cameras-for-stereorectify

	For rotation between camera 1 and camera 2, use inverse extrinsic matrix of camera 1, multiplied by the extrinsic matrix of camera 2
	*/


	Mat rotation = rMat1.inv() * rMat2;
	cout << "Rotation = " << endl << rotation << endl;

	Mat translation = tVec1 - tVec2;
	cout << "Translation = " << endl << translation << endl << endl;

	stereoRectify(
		calib.getCameraMatrix(),		// First camera matrix.
		distCoeffs,						//              distcoeffs.
		calib.getCameraMatrix(),		// Second camera matrix.
		distCoeffs,						//              distcoeffs.
		Size(1280, 960),				// Size of the image used for stereo calibration.
		rotation,						// Rotation matrix between the coordinate systems of the two cameras.
		translation,					// Translation vector between the coordinate systems of the two cameras.
		R1,								// Output
		R2,								// Output
		P1,								// Output
		P2,								// Output
		Q								// Output 4x4 disparity-to-depth mapping matrix, to be used in reprojectImageTo3D().
										// Optional flags, should this be set?
		);

	Mat _3dImage;
	reprojectImageTo3D(
		disparity,		// Input disparity image.
		_3dImage,		// Output image of the same size as disp. Each element of _3dImage(x,y) contains 3D coordinates of the point (x,y) computed from the disparity map.
		Q,				// 4x4 perspective transformation matrix that can be obtained by stereoRectify().
		true			// handleMissingValues indicates whether the function should handle missing values (i.e. points where the disparity was not computed).
		);

	//cout << "size of _3dImage is " << _3dImage.size() << endl;
	cout << "An item from _3dImage = " << endl << _3dImage.row(490).col(250) << endl << endl;

	return _3dImage;
}


/*
*	Projection algorithm from: http://stackoverflow.com/questions/13957150/opencv-computing-camera-position-rotation
*
*	Projection details: http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/EPSRC_SSAZ/node3.html
*
*	Algebraic solution: https://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter1.htm
*
*	v' is camera space coordinate, v is world space coordinate
*	v = R^T * v' - R^T * t
*
*	x = K * [R|t] * X
*
*/
vector<double> Reprojection::backproject(Mat T, Mat	K, Point2d imagepoint, pcl::PointCloud<pcl::PointXYZ>::Ptr &cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree)
{
    double THRESHOLD 	= 0.05;
    double MIN_DIST 	= 15;
    double MAX_DIST 	= 60;
    double DELTA_Z 		= 0.05;

    vector<double> bestPoint{ 0, 0, 0, 1000 };
    Mat p, p_, p3d;
    double newX, newY, newZ;

    // setup the camera origin in world coordinate
    Mat origin_p = (Mat_<double>(3,1) << 0, 0, 0);
    Mat origin_c = K.inv() * origin_p;
    Mat origin_w = T * (Mat_<double>(4, 1) << origin_c.at<double>(0, 0), origin_c.at<double>(1, 0), origin_c.at<double>(2, 0), 1);

    for (double i = MIN_DIST; i < MAX_DIST; i += DELTA_Z)
    {
        /*
         *	Take the image coordinates (X and Y) of the feature point,
         *		together with i which represents going "one step further" on the ray.
         */
        p = (Mat_<double>(3, 1) << i*imagepoint.x, i*imagepoint.y, i);


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
        p3d = (Mat_<double>(4, 1) << p.at<double>(0, 0), p.at<double>(1, 0), p.at<double>(2, 0), 1);


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
        vector<double> newPoint = PCLCloudSearch::FindClosestPoint(newX, newY, newZ, cloud, kdtree);

        /*
         *	As soon as we find a "good enough" point, return it,
         *		since we don't want to risk going too deep into the cloud.
         */
        if (newPoint[3] < THRESHOLD)
        {
            // return the lerp
            bestPoint = LinearInterpolation (newPoint, origin_w, p_);

            break;
        }
    }

    return bestPoint;
}

// using radius instead
vector<double> Reprojection::backprojectRadius(Mat T, Mat K, Point2d imagepoint, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::KdTreeFLANN<pcl::PointXYZ>& kdtree)
{
	double MIN_DIST = 10.0f;			// in pixel
	double MAX_DIST = 100.0f;		// in pixel
    double min_dist_L2;
    double max_dist_L2;
	double RADIUS = 0.05f;			// 5 centimeter
	double THRESHOLD = 0.01f;		// 1 centimeter
	double DELTA_Z;					// meter

	vector<double> bestPoint;
	Mat p, p_, p3d;
	double newX, newY, newZ;

	// setup the camera origin in world coordinate
	Mat origin_p = (Mat_<double>(3,1) << 0, 0, 0);
	Mat origin_c = K.inv() * origin_p;
	Mat origin_w = T * (Mat_<double>(4, 1) << origin_c.at<double>(0,0), origin_c.at<double>(1,0), origin_c.at<double>(2,0), 1);

	// determine the DELTA_Z or distance between each search
	// for each search point X with search radius r, if we got no best point, then we have to move
	// the search point X+l distance. this l distance should be < 2r in order to handle possibility
	// of the best point that is located right outside the intersection area of two search areas.
	// more here:
	//		http://mathworld.wolfram.com/Circle-CircleIntersection.html

	// update the DELTA_Z (in meter)
	DELTA_Z = 2 * sqrt (pow(RADIUS, 2) - pow(THRESHOLD, 2));

	// find where this MIN_DIST and MAX_DIST in meter according to the imagepoint coordinate which still in pixels
	Mat mindist_p = (Mat_<double>(3,1) << imagepoint.x, imagepoint.y, MIN_DIST);
	Mat mindist_c = K.inv() * mindist_p;
	Mat mindist_w = T * (Mat_<double>(4,1) << mindist_c.at<double>(0,0), mindist_c.at<double>(1,0), mindist_c.at<double>(2,0), 1);

	// find where this MIN_DIST and MAX_DIST in meter according to the imagepoint coordinate which still in pixels
	Mat maxdist_p = (Mat_<double>(3,1) << imagepoint.x, imagepoint.y, MAX_DIST);
	Mat maxdist_c = K.inv() * maxdist_p;
	Mat maxdist_w = T * (Mat_<double>(4,1) << maxdist_c.at<double>(0,0), maxdist_c.at<double>(1,0), maxdist_c.at<double>(2,0), 1);

    // create a vector, start from the origin_w to mindist_w
    Mat ray_w     = mindist_w - origin_w;

    min_dist_L2 = norm(origin_w, mindist_w, NORM_L2);
    max_dist_L2 = norm(origin_w, maxdist_w, NORM_L2);

    for (double i = min_dist_L2; i < max_dist_L2; i += DELTA_Z)
    {
        // calculation of the search point should be like this
        Mat sPoint = mindist_w + i * ray_w;

		//cout << i << endl;
		//cout << sPoint << endl;

        newX = sPoint.at<double>(0,0); newY = sPoint.at<double>(1,0), newZ = sPoint.at<double>(2,0);
        bestPoint = PCLCloudSearch::FindClosestPointRadius(	newX, newY, newZ, RADIUS, THRESHOLD,
                                                            std::ref(cloud), std::ref(kdtree),
                                                            origin_w);

		// return current bestPoint to the upper stack if the lerp distance is less than 1000.0f (default)
        if (bestPoint[3] < 1000)
            break;
    }

	return bestPoint;
}

vector<double> Reprojection::LinearInterpolation(vector<double> bestPoint, Mat origin, Mat vectorPoint)
{
	// basically if known two points in 3D A and B, and a point P (bestPoint) that does not belong to vector AB
	// a perpendicular vector xP has 90 degree angle from AB and has length of bestPoint[3].
	//
	// x is the point in which we need to return. Ax is basically the orthogonal projection of AP
	// to vector AB.
	//
	// it given as
	// 		x = A + dot(AP,AB) / dot(AB,AB) * AB
	//				with
	//				 A   = origin
	//				 P   = bestPoint
	//				 B   = 3D point in line
	//				 AP  = vector from origin to the point P (bestPoint)
	// 				 AB  = vector from origin to the AB
	//				 dot = dot product
	// 				 *   = scalar multiplication

    Mat vectorAP = (Mat_<double>(4, 1) << bestPoint[0]-origin.at<double>(0), bestPoint[1]-origin.at<double>(1), bestPoint[2]-origin.at<double>(2), 0);
    Mat vectorAB = vectorPoint - origin;

    Mat output = origin + (vectorAP.dot(vectorAB) / vectorAB.dot(vectorAB)) * vectorAB;

    // if (false)
    // {
    //     cout << endl;
    //     cout << vectorAP << endl;
    //     cout << vectorAB << endl;
    //     cout << output << endl;
    //     cout << "[" << bestPoint[0] << ";\n" << bestPoint[1] << ";\n" << bestPoint[2] << "]" << endl;
    // }

    return {output.at<double>(0), output.at<double>(1), output.at<double>(2)};
}
