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
#include <pcl/kdtree/kdtree_flann.h>

//OUR STUFF
#include "FeatureDetection.h"
#include "PnPSolver.h"
#include "Calibration.h"
<<<<<<< HEAD
//#include "VisualOdometry.h"a
=======
//#include "VisualOdometry.h"
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
#include "Reprojection.h"
#include "Common.h"

//C++ STUFF
#include <iostream>
#include <string>
#include <thread>
#include <mutex>
#include <chrono>
#include <fstream>

using namespace std;
using namespace cv;

<<<<<<< HEAD
const int NR_OF_FRAMES = 15;
const int FIRST_INDEX = 433, LAST_INDEX = FIRST_INDEX + NR_OF_FRAMES;
const int NUMBEROFTHREADS = 8;

const bool PAR_MODE = true;
static const bool DRAWKPTS = true;

std::mutex global_mutex;

=======
const int NR_OF_FRAMES = 40;
const int FIRST_INDEX = 432, LAST_INDEX = FIRST_INDEX + NR_OF_FRAMES;
const int NUMBEROFTHREADS = 8;
const bool PAR_MODE = true;

std::mutex global_mutex;

vector<Point2d>             tunnel2D;
vector<Point3d>             tunnel3D;
Mat                         tunnelDescriptor;
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de



<<<<<<< HEAD
/*
*	Update the Look Up Table for what descriptor belongs to which image point
*
*	_3dToDescriptorMap.first[0]  == 3D coordinates vector
*
*	_3dToDescriptorVector[i].first.x == X
*	_3dToDescriptorVector[i].first.y == Y
*	_3dToDescriptorVector[i].first.z == Z
*
*	_3dToDescriptorVector[i].second == its descriptor

		// Output: The Look Up Table containing all our descriptors and 3D-points
		// Output: vector containing the keypoints for which we found 3D-points
		// Output: vector containing the corresponding 3D-points
		// Output: vector saving the indexes for which we got results
		// Input:  the transformation matrix (camera pose)
		// Input:  the camera matrix
		// Input:  All the keypoints that SIFT found, these are the pixels we want to backproject
		// Input:  The descriptors for those keypoints that should go into our LUT
		// Input:  The point cloud used for the backprojection
		// Input:  The kdtree that PCL uses for finding the closest neighbour
		// Input:  Each thread gets a number of backprojections to do,
		// Input:		from index "start" to "end"

*/

=======
void calcBestPoint(
	vector< pair<Point3d, Mat> >& _3dToDescriptorVector,
	Mat T,
	Mat K,
	vector<KeyPoint> keypoints1,
	Mat descriptors1,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree,
	int start,
	int end,
	int threadIndex);


void prepareMap(char* mapCoordinateFile, char* mapKeypointsFile);
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
//vector< pair<Point3d, Mat> > manualStuff(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> kdtree);
string type2str(int type);

int main(int argc, char** argv)
{
	Calibration calib;
	PnPSolver solver1(1000, 5, 0.99);
	FeatureDetection fdetect;
	Common com;

	ofstream logFile;
	logFile.open("logs/cameraPositions.txt", std::ios::app);
	logFile << endl << endl << "Camera Positions:" << endl << endl << "[ " << flush;

	Mat frame1;
	Mat descriptors1;
	vector<KeyPoint> keypoints1;
	vector<double> reprojectionErrors;

	auto beginningOfMain = std::chrono::high_resolution_clock::now();

	auto beginningOfMain = std::chrono::high_resolution_clock::now();

	auto begin = std::chrono::high_resolution_clock::now();
	// We load the point cloud once and then keep it open for the rest of the execution,
	//    since the loading takes alot of time.
	cout << endl << "Loading point cloud... ";
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile("gnistangtunneln-semifull-voxelized.pcd", *cloud);
<<<<<<< HEAD

	cout << " And the kdtree... ";
=======
    

>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
	//Build the kdtree for searching in the point cloud.
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);

	auto end = std::chrono::high_resolution_clock::now();
	cout << "Done! (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms)" << endl << endl;

	//begin = std::chrono::high_resolution_clock::now();
	// Before running our loop through the image sequence, do the manual stuff of the first image.
	//cout << "Performing manual labour... ";
	//vector< pair<Point3d, Mat> > _3dToDescriptorVector = manualStuff(cloud);
	vector< pair<Point3d, Mat> > _3dToDescriptorVector;
	//end = std::chrono::high_resolution_clock::now();
	//cout << "Done! (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms)" << endl << endl;

<<<<<<< HEAD


	// 2. prepare the manual correspondences as a lookup table
	char map2Dto3D[100];
	char mapDescrip[100];
	char nextimage[100];
	vector<Point2d> tunnel2D;
	vector<Point3d> tunnel3D;
	Mat tunnelDescriptor;
	sprintf(map2Dto3D, "ManualCorrespondences.txt");
	sprintf(mapDescrip, "ManualCorrespondences.yml");

	com.prepareMap(map2Dto3D, mapDescrip, ref(tunnel2D), ref(tunnel3D), ref(tunnelDescriptor));
	com.updatelut(tunnel3D, tunnelDescriptor, ref(_3dToDescriptorVector));

	int clearingCounter = 0;
	int frameCounter = 0;
=======
	Calibration calib;
	PnPSolver solver1;
	FeatureDetection fdetect;

	// 2. prepare the manual correspondences as a lookup table
	char map2Dto3D[100];
	char mapDescrip[100];
	char nextimage[100];
	sprintf(map2Dto3D, "ManualCorrespondences.txt");
	sprintf(mapDescrip, "ManualCorrespondences.yml");

	prepareMap(map2Dto3D, mapDescrip);
	for (int h = 0; h < tunnel3D.size(); h++)
	{
		_3dToDescriptorVector.push_back(make_pair(tunnel3D[h], tunnelDescriptor.row(h)));
	}

	int clearingCounter = 0;
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
	// Don't start the loop on the image we handled manually.
	for (int i = FIRST_INDEX; i < LAST_INDEX; i++)
	{
<<<<<<< HEAD
		begin = std::chrono::high_resolution_clock::now();
		sprintf(nextimage, "/Users/januaditya/Thesis/exjobb-data/volvo/out0/img_%05d.png", i);
		cout << "Loading image: " << nextimage << "... ";

		Mat frame1 = imread(nextimage);
		Mat desc = com.getdescriptor(_3dToDescriptorVector);

		vector<Point2d> retrieved2D;
		vector<Point3d> retrieved3D;

		begin = std::chrono::high_resolution_clock::now();

		cout << "Running SIFT... ";
		cout << "Performing matching... ";
		fdetect.computeFeaturesAndMatching (frame1, tunnel2D, tunnel3D, desc, frameCounter,
											&keypoints1, &descriptors1, &retrieved2D, &retrieved3D);
		cout << "Done!" << endl;
		solver1.setImagePoints(retrieved2D);
		solver1.setWorldPoints(retrieved3D);
		solver1.run(1);
=======
		//tunnel2D.clear(); tunnel3D.clear();
		
		begin = std::chrono::high_resolution_clock::now();
		sprintf(nextimage, "/Users/januaditya/Thesis/exjobb-data/volvo/out0/img_%05d.png", i);
		cout << "Loading image: " << nextimage << "... ";
		Mat frame1 = imread(nextimage);
		//String filename = "imageSequence\\img_00" + to_string(i) + ".png";
		//cout << "Loading image: " << filename << "... ";
		//frame1 = imread(filename);

		// set the RoI (Region of Interest)
		// this mask is to take only 50% upper part of the image
		Mat img_maskUpperPart = Mat::zeros(frame1.size(), CV_8U);
		Mat img_roiUpperPart(img_maskUpperPart, Rect(0, 0, frame1.cols, frame1.rows * 4 / 5));
		img_roiUpperPart = Scalar(255, 255, 255);
		

		cout << "Done!" << endl;






		begin = std::chrono::high_resolution_clock::now();
		cout << "Running SIFT... ";
		fdetect.siftDetector(frame1, keypoints1, img_maskUpperPart);
		fdetect.siftExtraction(frame1, keypoints1, descriptors1);
		end = std::chrono::high_resolution_clock::now();
		cout << "Done!\tFound " << descriptors1.rows << " descriptors (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "/" << std::chrono::duration_cast<std::chrono::milliseconds>(end - beginningOfMain).count() << " ms)" << endl;

		

		// Take the descriptor mat out of our lookup table for the matching
		Mat tunnelDescriptors;
		for (pair<Point3d, Mat> pair : _3dToDescriptorVector)
		{
			tunnelDescriptors.push_back(pair.second);
			//cout << "pushing back:" << endl << pair.second << endl << endl;
		}
		
		begin = std::chrono::high_resolution_clock::now();
		cout << "Performing matching... ";
		vector<vector<DMatch> > matches;
		fdetect.bfMatcher(descriptors1, tunnelDescriptors, matches);

		// 8. retrieve the matches indices from the descriptor
		vector<int> matchedIndices;
		vector<int> matchedXYZ;
		float dist1 = 0.0f, dist2 = 0.0f;

		for (int j = 0; j < matches.size(); j++)
		{
			DMatch first = matches[j][0];
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de

		end = std::chrono::high_resolution_clock::now();
<<<<<<< HEAD
		cout << "Done!\tFound " << descriptors1.rows << " descriptors (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "/" << std::chrono::duration_cast<std::chrono::milliseconds>(end - beginningOfMain).count() << " ms)" << endl;
		// cout << "Done!\tMatched " << matches.size() << "/" << descriptors1.rows << " descriptors (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "/" << std::chrono::duration_cast<std::chrono::milliseconds>(end - beginningOfMain).count() << " ms)" << endl;
		cout << "Done!" << endl << endl;
		cout << "Camera Position:" << endl << solver1.getCameraPosition() << endl;
=======
		cout << "Done!\tMatched " << matches.size() << "/" << descriptors1.rows << " descriptors (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "/" << std::chrono::duration_cast<std::chrono::milliseconds>(end - beginningOfMain).count() << " ms)" << endl;

>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de

		Mat T = solver1.getCameraPose().clone();
		Mat K = calib.getCameraMatrix();

		Mat R = solver1.getRotationMatrix();
		Mat t = solver1.getTranslationVector();


<<<<<<< HEAD
=======
		
		vector< pair<Point3d, Mat> > tempLUT;
		vector< pair<Point3d, Mat> >::iterator halfwayItr = _3dToDescriptorVector.begin() + _3dToDescriptorVector.size() / 2;
		vector< pair<Point3d, Mat> >::iterator beginItr = _3dToDescriptorVector.begin();
		clearingCounter++;
		if(clearingCounter == 5)
		{
			cout << "Clearing the LUT..." << endl;
			clearingCounter = 0;

			tempLUT.insert(tempLUT.end(),
							make_move_iterator(beginItr),
							make_move_iterator(halfwayItr));
			
			_3dToDescriptorVector.clear();

			_3dToDescriptorVector = move(tempLUT);

			tempLUT.clear();

			cout << "New size of LUT: " << _3dToDescriptorVector.size() << endl;
		}
		
		retrieved2D.clear();
		retrieved3D.clear();
		
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de

		/*
		*	To prevent the LUT from growing too big and possibly match against too old entries,
		*		we periodically clear the first half.
		*/
		vector< pair<Point3d, Mat> > tempLUT;
		vector< pair<Point3d, Mat> >::iterator halfwayItr = _3dToDescriptorVector.begin() + _3dToDescriptorVector.size() / 2;
		vector< pair<Point3d, Mat> >::iterator endItr = _3dToDescriptorVector.end();
		clearingCounter++;
		if (clearingCounter == 5)
		{
<<<<<<< HEAD
			cout << "Clearing the LUT..." << endl;
			clearingCounter = 0;

			tempLUT.insert(tempLUT.end(),
				make_move_iterator(halfwayItr),
				make_move_iterator(endItr));

			_3dToDescriptorVector.clear();

			_3dToDescriptorVector = move(tempLUT);
=======
			retrieved2D.push_back(Point2d(keypoints1[matchedIndices[k]].pt.x, keypoints1[matchedIndices[k]].pt.y));

			retrieved3D.push_back(Point3d(_3dToDescriptorVector[matchedXYZ[k]].first.x, _3dToDescriptorVector[matchedXYZ[k]].first.y, _3dToDescriptorVector[matchedXYZ[k]].first.z));
			//retrieved3D.push_back(Point3d(tunnel3D[matchedXYZ[k]].x, tunnel3D[matchedXYZ[k]].y, tunnel3D[matchedXYZ[k]].z));

			//cout << std::fixed << setprecision(4);
			//cout << "   pushed {" << keypoints1[matchedIndices[i]].pt.x << ", " << keypoints1[matchedIndices[i]].pt.y << "} --> {"
			//	<< tunnel3D[matchedXYZ[i]].x << ", " << tunnel3D[matchedXYZ[i]].y << ", " << tunnel3D[matchedXYZ[i]].z << "}" << endl;
		}

		solver1.setImagePoints(retrieved2D);
		solver1.setWorldPoints(retrieved3D);
		solver1.run(0);
		cout << "Done!" << endl << endl;
		cout << "Camera Position:" << endl << solver1.getCameraPosition() << endl;
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de

			tempLUT.clear();

			cout << "New size of LUT: " << _3dToDescriptorVector.size() << endl;
		}


<<<<<<< HEAD
		// Keep track of all our threads
=======

>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
		begin = std::chrono::high_resolution_clock::now();
		cout << "Performing backprojection of " << descriptors1.rows << " descriptors... ";
		/*
<<<<<<< HEAD
		 *	After finding our projection matrix, we can backproject the descriptors we found
		 *		onto the point cloud.
		 */
		vector<Point2d> projectedKeypoints;
		vector<Point3d> projectedWorldpoints;
		vector<int>		projectedIndex;
		if(PAR_MODE)		//RUN THE BACKPROJECTION WITH THREADS IN PARALLELL
		{
			cout << "Running multithreaded..." << endl;

			com.threading(NUMBEROFTHREADS, T, K, keypoints1, descriptors1, cloud, kdtree,
							ref(_3dToDescriptorVector),
							ref(projectedWorldpoints),
							ref(projectedKeypoints),
							ref(projectedIndex));
		}
=======
		*	For every feature point that we find in our image, we do the backprojection.
		*/
		if(PAR_MODE)		//RUN THE BACKPROJECTION WITH THREADS IN PARALLELL
		{
			cout << "Running multithreaded..." << endl;

			int keypointsPerThread;

			for (int threadIndex = 0; threadIndex < NUMBEROFTHREADS; threadIndex++)
			{
				workerCount++;

				int start = threadIndex * keypointsPerThread;
				int end = (threadIndex + 1) * keypointsPerThread;

				// We create one worker for each keypoint.
				// The order in which they push their results into the look up table does not matter.
				workers.push_back(
						thread( &calcBestPoint,
								ref(_3dToDescriptorVector),
								T,
								K,
								keypoints1,
								descriptors1,
								cloud,
								kdtree,
								start,
								end,
								threadIndex));
			}
			// Join acts as a "wall", so that all threads finish before the main thread continues.
			for (int l = 0; l < workers.size(); l++)
			{
				//cout << "Joining thread #" << workers[l].get_id() << endl;
				if (workers[l].joinable())
					workers[l].join();
			}
			workers.clear();

		}


>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
		else		// RUN IN SEQUENTIAL MODE INSTEAD
		{
			cout << "Running singlethreaded..." << endl;
			for (int counter = 0; counter < keypoints1.size(); counter = counter + 20)
			{
				vector<double> bestPoint = Reprojection::backproject(
					T,
					K,
					Point2d(keypoints1[counter].pt.x, keypoints1[counter].pt.y),
					cloud,
					kdtree);

				if (bestPoint[0] == 0 || bestPoint[1] == 0 || bestPoint[2] == 0)
<<<<<<< HEAD
				{
=======
				{	
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
					cout << "Threw away empty projection result!" << endl;
					continue;
				}

				/*
				*	Update the Look Up Table for what descriptor belongs to which image point
				*
				*	_3dToDescriptorMap.first[0]  == 3D coordinates vector
				*
				*	_3dToDescriptorVector[i].first.x == X
				*	_3dToDescriptorVector[i].first.y == Y
				*	_3dToDescriptorVector[i].first.z == Z
				*
				*	_3dToDescriptorVector[i].second == its descriptor
				*/
				// Define the 3D coordinate
				Point3d _3dcoord; _3dcoord.x = bestPoint[0]; _3dcoord.y = bestPoint[1]; _3dcoord.z = bestPoint[2];

				// Define its descriptor, should have size 1x128
				Mat desc;
				if (counter > descriptors1.rows)
					continue;
				desc = descriptors1.row(counter);

				// Push the pair into the lookup table
				_3dToDescriptorVector.push_back(make_pair(_3dcoord, desc));
				//tunnel3D.push_back(_3dcoord);
				//tunnelDescriptor.push_back(descriptors1.row(counter));
			}
		}
		//END OF BACKPROJECTION
<<<<<<< HEAD

		// refine the camera pose using more 3D points and stricter RANSAC
		PnPSolver solver2(1000, 2, 0.99);
		solver2.setImagePoints(projectedKeypoints);
		solver2.setWorldPoints(projectedWorldpoints);
		solver2.run(1);

		end = std::chrono::high_resolution_clock::now();
		cout << "Done! (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "/" << std::chrono::duration_cast<std::chrono::milliseconds>(end - beginningOfMain).count() << " ms)" << endl << endl;

		cout << "Size of our LUT: " << endl << _3dToDescriptorVector.size() << endl;
		cout << "Camera Position:" << endl << solver2.getCameraPosition() << endl;

		// 15. check reprojection error of each backprojected world points
		vector<Point2d> reprojectedPixels;
		projectPoints(projectedWorldpoints,
			solver1.getRotationMatrix(),
			solver1.getTranslationVector(),
			calib.getCameraMatrix(),
			calib.getDistortionCoeffs(),
			reprojectedPixels);

		double repError = 0;
		for (int itx = 0; itx < projectedIndex.size(); itx++)
		{
			double dx, dy;

			dx = pow(abs(reprojectedPixels[itx].x - keypoints1[projectedIndex[itx]].pt.x), 2);
			dy = pow(abs(reprojectedPixels[itx].y - keypoints1[projectedIndex[itx]].pt.y), 2);

			repError += sqrt(dx + dy);
			reprojectionErrors.push_back(repError);
		}
		cout << setprecision(15) << "Reprojection error: " << repError / projectedWorldpoints.size() << " pixels"
			 << "\t(" << repError << "/" << projectedWorldpoints.size() << ")" << endl;

		//double sum = 0;
		//for (double err : reprojectionErrors)
		//	sum += err;
		//cout << "Average so far is: " << sum / reprojectionErrors.size() << endl;
=======
		
		end = std::chrono::high_resolution_clock::now();
		cout << "Done! (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "/" << std::chrono::duration_cast<std::chrono::milliseconds>(end - beginningOfMain).count() << " ms)" << endl << endl;

		cout << "Size of our LUT: " << endl << _3dToDescriptorVector.size() << endl;
		



		/*
		// Calculate the miss rate
		Point3d cmp; cmp.x = 0; cmp.y = 0; cmp.z = 0;
		int misses = 0;
		
		cout << "All found points: " << endl;
		
		for (pair<Point3d, Mat> item : _3dToDescriptorVector)
		{
			cout << "[" << item.first.x << ", " << item.first.y << ", " << item.first.z << "]" << endl;
			if (item.first == cmp)
				misses++;
		}
		cout << "Number of misses: " << misses << " (" << ((double)misses / _3dToDescriptorVector.size()) * 100 << "%)" << endl;
		*/
		cout << "Camera Position:" << endl << solver1.getCameraPosition() << endl;
		

		vector<Point3d> _3dTemp;
		for (int m = (_3dToDescriptorVector.size() - workerCount); m <= _3dToDescriptorVector.size(); m++)
		{
			_3dTemp.push_back(_3dToDescriptorVector[m].first); 
		}

		// 15. check reprojection error of each backprojected world points
		/*
		vector<Point2d> reprojectedPixels;
		projectPoints(_3dTemp,
			solver1.getRotationMatrix(),
			solver1.getTranslationVector(),
			calib.getCameraMatrix(),
			calib.getDistortionCoeffs(),
			reprojectedPixels);

		double repError = 0;
		for (int itx = 0; itx < _1dTemp.size(); itx++)
		{
			double dx, dy;

			dx = pow(abs(reprojectedPixels[itx].x - detectedkpts[_1dTemp[itx]].pt.x), 2);
			dy = pow(abs(reprojectedPixels[itx].y - detectedkpts[_1dTemp[itx]].pt.y), 2);

			repError += sqrt(dx + dy);
		}
		*/


		cout << "****************** STARTING OVER ******************" << endl;
	}

	auto endOfMain = std::chrono::high_resolution_clock::now();
	cout << "Done! (" << std::chrono::duration_cast<std::chrono::milliseconds>(endOfMain - beginningOfMain).count() << " ms)" << endl << endl;

	cout << "Camera Positions:" << endl;
	cout << "[ ";
	for(Mat pos : solver1.camPositions)
		cout << setprecision(10) << pos.at<double>(0,0) << " " << pos.at<double>(1, 0) << " " << pos.at<double>(2, 0) << "; ..."<< endl;
	cout << "];" << endl << endl;
	
	/*
	// Skeleton code for iterating through the image sequence
	while (vc.read(frame2))
	{
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de

		/*
		auto begin = std::chrono::high_resolution_clock::now();
		cout << "Performing Bundle Adjustment... ";
		vector<Point3d> points;
		for (pair<Point3d, Mat> pair : _3dToDescriptorVector)
			points.push_back(pair.first);
		const vector<vector <Point2d> > imagePoints = { solver1.getImagePoints() };
		const vector<vector<int> > visibility = { vector<int>(imagePoints.size(), 1) };
		vector<Mat> cameraMatrices = { calib.getCameraMatrix() };
		vector<Mat> rotationMatrices = { solver1.getRotationMatrix() };
		vector<Mat> translationMatrices = { solver1.getTranslationVector() };
		vector<Mat> distCoeffMatrices = { calib.getDistortionCoeffs() };

		cvsba::Sba sba;
		sba.run(		points,
						imagePoints,
						visibility,
						cameraMatrices,
						rotationMatrices,
						translationMatrices,
						distCoeffMatrices);

		cout << " Done!" << endl;
		auto end = std::chrono::high_resolution_clock::now();
		cout << "Done! (" << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << " ms)" << endl << endl;
		*/
		/*
		*		Let's print all the calculated camera positions!
		*/
		Mat pos = solver1.getCameraPosition();
		logFile << setprecision(10)
			<< pos.at<double>(0, 0) << " "
			<< pos.at<double>(1, 0) << " "
			<< pos.at<double>(2, 0);
		if (i != LAST_INDEX - 1)
			logFile << "; ..." << endl;
		else
			logFile << "];" << endl;
		logFile << flush;

		frameCounter++;

		cout << endl << "****************** STARTING OVER ******************" << endl << endl << endl;
	}

	auto endOfMain = std::chrono::high_resolution_clock::now();
	cout << "Done! (" << std::chrono::duration_cast<std::chrono::milliseconds>(endOfMain - beginningOfMain).count() << " ms)" << endl << endl;


	double sum = 0;
	for (double err : reprojectionErrors)
		sum += err;
	cout << "Average reprojection error: " << sum / reprojectionErrors.size() << endl;




	logFile.close();

	return 0;
}








/*

vector< pair<Point3d, Mat> > manualStuff(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> kdtree)
{
	Mat frame1;
	Mat descriptors1, descriptors2;
	vector<KeyPoint> keypoints1, keypoints2;
	vector< pair<Point3d, Mat> > _3dToDescriptorVector;

<<<<<<< HEAD

=======
	
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
	Calibration calib;
	PnPSolver solver1, solver2;

	FeatureDetection fdetect;

	Mat distCoeffs = calib.getDistortionCoeffs();

	// MANUAL CORRESPONDENCES PNP SOLVER
	//cout << "Running first solver... ";

	vector<Point2d> imageOne;
	imageOne.push_back(Point2d(397.210571, 145.146866));	imageOne.push_back(Point2d(650.494934, 129.172379));
	imageOne.push_back(Point2d(519.567688, 131.898239));	imageOne.push_back(Point2d(531.834473, 267.480103));
	imageOne.push_back(Point2d(239.835358, 207.141220));	imageOne.push_back(Point2d(834.740051, 174.580566));
	imageOne.push_back(Point2d(211.190155, 510.402740));	imageOne.push_back(Point2d(437.319458, 218.244186));
	imageOne.push_back(Point2d(845.259948, 160.413910));	imageOne.push_back(Point2d(559.729248, 170.678528));

	solver1.setImagePoints(imageOne);
	solver1.run(1);
	//cout << "Done!" << endl << endl;

	//cout << "Running second solver... ";
	//
	//vector<Point2d> imageTwo;
	//imageTwo.push_back(Point2d(490, 250));	imageTwo.push_back(Point2d(668, 242));
	//imageTwo.push_back(Point2d(578, 242));	imageTwo.push_back(Point2d(582, 335));
	//imageTwo.push_back(Point2d(380, 294));	imageTwo.push_back(Point2d(793, 278));
	//imageTwo.push_back(Point2d(367, 499));	imageTwo.push_back(Point2d(521, 306));
	//imageTwo.push_back(Point2d(806, 262));	imageTwo.push_back(Point2d(604, 272));
	//
	//solver2.setImagePoints(imageTwo);
	//solver2.foo(0);
	//cout << "Done!" << endl;

	// This is the image we did manual matching on
	frame1 = imread("img_00433.png");

	//cout << "Running SURF/SIFT... ";
	fdetect.siftDetector(frame1, keypoints1);
	fdetect.siftExtraction(frame1, keypoints1, descriptors1);
	//cout << "Done!" << endl;

	vector<int> indexingVector;

	for (int i = 0; i < keypoints1.size(); i++)
	{
		for (int j = 0; j<imageOne.size(); j++)
		{
			if (areSame(i, imageOne[j].x, imageOne[j].y, keypoints1[i].pt.x, keypoints1[i].pt.y))
			{
				//cout << i << endl;
				//pxIdx.push_back(kpts[i]);
				indexingVector.push_back(i);
				imageOne.erase(imageOne.begin() + j);
			}
		}
	}


	Mat T = solver1.getCameraPose().clone();
	Mat K = calib.getCameraMatrix();
	vector<Point2d> imagepoints = solver1.getVoVImagePoints()[0];

	// Create a vector of "workers", each doing a separate backproject calculation.
	vector<thread> workers;

	for (int counter = 0; counter < imagepoints.size(); counter++)
<<<<<<< HEAD
	{
=======
	{	
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
		workers.push_back(
			thread(&calcBestPoint,
				&_3dToDescriptorVector,
				T,
				K,
				keypoints1,
				cloud,
				descriptors1,
				counter));

	}

	// Join acts as a "wall", so that all threads finish before the main thread continues.
	for (int i = 0; i < workers.size(); i++)
	{
		//cout << "Joining thread #" << workers[i].get_id() << endl;
		if (workers[i].joinable())
			workers[i].join();
	}

	//cout << "Size of our LUT: " << endl << _3dToDescriptorVector.size() << endl;

	return _3dToDescriptorVector;
}
*/

string type2str(int type) {
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




<<<<<<< HEAD
void prepareMap(char* mapCoordinateFile, char* mapKeypointsFile, vector<Point2d>& tunnel2D, vector<Point3d>& tunnel3D, Mat& tunnelDescriptor)
=======
void prepareMap(char* mapCoordinateFile, char* mapKeypointsFile)
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
{
	// load descriptor
	string line;
	ifstream mapFile;
	mapFile.open(mapCoordinateFile);

	double temp = 0, a = 0, b = 0, x = 0, y = 0, z = 0;

	if (mapFile.is_open())
	{
		while (getline(mapFile, line) && !mapFile.eof())
		{
			istringstream in(line);

			for (int i = 0; i<5; i++)
			{
				in >> temp;

				if (i == 0)
					a = temp;
				if (i == 1)
					b = temp;
				if (i == 2)
					x = temp;
				if (i == 3)
					y = temp;
				if (i == 4)
					z = temp;
			}
			tunnel2D.push_back(Point2f(a, b));
			tunnel3D.push_back(Point3f(x, y, z));
		}
	}
	mapFile.close();

	// load keypoints
	cv::FileStorage lstorage(mapKeypointsFile, cv::FileStorage::READ);
	lstorage["img"] >> tunnelDescriptor;
	lstorage.release();

}



<<<<<<< HEAD



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
=======
void calcBestPoint(
	vector< pair<Point3d, Mat> >& _3dToDescriptorVector,
	Mat T,
	Mat K,
	vector<KeyPoint> keypoints1,
	Mat descriptors1,
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree,
	int start,
	int end,
	int threadIndex)
{
	//vector<double> bestPoint = Reprojection::backproject(
	//	T,
	//	K,
	//	Point2d(keypoints1[counter].pt.x, keypoints1[counter].pt.y),
	//	cloud);


	for(int i = start; i < end; i += 4)
	{
		Point2d queryPoint = Point2d(keypoints1[i].pt.x, keypoints1[i].pt.y);

		vector<double> bestPoint = Reprojection::backproject(T, K, queryPoint, cloud, kdtree);

		if (bestPoint[0] == 0 && bestPoint[1] == 0 && bestPoint[2] == 0)
			return;
		//cout << setprecision(15);
		//cout << "*****************************" << endl;
		//cout << "Seaching for image point\t" << imagepoints[counter] << endl << endl;
		//cout << "The best point found:" << endl
		//	<< "X = \t" << bestPoint[0] << endl
		//	<< "Y = \t" << bestPoint[1] << endl
		//	<< "Z = \t" << bestPoint[2] << endl
		//	<< "DIST = \t" << bestPoint[3] << endl;
		//cout << "*****************************\n\n\n\n\n";

		/*
		*	Update the Look Up Table for what descriptor belongs to which image point
		*
		*	_3dToDescriptorMap.first[0]  == 3D coordinates vector
		*
		*	_3dToDescriptorVector[i].first.x == X
		*	_3dToDescriptorVector[i].first.y == Y
		*	_3dToDescriptorVector[i].first.z == Z
		*
		*	_3dToDescriptorVector[i].second == its descriptor
		*/
		// Define the 3D coordinate
		Point3d _3dcoord; _3dcoord.x = bestPoint[0]; _3dcoord.y = bestPoint[1]; _3dcoord.z = bestPoint[2];

		// Define its descriptor, should have size 1x128
		Mat desc;
		if (i > descriptors1.rows)
			return;

		desc = descriptors1.row(i);

		// Vectors are not thread safe, make sure only one thread at a time access it.
		global_mutex.lock();
		//cout << "thread " << this_thread::get_id() << " found point " << endl;
		// Push the pair into the lookup table
		_3dToDescriptorVector.push_back(make_pair(_3dcoord, desc));
		//tunnel3D.push_back(_3dcoord);
		//tunnelDescriptor.push_back(descriptors1.row(counter));
		global_mutex.unlock();
	}
}
>>>>>>> d7230b504ded9a52b2948eb00c4941adc08410de
