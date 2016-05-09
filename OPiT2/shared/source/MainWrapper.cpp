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
#include "FeatureDetection.h"
#include "Calibration.h"
#include "PnPSolver.h"
#include "PointProjection.h"
#include "PCLCloudSearch.h"
#include "Reprojection.h"

#include <iostream>
#include <numeric>
#include <thread>
#include <future>
#include <chrono>
#include <mutex>

#define STARTIDX                433
#define FINISHIDX               533
#define NUMTHREADS              8
#define SEQMODE                 0
#define STATICNTASK             480
#define DRAWKPTS                1
#define KEYPOINTSUPPERLIM       2400

using namespace std;
using namespace cv;
using namespace pcl;
using namespace std::chrono;

// global variable for tunnel GPS mapping
//unordered_map<Mat, Point3f> tunnelLut;
vector<Point2d>        tunnel2D;
vector<Point3d>        tunnel3D;
vector<Point3d>        _3dTemp;
vector<Point2d>        _2dTemp;
Mat                    tunnelDescriptor;
mutex                  g_mutex;

int tempCount = 0;
void prepareMap (char* mapCoordinateFile, char* mapKeypointsFile);
void mpThread (Mat T, Mat K, vector<KeyPoint> imagepoint, Mat descriptor, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int start, int end, int tidx);

int MainWrapper()
{
    //threadTest();

    //Calibration moved to its own class.
    Calibration calib;
	PnPSolver solver;
    FeatureDetection fdetect;

    // 1. load pointcloud
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    io::loadPCDFile("cloud-voxelized.pcd", *cloud);
    std::cerr 	<< "PointCloud before filtering: " << cloud->width * cloud->height
                << " data points (" << pcl::getFieldsList (*cloud) << ")" << std::endl;

    // 2. prepare the manual correspondences as a lookup table
    char map2Dto3D  [100];
    char mapDescrip [100];

    sprintf(map2Dto3D, "/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT/ManualCorrespondences.txt");
    sprintf(mapDescrip,"/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT/ManualCorrespondences.yml");

    prepareMap(map2Dto3D, mapDescrip);

    // 3. start the routing and initiate all variables
    char pathname[100] = "/Users/januaditya/Thesis/exjobb-data/volvo/out0/";
    char nextimage[100];

    Mat T, K;
    Point3d _3dcoord;
    Mat descTemp;

    int idx = STARTIDX;
    while (idx < FINISHIDX)
    {
        // start timer
        high_resolution_clock::time_point t1, t2;
        t1 = high_resolution_clock::now();

        cout << "processing image-" << idx << "...";

        // 4. load images
        sprintf(nextimage, "%simg_%05d.png", pathname, idx);
        Mat img = imread(nextimage);

        // 4.1 set the ROI (region of interest)
        // this mask is to take only 50% upper part of the image
        Mat img_maskUpperPart = Mat::zeros(img.size(), CV_8U);
        Mat img_roiUpperPart (img_maskUpperPart, Rect(0, 0, img.cols, img.rows/2));
        img_roiUpperPart = Scalar(255, 255, 255);

        // this mask is to take 25% of the bottom right part of the image
        Mat img_maskRightPart = Mat::zeros(img.size(), CV_8U);
        Mat img_roiRightPart (img_maskRightPart, Rect(img.cols*2/5, img.rows/2, img.cols*2/5, img.rows*2/5));
        img_roiRightPart = Scalar(255, 255, 255);

        // combine the masks
        Mat img_combinedMask = img_maskUpperPart | img_maskRightPart;

        // 5. perform sift feature detection and get kpt_i
        vector<KeyPoint> detectedkpts;
        fdetect.siftDetector(img, detectedkpts, img_combinedMask);

        // 6. perform sift feature extraction and get desc_i
        Mat descriptor;
        fdetect.siftExtraction(img, detectedkpts, descriptor);

        // 6.1 draw the features
        if (DRAWKPTS && (idx == 433))
        {
            vector<KeyPoint> detectedkptsnonROI;
            Mat descriptorNonROI;
            Mat outputROI;
            Mat outputNonROI;

            Mat img2 = imread(nextimage);

            fdetect.siftDetector(img2, detectedkptsnonROI);
            fdetect.siftExtraction(img2, detectedkptsnonROI, descriptorNonROI);

            drawKeypoints(img2, detectedkptsnonROI, outputNonROI, Scalar(249, 205, 47), 4);
            drawKeypoints(img, detectedkpts, outputROI, Scalar(249, 205, 47), 4);

            imwrite("entrance-sift-non-roi.png", outputNonROI);
            imwrite("entrance-sift-roi.png", outputROI);
        }

        // 7. match between
        vector<vector<DMatch> > matches;
        fdetect.bfMatcher(descriptor, tunnelDescriptor, matches);

        // 8. retrieve the matches indices from the descriptor
        vector<int> matchedIndices;
        vector<int> matchedXYZ;

        for (int i = 0; i < matches.size(); ++i)
        {
            DMatch first  = matches[i][0];

            auto dist1 = matches[i][0].distance;
            auto dist2 = matches[i][1].distance;

            // based on lowe's it's 0.8 * dist, however we're using 0.6 for now
            if(dist1 < fdetect.getSiftMatchingRatio() * dist2)
            {
                matchedIndices.push_back(first.trainIdx);
                matchedXYZ.push_back(first.queryIdx);
            }
        }

		cout << setprecision(10);
        cout << "found " << matchedXYZ.size() << " matches with the LUT" << endl;

        // 9. use the indices to retrieve the XYZ from LUT
        vector<Point2d> retrieved2D;
        vector<Point3d> retrieved3D;

        retrieved2D.clear();
        retrieved3D.clear();

        for (int i = 0; i< matchedIndices.size(); i++)
        {
            retrieved2D.push_back(Point2d(detectedkpts[matchedIndices[i]].pt.x,detectedkpts[matchedIndices[i]].pt.y));
            retrieved3D.push_back(Point3d(tunnel3D[matchedXYZ[i]].x,tunnel3D[matchedXYZ[i]].y,tunnel3D[matchedXYZ[i]].z));
            cout << std::fixed << setprecision(4);
            cout << "   pushed {" << detectedkpts[matchedIndices[i]].pt.x << ", " << detectedkpts[matchedIndices[i]].pt.y << "} --> {"
                                 << tunnel3D[matchedXYZ[i]].x << ", " << tunnel3D[matchedXYZ[i]].y << ", " << tunnel3D[matchedXYZ[i]].z << "}" << endl;
        }

        // 10. solvePNP using the XYZ
        solver.setImagePoints(retrieved2D);
        solver.setWorldPoints(retrieved3D);
        solver.foo(1);

        // 11. reproject all 2D keypoints to 3D
        T = solver.getCameraPose().clone();
        K = calib.getCameraMatrix();

        // print camera pose
        cout << std::fixed << setprecision(4);
        cout << "initial camera pose [" << T.at<double>(0,3) << ", "
                                        << T.at<double>(1,3) << ", "
                                        << T.at<double>(2,3) << "]" << endl;

        // 12. clear
        _2dTemp.clear();
        _3dTemp.clear();
        tunnel3D.clear();
        tunnelDescriptor.release();

        if (SEQMODE)
        {
            vector<double> bestPoint{ 0, 0, 0, 1000 };
            for(int counter = 0; counter < detectedkpts.size(); counter++)
            {
                cout << "  " << counter << "-th step: ";

                Point2d queryPoints = Point2d(detectedkpts[counter].pt.x, detectedkpts[counter].pt.y);

                cout << "backprojecting " << "(" << queryPoints.x << "," << queryPoints.y << ")" << "...";
                bestPoint = Reprojection::backprojectRadius(T, K, queryPoints, cloud);

                // Define the 3D coordinate
                _3dcoord.x = bestPoint[0];
                _3dcoord.y = bestPoint[1];
                _3dcoord.z = bestPoint[2];

                // Push the pair into the lookup table if it's not zero
                if ((_3dcoord.x > 0.0f) && (_3dcoord.y > 0.0f) && (_3dcoord.z > 0.0f))
                {
                    tempCount++;
                    cout << "found a point...(" << tempCount << ") at px-" << counter << endl;

                    // 12. Update the LUT
                    tunnel3D.push_back(_3dcoord);
                    tunnelDescriptor.push_back(descriptor.row(counter));

                    // For verifying PnP
                    _3dTemp.push_back(_3dcoord);
                    _2dTemp.push_back(queryPoints);
                }
                else
                cout << "nothing found..." << endl;
            }
        }
        else
        // parallel
        {
            thread *ts[NUMTHREADS];
            cout << "\ngoing parallel..." << endl;

            int numtask;
            // if keypoints are too many
            if (detectedkpts.size() > KEYPOINTSUPPERLIM)
                numtask = KEYPOINTSUPPERLIM/NUMTHREADS;
            // if keypoints are sufficient
            else
                numtask = floor(detectedkpts.size()/NUMTHREADS);

            for (int tidx = 0; tidx < NUMTHREADS; tidx++)
            {
                int start = tidx    * numtask;
                int end   = (tidx+1)* numtask;

                // spawn threads
                ts[tidx] = new thread (mpThread, T, K, detectedkpts, descriptor, cloud, start, end, tidx);
            }

            for (int tidx = 0; tidx < NUMTHREADS; tidx ++)
            {
                ts[tidx]->join();
            }
        }

        //redo the pnp solver
        cout << "\nnow solving again using " << _2dTemp.size() << " keypoints ";
        cout << "and 3d correspondences" << endl;
        solver.setImagePoints(_2dTemp);
        solver.setWorldPoints(_3dTemp);
        solver.foo(1);

        // print camera pose
        cout << std::fixed << setprecision(4);
        cout << "updated camera pose [" << T.at<double>(0,3) << ", "
                                        << T.at<double>(1,3) << ", "
                                        << T.at<double>(2,3) << "]" << endl;

        //increase the idx
        idx++;

        //report the exec
        t2 = high_resolution_clock::now();
        auto duration1 = duration_cast<milliseconds>( t2 - t1 ).count();
        cout << "finish in " << duration1 << "ms (" << duration1/1000 << " sec)" << endl;
    }

    return 1;
}

void prepareMap (char* mapCoordinateFile, char* mapKeypointsFile)
{
    // load descriptor
    string line;
    ifstream mapFile;
    mapFile.open(mapCoordinateFile);

    double temp=0,a=0,b=0,x=0,y=0,z=0;

    if (mapFile.is_open())
    {
        while (getline(mapFile, line) && !mapFile.eof())
        {
            istringstream in(line);

            for (int i=0; i<5; i++)
            {
                in >> temp;

                if (i==0)
                    a = temp;
                if (i==1)
                    b = temp;
                if (i==2)
                    x = temp;
                if (i==3)
                    y = temp;
                if (i==4)
                    z = temp;
            }
            tunnel2D.push_back(Point2f(a,b));
            tunnel3D.push_back(Point3f(x,y,z));
        }
    }
    mapFile.close();

    // load keypoints
    cv::FileStorage lstorage(mapKeypointsFile, cv::FileStorage::READ);
    lstorage["img"] >> tunnelDescriptor;
    lstorage.release();

}

void mpThread (Mat T, Mat K, vector<KeyPoint> imagepoint, Mat descriptor, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, int start, int end, int tidx)
{
    vector <double> temp = {0,0,0,1000};
    Point3d _mp3dcoord;

    for (int i=start; i<end; i+=4)
    {
        temp = Reprojection::backprojectRadius(T, K, Point2d(imagepoint[i].pt.x,imagepoint[i].pt.y), cloud);

        // Define the 3D coordinate
        _mp3dcoord.x = temp[0];
        _mp3dcoord.y = temp[1];
        _mp3dcoord.z = temp[2];

        // Push the pair into the lookup table if it's not zero
        if ((_mp3dcoord.x > 0.0f) && (_mp3dcoord.y > 0.0f) && (_mp3dcoord.z > 0.0f))
        {
            // 12. Update the LUT
            lock_guard<mutex> lock(g_mutex);
            {
                tunnel2D.push_back(Point2d(imagepoint[i].pt.x,imagepoint[i].pt.y));
                tunnel3D.push_back(_mp3dcoord);
                tunnelDescriptor.push_back(descriptor.row(i));
                tempCount++;

                // For verifying PnP
                _3dTemp.push_back(_mp3dcoord);
                _2dTemp.push_back(Point2d(imagepoint[i].pt.x,imagepoint[i].pt.y));

                //cout << "      tidx-" << tidx << " - found a point at px-" << i << "(" << tempCount << ")" << endl;
            }
        }
        // else
        //     cout << "      tidx-" << tidx << " - nothing found at px-" << i << endl;
    }
}
