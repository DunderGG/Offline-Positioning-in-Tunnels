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
#include "Common.h"

#include <iostream>
#include <fstream>
#include <numeric>
#include <thread>
#include <future>
#include <chrono>
#include <mutex>

#define STARTIDX                433             // frame start, it could be the previous frame or the latter one as long it finds a hit with LUT
#define FINISHIDX               523             // last frame in the sequence
#define SLIDINGWINDOWSIZE       1               // number of frames for sliding window, both LUT window and Bundle Adjustment
#define NUMTHREADS              8               // http://stackoverflow.com/questions/1718465/optimal-number-of-threads-per-core
#define STATICNTASK             480             // only for static task assignments to each threads. 480 tasks for each thread
#define DRAWKPTS                1               // mode for drawing keypoints within cv::Mat input image and save it to .png
#define SEQMODE                 0               // mode for parallel threads or sequential
// #define LOGMODE                 1               // mode for logging, uncomment if not using cmake

using namespace std;
using namespace cv;
using namespace pcl;
using namespace std::chrono;

// global variable for tunnel GPS mapping
// unordered_map<Mat, Point3f> tunnelLut;
vector<Point2d>        tunnel2D;
vector<Point3d>        tunnel3D;
vector<Point3d>        _3dTemp;             // found 3d world points
vector<Point2d>        _2dTemp;             // corresponding 2d pixels
vector<int>            _1dTemp;             // corresponding indices relative to the query set
vector<int>            _slidingWindowSize;  // contains of last 5 frame's found 3D points
Mat                    tunnelDescriptor;
mutex                  g_mutex;

void mpThread ( Mat T, Mat K, vector<KeyPoint> imagepoint, Mat descriptor,
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> kdtree,
                int start, int end, int tidx);

// THE MAIN START HERE
int main (int argc, char *argv[])
{
    //all class objects
    Calibration calib;
    PnPSolver solver, solverRefined;
    FeatureDetection fdetect;
    Common com;

    //for logging
    ofstream logFile, logMatrix, correspondences, correspondencesRefined;

    // process the input arguments if it's correct
    int startFrame, lastFrame;
    if (argc == 3)
    {
        startFrame = atoi(argv[1]);
        lastFrame  = atoi(argv[2]);
    }
    else
    {
        startFrame = STARTIDX;
        lastFrame = FINISHIDX;
    }

    cout << "starting the simulation starting at frame-" << startFrame << " to " << lastFrame << endl;

    //create directory for log
    if (LOGMODE)
    {
        com.createDir("log");

        // create a file for logging posiions
        logFile.open ("./log/logPoses.txt", std::ios::out);
        logMatrix.open ("./log/logMatrix.txt", std::ios::out);
    }

    // 1. load pointcloud
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    io::loadPCDFile("gnistangtunneln-semifull-voxelized.pcd", *cloud);
    std::cerr 	<< "pointcloud before filtering: " << cloud->width * cloud->height
                << " data points (" << pcl::getFieldsList (*cloud) << ")" << std::endl;

    // prepare the kdtree
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);

    // 2. prepare the manual correspondences as a lookup table
    char map2Dto3D  [100];
    char mapDescrip [100];

    sprintf(map2Dto3D, "/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT/ManualCorrespondences.txt");
    sprintf(mapDescrip,"/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT/ManualCorrespondences.yml");

    com.prepareMap(map2Dto3D, mapDescrip, ref(tunnel2D), ref(tunnel3D), ref(tunnelDescriptor));

    // 3. start the routing and initiate all variables
    char pathname[100] = "/Users/januaditya/Thesis/exjobb-data/volvo/out0/";
    char nextimage[100];
    char poseFileIdx[100];
    char poseRefinedFileIdx[100];

    Mat T, K;
    Point3d _3dcoord;
    Mat descTemp;
    int clearCounter = 0;

    int idx = startFrame;
    while (idx < lastFrame)
    {
        // init the logging, create a file, two for each frame index. one for initial pose and one for refined (after backprojection)
        if (LOGMODE)
        {
            sprintf(poseFileIdx, "./log/%d-poseFile.txt", idx);
            sprintf(poseRefinedFileIdx, "./log/%d-poseRefined.txt", idx);

            correspondences.open(poseFileIdx, std::ios::out);
            correspondencesRefined.open(poseRefinedFileIdx, std::ios::out);
        }

        // start timer
        com.startTimer();

        cout << "processing image-" << idx << "...";

        // 4. load images
        sprintf(nextimage, "%simg_%05d.png", pathname, idx);
        Mat img = imread(nextimage);

        // 4.1 set the ROI (region of interest)
        // this mask is to take only 50% upper part of the image
        Mat img_maskUpperPart = Mat::zeros(img.size(), CV_8U);
        Mat img_roiUpperPart (img_maskUpperPart, Rect(0, 0, img.cols, img.rows*7/8));
        img_roiUpperPart = Scalar(255, 255, 255);

        // this mask is to take 25% of the bottom right part of the image
        // Mat img_maskRightPart = Mat::zeros(img.size(), CV_8U);
        // Mat img_roiRightPart (img_maskRightPart, Rect(img.cols*3/5, img.rows*2/5, img.cols*2/5, img.rows*2/5));
        // img_roiRightPart = Scalar(255, 255, 255);

        // combine the masks
        // Mat img_combinedMask = img_maskUpperPart | img_maskRightPart;

        // 5. perform sift feature detection and get kpt_i
        vector<KeyPoint> detectedkpts;
        fdetect.siftDetector(img, detectedkpts, img_maskUpperPart);

        // 6. perform sift feature extraction and get desc_i
        Mat descriptor;
        fdetect.siftExtraction(img, detectedkpts, descriptor);

        // 6.1 draw the features
        if (DRAWKPTS && (idx == startFrame))
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
        cout << "size of lookup table is " << tunnelDescriptor.rows << endl;

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

            // based on lowe's it's 0.8 * dist, however we're using 0.7 for now
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

            // save the initial matched 3D-to-2D correspondences
            if (LOGMODE)
            {
               correspondences << std::fixed << setprecision(4)
                               << tunnel3D[matchedXYZ[i]].x << ", "
                               << tunnel3D[matchedXYZ[i]].y << ", "
                               << tunnel3D[matchedXYZ[i]].z << ", " << std::flush;

               correspondences << std::fixed << setprecision(4)
                               << detectedkpts[matchedIndices[i]].pt.x << ", "
                               << detectedkpts[matchedIndices[i]].pt.y << "\n" << std::flush;
            }
        }

        // 10. solvePNP using the XYZ
        cout << "  camera pose at frame-" << idx << ": ";
        solver.setImagePoints(retrieved2D);
        solver.setWorldPoints(retrieved3D);
        solver.run(1);

        // 11. reproject all 2D keypoints to 3D
        T = solver.getCameraPose().clone();                 // it is the camera center in world coordinates
        K = calib.getCameraMatrix();

        Mat R = solver.getRotationMatrix();
        Mat t = solver.getTranslationVector();

        // save the initial camera pose for frame-idx
        if (LOGMODE)
        {
            // save the initial camera position
            logFile << std::fixed << setprecision(10)
                    << T.at<double>(0,3) << ", "
                    << T.at<double>(1,3) << ", "
                    << T.at<double>(2,3) << ", " << std::flush;

            // save the initial Rotation and Translation matrices from PnP solver
            logMatrix << std::fixed << setprecision(10)
                      << R.at<double>(0,0) << ", " << R.at<double>(0,1) << ", " << R.at<double>(0,2) << ", "
                      << R.at<double>(1,0) << ", " << R.at<double>(1,1) << ", " << R.at<double>(1,2) << ", "
                      << R.at<double>(2,0) << ", " << R.at<double>(2,1) << ", " << R.at<double>(2,2) << ", "
                      << t.at<double>(0)   << ", " << t.at<double>(1)   << ", " << t.at<double>(2)   << ", " << std::flush;
        }

        // 12. clear, timewindow for LUT is 5 frames
//        if ((clearCounter != 0) && (clearCounter % SLIDINGWINDOWSIZE == 0))
//        {
//            Mat temp;
//
//            // erase tunnel3D
//            for (int i=0; i < _slidingWindowSize[0]; i++)
//                tunnel3D.erase(tunnel3D.begin());
//
//            // copy the remaining last windowsize-1 descriptors in a temporary place
//            if (temp.rows != 0)
//                temp.release();
//
//            for (int i=_slidingWindowSize[0]; i < tunnelDescriptor.rows; i++)
//                temp.push_back(tunnelDescriptor.row(i));
//
//            // release the descriptor
//            tunnelDescriptor.release();
//
//            // copy back it again
//            tunnelDescriptor = temp.clone();
//        }

        tunnel2D.clear();
        tunnel3D.clear();
        tunnelDescriptor.release();

        // 13. backprojecting all keypoints to the cloud
        if (SEQMODE)
        {
            vector<double> bestPoint{ 0, 0, 0, 1000 };
            for(int counter = 0; counter < detectedkpts.size(); counter++)
            {
                cout << "  " << counter << "-th step: ";

                Point2d queryPoints = Point2d(detectedkpts[counter].pt.x, detectedkpts[counter].pt.y);

                cout << "backprojecting " << "(" << queryPoints.x << "," << queryPoints.y << ")" << "...";
                bestPoint = Reprojection::backprojectRadius(T, K, queryPoints, cloud, kdtree);

                // Define the 3D coordinate
                _3dcoord.x = bestPoint[0];
                _3dcoord.y = bestPoint[1];
                _3dcoord.z = bestPoint[2];

                // Push the pair into the lookup table if it's not zero
                if ((_3dcoord.x > 0.0f) && (_3dcoord.y > 0.0f) && (_3dcoord.z > 0.0f))
                {
                    cout << "found a point...(" << ") at px-" << counter << endl;

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
            cout << "  going parallel to backproject " << detectedkpts.size() << " keypoints into the cloud" << endl;

            int numtask;

            // if keypoints are too many
            numtask = floor(detectedkpts.size()/NUMTHREADS);

            for (int tidx = 0; tidx < NUMTHREADS; tidx++)
            {
                int start = tidx    * numtask;
                int end   = (tidx+1)* numtask;

                // spawn threads
                ts[tidx] = new thread (mpThread, T, K, detectedkpts, descriptor, cloud, kdtree, start, end, tidx);
            }

            for (int tidx = 0; tidx < NUMTHREADS; tidx ++)
            {
                ts[tidx]->join();
            }

            cout << "  succesfully backproject " << _3dTemp.size() << " 3d points" << endl;

            // add found 3D points into 5 frames sliding window
            _slidingWindowSize.push_back(_3dTemp.size());
        }


        // 14. redo the pnp solver, now using all found 3D points
        cout << "  now solving again using " << _2dTemp.size() << " keypoints ";
        cout << "and 3d correspondences" << endl;

        cout << "  refined pose at frame-" << idx << ": ";
        solverRefined.setImagePoints(_2dTemp);
        solverRefined.setWorldPoints(_3dTemp);
        solverRefined.run(1);

        // 15. check reprojection error of each backprojected world points
        vector<Point2d> reprojectedPixels;
        projectPoints(_3dTemp,
                      solver.getRotationMatrix(),
                      solver.getTranslationVector(),
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

        cout << "  accumulate reprojection error is " << repError << endl;
        cout << "  avg reprojection error for " << _1dTemp.size() << " points is: " << repError/_1dTemp.size() << " px " << endl;


        // 16. one of the optimization part


        // save the refined initial camera pose for frame-idx
        if (LOGMODE)
        {
            // save the 3D-to-2D correspondences
            for (int i=0; i<_2dTemp.size(); i++)
            {
                correspondencesRefined << std::fixed << setprecision(4)
                                       << _3dTemp[i].x << ", "
                                       << _3dTemp[i].y << ", "
                                       << _3dTemp[i].z << ", " << std::flush;

                correspondencesRefined << std::fixed << setprecision(4)
                                       << _2dTemp[i].x << ", "
                                       << _2dTemp[i].y << "\n" << std::flush;
            }

            // save the refined vehicle position, num of keypoints, num of backprojected points and reprojection errror
            T = solver.getCameraPose().clone();

            logFile << std::fixed << setprecision(4)
                    << T.at<double>(0,3) << ", "
                    << T.at<double>(1,3) << ", "
                    << T.at<double>(2,3) << ", " << std::flush;

            logFile << std::fixed << setprecision(4)
                    << detectedkpts.size()      << ", "
                    << _3dTemp.size()           << ", "
                    << repError/_1dTemp.size()  << "\n" << std::flush;      // end of logPose

            // save the Rotation and Translation matrices
            R = solver.getRotationMatrix();
            t = solver.getTranslationVector();

            logMatrix << std::fixed << setprecision(4)
                      << R.at<double>(0,0) << ", " << R.at<double>(0,1) << ", " << R.at<double>(0,2) << ", "
                      << R.at<double>(1,0) << ", " << R.at<double>(1,1) << ", " << R.at<double>(1,2) << ", "
                      << R.at<double>(2,0) << ", " << R.at<double>(2,1) << ", " << R.at<double>(2,2) << ", "
                      // end of logMatrix
                      << t.at<double>(0)   << ", " << t.at<double>(1)   << ", " << t.at<double>(2)   << "\n" << std::flush;
        }


        // 17. report the exec
        com.reportTimer();


        // 18. draw succesfully reprojected 2Ds


        //close the files
        if (LOGMODE)
        {
           correspondences.close();
           correspondencesRefined.close();
        }

        // 19. clear all temporary variables
        _1dTemp.clear();
        _2dTemp.clear();
        _3dTemp.clear();

        // increase necessary counter
        idx++;
        // clearCounter++;
    }

    logFile.close();

    return 0;
}

void mpThread ( Mat T, Mat K, vector<KeyPoint> imagepoint, Mat descriptor,
                pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> kdtree,
                int start, int end, int tidx)
{
    vector <double> temp = {0,0,0,1000};
    Point3d _mp3dcoord;

    for (int i=start; i<end; i++)
    {
        // temp = Reprojection::backprojectRadius(T, K, Point2d(imagepoint[i].pt.x,imagepoint[i].pt.y), cloud, kdtree);
        temp = Reprojection::backproject(T, K, Point2d(imagepoint[i].pt.x,imagepoint[i].pt.y), cloud, kdtree);

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

                // For verifying PnP
                _3dTemp.push_back(_mp3dcoord);
                _2dTemp.push_back(Point2d(imagepoint[i].pt.x,imagepoint[i].pt.y));
                _1dTemp.push_back(i);
                //cout << "      tidx-" << tidx << " - found a point at px-" << i << endl;
            }
        }
        // else
        //     cout << "      tidx-" << tidx << " - nothing found at px-" << i << endl;
    }
}
