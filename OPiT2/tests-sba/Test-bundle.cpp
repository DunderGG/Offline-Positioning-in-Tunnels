// Test-bundle.cpp
//  a main .cpp file for testing whole solution and bundle adjustment at once
#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <numeric>

#include <opencv2/opencv.hpp>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

//  include all class files
#include "Common.h"
#include "Frame.h"
#include "FeatureDetection.h"
#include "PnPSolver.h"
#include "Calibration.h"
#include "BundleAdjust.h"

//  all definitions of variables
#define WINDOWSIZE              5                       // number of tracked frames every timestep
#define NUMTHREADS              8                       // number of threads spawned for backprojections
#define PNPITERATION            1000                    // number of iteration for pnp solver
#define PNPPIXELERROR           5                       // toleration of error pin pixel square
#define PNPCONFIDENCE           0.99                    // confidence level of 99%
#define LENGTHFRAME             100                     // number of processed frames
#define MINFRAMEIDX             433                     // default frame index
#define MAXFRAMEIDX             MINFRAMEIDX-LENGTHFRAME // default frame index + length
#define MINCORRESPONDENCES      10                       // minimum amount of 3D-to-2D correspondencs of PnP

//  all namespaces
using namespace std;
using namespace cv;
using namespace pcl;

// path to the cloud, tunnel initial correspondences and image sequences
const string cloudPath   = "/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT2/cloud/gnistangtunneln-full-voxelized.pcd";
const string map2Dto3D   = "/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT/ManualCorrespondences.txt";
const string mapDesc     = "/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT/ManualCorrespondences.yml";
const string imgPath     = "/Users/januaditya/Thesis/exjobb-data/volvo/tunnel-frames/";
//const string imgPath     = "/Users/januaditya/Desktop/thesis/gopro/frames/";

//  start the main
int main (int argc, char* argv[])
{
    // declare all the class objects
    Common com;
    FeatureDetection fdet;
    PnPSolver solver;
    Calibration cal;
    BundleAdjust bundle;

    // declare all variables for global lookup table
    vector<pair<Point3d, Mat> >     _3dToDescriptorTable;

    // declare all variables for local frame information
    vector<Point3d>                 _tunnel3D;
    vector<Point2d>                 _tunnel2D;
    Mat                             _tunnelDescriptor;

    // set class objects initial parameters
    solver.setPnPParam (PNPITERATION, PNPPIXELERROR, PNPCONFIDENCE);

    // point cloud variables
    int startFrame, endFrame;                                   // marks index for frame start/end
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);  // pointer to the cloud
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;

    // log
    ofstream logFile, logMatrix, correspondences, correspondencesRefined;
    char poseFileIdx[100]; char poseRefinedFileIdx[100];
    {
        com.createDir("log");

        //open files for logging
        logFile.open("./log/logPoses.txt", std::ios::out);
        logMatrix.open("./log/logMatrix.txt", std::ios::out);
    }

    // handle the input arguments
    if (argc !=3)
    {
        cout << "running the simulation for " << LENGTHFRAME << " frames" << endl;
        cerr << "wrong number of input arguments..." << endl;
        cerr << "using default parameter instead..." << endl;

        startFrame = MINFRAMEIDX;
        endFrame   = MAXFRAMEIDX;
     }
    else
    {
        startFrame = atoi(argv[1]);
        endFrame   = atoi(argv[2]);
        cout << "running the simulation for " << endFrame - startFrame << " frames" << endl;
    }

    // load the point cloud, create kd-tree, and report the cloud dimension
    io::loadPCDFile(cloudPath, *cloud);
    kdtree.setInputCloud(cloud);
    cout << "loaded cloud with " << cloud->width * cloud->height << " points ("
         << getFieldsList (*cloud) << ")" << endl;

    // prepare the 2D, 3D and descriptor correspondences from files and initialise the lookuptable
    com.prepareMap(map2Dto3D, mapDesc, ref(_tunnel2D), ref(_tunnel3D), ref(_tunnelDescriptor));
    com.updatelut(_tunnel3D, _tunnelDescriptor, ref(_3dToDescriptorTable));

    // init all objects and vars for the main sequences, in order of definition
    int frameIndex = startFrame;
    int frameCount = 0;
    char currImgPath[100];
    vector<Frame> windowedFrame;

    Frame current;
    Frame prev;

    
    
    
    // start the positioning sequences
    while (frameIndex > endFrame)
    {
        cout << "processing frame-" << frameCount << "..." << endl;

        
        // log
        {
            sprintf(poseFileIdx, "./log/%d-poseFile.txt", frameIndex);
            sprintf(poseRefinedFileIdx, "./log/%d-poseRefined.txt", frameIndex);

            correspondences.open(poseFileIdx, std::ios::out);
            correspondencesRefined.open(poseRefinedFileIdx, std::ios::out);
        }

        
        // create local object of frame and automatically cleared every new iteration
        vector<int> matchesIndex3D;
        vector<int> matchesIndex2D;

        // load the image into the current frame
        sprintf(currImgPath, "%simg_%05d.png", imgPath.c_str(), frameIndex);
        //sprintf(currImgPath, "%s%04d.png", imgPath.c_str(), frameIndex);

        
        // init current Frame
        current = Frame();
        current.frameIdx = frameCount;
        current.image    = imread(currImgPath);

        
        
        // preprocess the image to remove visible outliers, e.g. dashboard
        Mat mask = Mat::zeros(current.image.size(), CV_8U);
        Mat roi (mask, Rect(0, 0, current.image.cols, current.image.rows*7/8));
        roi = Scalar(255, 255, 255);

        
        
        // detect features from current frame with provided region of interest mask
        fdet.siftDetector(current.image, current.keypoints, mask);

        // extract feature descriptors from current frame
        fdet.siftExtraction(current.image, current.keypoints, current.descriptors);

        
        
        Mat lutDesc;
        // match the current frame descriptor with every frames in the window
        if (windowedFrame.size() != 0)
        {
            for (int i=windowedFrame.size()-1; i>=0; i--)
            {
                // if not empty, then correspondences has to be checked for every frames in the window
                lutDesc = com.getdescriptor(windowedFrame[i]._3dToDescriptor);

                // match descriptor
                fdet.bfMatcher(lutDesc, current.descriptors, current.matches);

                // perform lowe's ratio test, last parameter is for cout verbose (true/false)
                fdet.ratioTestRansac(current.matches, ref(windowedFrame[i]), ref(current), false);
            }
        }
        // if window empty
        else
        {
            // if empty, the correspondences are obtained from the lookuptable
            lutDesc = com.getdescriptor(_3dToDescriptorTable);

            // clean LUT
            _3dToDescriptorTable.clear();

            // matcher
            fdet.bfMatcher(lutDesc, current.descriptors, current.matches);

            // perform David Lowe's ratio test. it gives 3D/2D indices to use in the next step
            fdet.ratioTest(current.matches, ref(matchesIndex3D), ref(matchesIndex2D));

            // retrieve the 3D from the lookup table, 2D from current frame's keypoints
            if (matchesIndex2D.size() != matchesIndex3D.size())
                cerr << "wrong size of 2D/3D correspondences" << endl;
            else
            {
                for (int i=0; i<matchesIndex2D.size(); i++)
                {
                    current.matchedWorldPoints.push_back(Point3d(_3dToDescriptorTable[matchesIndex3D[i]].first.x,
                                                                 _3dToDescriptorTable[matchesIndex3D[i]].first.y,
                                                                 _3dToDescriptorTable[matchesIndex3D[i]].first.z));
                    current.matchedImagePoints.push_back(Point2d(current.keypoints[matchesIndex2D[i]].pt.x,
                                                                 current.keypoints[matchesIndex2D[i]].pt.y));
                }
            }
        }

        
        
        // once we got the 3D and 2D correspondences and bigger than min correspondences, compute the camera pose
        if (current.matchedWorldPoints.size() < MINCORRESPONDENCES)
        {
            cerr << "not enough correspondences for PnP" << endl;
            exit(EXIT_FAILURE);
        }
        else
        {
            cout << "  found " << current.matchedWorldPoints.size() << " 3D points from lookup table window" << endl;

            // reinit solver
            solver = PnPSolver();
            solver.setPnPParam (PNPITERATION, PNPPIXELERROR, PNPCONFIDENCE);

            solver.setImagePoints(current.matchedImagePoints);
            solver.setWorldPoints(current.matchedWorldPoints);
            cout << "  result after matching...";
            solver.run(1);
        }

        
        
        
        // find the rest image points 3D representation using backprojections
        // prepare the matrices
        {
            current.K           = cal.getCameraMatrix();
            current.distCoef    = cal.getDistortionCoeffs();
            current.R           = solver.getRotationMatrix();
            current.R_rodrigues = solver.getRotationVector();
            current.t           = solver.getTranslationVector();

            // get the camera pose matrix (3x4). it consists of camera position and the transposed rotation matrix
            current.cameraPose  = solver.getCameraPose();

            // get the transposed/inversed rotation matrix (3x3).
            // we can copy 3x3 upper left from the cameraPose (3x4) or just take the inverse of rotation matrix
            current.R_invert    = current.R.inv();
            current.t_invert    = (-(current.R_invert) * current.t);

            // for the bundle adjustment, we need every 3d points to be in camera coordinates
            current.projectWorldtoCamera();

            // get the camera position (3x1) as t_invert.
            // the alternative way is to copy 3x1 rightmost column from the cameraPose (3x4)
            if (frameCount == 0)        // initial frame/camera at index-0
            {
                current.t_translation   = current.t_invert;
            }
            else                        // rest of the frames
            {
                current.t_translation   = current.t_invert - prev.t_invert;
            }

            // log for matched points
            {
                // save the current frame retrieved pair 2D and 3D
                for (int i=0; i<current.matchedImagePoints.size(); i++)
                {
                    correspondences << std::fixed << setprecision(4)
                    << current.matchedWorldPoints[i].x << ", "
                    << current.matchedWorldPoints[i].y << ", "
                    << current.matchedWorldPoints[i].z << ", " << std::flush;

                    correspondences << std::fixed << setprecision(4)
                    << current.matchedImagePoints[i].x << ", "
                    << current.matchedImagePoints[i].y << "\n" << std::flush;
                }

                // save the initial camera position
                logFile << std::fixed << setprecision(10)
                        << current.cameraPose.at<double>(0,3) << ", "
                        << current.cameraPose.at<double>(1,3) << ", "
                        << current.cameraPose.at<double>(2,3) << "\n" << std::flush;

                // save the initial Rotation and Translation matrices from PnP solver
                logMatrix << std::fixed << setprecision(10)
                          << current.R.at<double>(0,0) << ", " << current.R.at<double>(0,1) << ", " << current.R.at<double>(0,2) << ", "
                          << current.R.at<double>(1,0) << ", " << current.R.at<double>(1,1) << ", " << current.R.at<double>(1,2) << ", "
                          << current.R.at<double>(2,0) << ", " << current.R.at<double>(2,1) << ", " << current.R.at<double>(2,2) << ", "
                          << current.t.at<double>(0)   << ", " << current.t.at<double>(1)   << ", " << current.t.at<double>(2)   << "\n" << std::flush;
            }
        }

        
        
        
        
        // call the multithreaded backprojection wrapper
        com.threading(NUMTHREADS,
                      current.cameraPose,
                      current.K,
                      current.keypoints,
                      current.descriptors,
                      std::ref(cloud),
                      std::ref(kdtree),
                      std::ref(current._3dToDescriptor),                        // pair of 3d reprojected world points & descriptors (it's not LUT)
                      std::ref(current.reprojectedWorldPoints),                 // 3d reprojected world points
                      std::ref(current.reprojectedImagePoints),                 // 2d reprojected image points
                      std::ref(current.reprojectedIndices));                    // vector that contains reprojected points' indices

        cout << "  successfully reprojected " << current.reprojectedWorldPoints.size() << " points" << endl;

        
        
        // log for reprojected 3D points
        {
            for (int i=0; i<current.reprojectedWorldPoints.size(); i++)
            {
                correspondencesRefined << std::fixed << setprecision(4)
                                       << current.reprojectedWorldPoints[i].x << ", "
                                       << current.reprojectedWorldPoints[i].y << ", "
                                       << current.reprojectedWorldPoints[i].z << ", " << std::flush;

                correspondencesRefined << std::fixed << setprecision(4)
                                       << current.reprojectedImagePoints[i].x << ", "
                                       << current.reprojectedImagePoints[i].y << "\n" << std::flush;
            }
        }

        
        
        
        // update LUT
        _3dToDescriptorTable.insert(end(_3dToDescriptorTable), begin(current._3dToDescriptor), end(current._3dToDescriptor));

        // push frame into the window
        windowedFrame.push_back(current);

        // execute the bundle adjustment for both motion (R|t) and the structures (worldPoints)
        // after the execution, all frames's motion and worldPoints within sliding window will be optimized
        //bundle.run(ref(windowedFrame));

        
        
        if (windowedFrame.size() > WINDOWSIZE)
        {
            // clear the first index elements from lookup table
            _3dToDescriptorTable.erase(_3dToDescriptorTable.begin(), _3dToDescriptorTable.begin() + windowedFrame[0].reprojectedWorldPoints.size());

            // clear the first frame inside window if the window is full
            windowedFrame.erase(windowedFrame.begin());
        }

        
        
        // now current becomes previous frame
        // TODO: still don't know how to use previous frame, verification maybe?
        prev = Frame();
        prev = current;

        
        
        // end of sequences, go to the next frameIndex
        frameIndex=frameIndex-2;
        frameCount++;

        // log
        {
           correspondences.close();
           correspondencesRefined.close();
        }
    }

    
    
    
    // log
    {
        logFile.close();
        logMatrix.close();
    }

    // end of the main caller
    return 0;
}
