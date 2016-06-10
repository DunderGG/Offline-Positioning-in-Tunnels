
#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include "FeatureDetection.h"
#include "Calibration.h"
#include "PnPSolver.h"
#include "PointProjection.h"
#include "PCLCloudSearch.h"
#include "Reprojection.h"
#include "BundleAdjust.h"
#include "Common.h"
#include <iostream>
#include <fstream>
#include <numeric>
#include <chrono>
#include <mutex>
#define STARTIDX          433
#define FINISHIDX         523
#define WINDOWSIZE        10
#define NUMTHREADS        8
#define STATICNTASK       480
#define DRAWKPTS          1
#define SEQMODE           0
using namespace std;
using namespace cv;
using namespace pcl;
using namespace std::chrono;

/* -------------------------------------themaincaller---------------------------------------*/
int MainWrapper (int argc, char *argv[])
{
    vector<pair<Point3d, Mat> >     lookuptable;
    vector<Point3d>                 tunnel3D;
    vector<Point2d>                 tunnel2D;
    vector<int>		                tunnel1D;
    Mat                             tunnelDescriptor;

    Calibration cal;
    PnPSolver pnp(1000, 5, 0.99);
    PnPSolver pnprefined(1000, 5, 0.99);
    FeatureDetection feat;
    BundleAdjust ba;
    Common com;

    int startFrame, lastFrame;
    ofstream logFile, logMatrix, correspondences, correspondencesRefined;
    if (argc == 3) {startFrame = atoi(argv[1]);lastFrame  = atoi(argv[2]);}
    else {startFrame = STARTIDX;lastFrame = FINISHIDX;}

    char poseFileIdx[100]; char poseRefinedFileIdx[100];
    if (LOGMODE)
    {
        com.createDir("log");

        // create a file for logging posiions
        logFile.open ("./log/logPoses.txt", std::ios::out);
        logMatrix.open ("./log/logMatrix.txt", std::ios::out);
    }

    char map2Dto3D  [100]; char mapDescrip [100];
    sprintf(map2Dto3D, "/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT/ManualCorrespondences.txt");
    sprintf(mapDescrip,"/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT/ManualCorrespondences.yml");
    com.prepareMap(map2Dto3D, mapDescrip, ref(tunnel2D), ref(tunnel3D), ref(tunnelDescriptor));
    com.updatelut(tunnel3D, tunnelDescriptor, ref(lookuptable));
    PointCloud<PointXYZ>::Ptr cloud(new PointCloud<PointXYZ>);
    io::loadPCDFile("/Users/januaditya/Thesis/exjobb-data/git/Offline-Positioning-in-Tunnels/OPiT2/cloud/gnistangtunneln-semifull-voxelized.pcd", *cloud);
    std::cerr << "cloud: " << cloud->width * cloud->height << " points (" << pcl::getFieldsList (*cloud) << ")" << std::endl;
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud(cloud);
    char pathname[100] = "/Users/januaditya/Thesis/exjobb-data/volvo/out0/";
    char nextimage[100];
    Mat T, K;
    int frameIdx = startFrame;
    int frameCounter = 0;

    while (frameIdx < lastFrame)
    {
        if (LOGMODE)
        {
            sprintf(poseFileIdx, "./log/%d-poseFile.txt", frameIdx);
            sprintf(poseRefinedFileIdx, "./log/%d-poseRefined.txt", frameIdx);

            correspondences.open(poseFileIdx, std::ios::out);
            correspondencesRefined.open(poseRefinedFileIdx, std::ios::out);
        }

        com.startTimer();
        if (frameCounter > 0) pnp.setPnPParam(1000, 5, 0.99);
        sprintf(nextimage, "%simg_%05d.png", pathname, frameIdx);
        cout << "processing frame-" << frameIdx << endl;

        // load the image
        Mat img = imread(nextimage);
        vector<Point2d> retrieved2D; vector<Point3d> retrieved3D; vector<KeyPoint> detectedkpts; Mat descriptor;

        // get all the descriptors from the lookuptable
        Mat desc = com.getdescriptor(lookuptable);

        // perform the roi, feature matching and returns data from the lookuptable
        feat.computeFeaturesAndMatching(img,tunnel2D,tunnel3D,desc,frameCounter,&detectedkpts,&descriptor,&retrieved2D,&retrieved3D);

        // solve the camera pose and returns R, K, t
        pnp.setImagePoints(retrieved2D); pnp.setWorldPoints(retrieved3D); pnp.run(1);
        T = pnp.getCameraPose(); K = cal.getCameraMatrix();
        Mat R = pnp.getRotationMatrix(); Mat t = pnp.getTranslationVector();

        // save the initial camera pose for frame-idx
        if (LOGMODE)
        {
            // save the current frame retrieved pair 2D and 3D
            for (int i=0; i<retrieved2D.size(); i++)
            {
                correspondences << std::fixed << setprecision(4)
                << retrieved3D[i].x << ", "
                << retrieved3D[i].y << ", "
                << retrieved3D[i].z << ", " << std::flush;

                correspondences << std::fixed << setprecision(4)
                << retrieved2D[i].x << ", "
                << retrieved2D[i].y << "\n" << std::flush;
            }

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

        // TODO: a line that changes everything | clear the lut
        tunnel1D.clear(); tunnel2D.clear(); tunnel3D.clear();tunnelDescriptor.release();lookuptable.clear();

        // multithread backprojection
        com.threading(NUMTHREADS, T, K, detectedkpts, descriptor, std::ref(cloud), std::ref(kdtree), std::ref(lookuptable), std::ref(tunnel3D), std::ref(tunnel2D), std::ref(tunnel1D));

        // the size of lookuptable, tunnel3D, tunnel2D and tunnel1D are same

        // refine the camera pose
        pnprefined.setImagePoints(tunnel2D); pnprefined.setWorldPoints(tunnel3D); pnprefined.run(1);

        // test for the reprojection error
        vector<Point2d> reprojectedPixels;
        projectPoints(tunnel3D,pnp.getRotationMatrix(),pnp.getTranslationVector(),cal.getCameraMatrix(),cal.getDistortionCoeffs(),reprojectedPixels);
        double repError = 0;
        for (int itx = 0; itx < tunnel1D.size(); itx++)
        {
            double dx, dy;
            dx = pow(abs(reprojectedPixels[itx].x - detectedkpts[tunnel1D[itx]].pt.x), 2);
            dy = pow(abs(reprojectedPixels[itx].y - detectedkpts[tunnel1D[itx]].pt.y), 2);
            repError += sqrt(dx + dy);
        }
        cout << "  lut size is " << tunnel3D.size() << endl;
        cout << "  avg reprojection error for " << tunnel1D.size() << " points is: " << repError/tunnel1D.size() << " px " << endl;

        // save the refined initial camera pose for frame-idx
        if (LOGMODE)
        {
            // save the 3D-to-2D correspondences
            for (int i=0; i<tunnel2D.size(); i++)
            {
                correspondencesRefined << std::fixed << setprecision(4)
                                       << tunnel3D[i].x << ", "
                                       << tunnel3D[i].y << ", "
                                       << tunnel3D[i].z << ", " << std::flush;

                correspondencesRefined << std::fixed << setprecision(4)
                                       << tunnel2D[i].x << ", "
                                       << tunnel2D[i].y << "\n" << std::flush;
            }

            // save the refined vehicle position, num of keypoints, num of backprojected points and reprojection error
            T = pnprefined.getCameraPose().clone();

            logFile << std::fixed << setprecision(10)
                    << T.at<double>(0,3) << ", "
                    << T.at<double>(1,3) << ", "
                    << T.at<double>(2,3) << ", " << std::flush;

            logFile << std::fixed << setprecision(10)
                    << detectedkpts.size()       << ", "
                    << tunnel3D.size()           << ", "
                    << repError/tunnel1D.size()  << "\n" << std::flush;      // end of logPose

            // save the Rotation and Translation matrices
            R = pnprefined.getRotationMatrix();
            t = pnprefined.getTranslationVector();

            logMatrix << std::fixed << setprecision(10)
                      << R.at<double>(0,0) << ", " << R.at<double>(0,1) << ", " << R.at<double>(0,2) << ", "
                      << R.at<double>(1,0) << ", " << R.at<double>(1,1) << ", " << R.at<double>(1,2) << ", "
                      << R.at<double>(2,0) << ", " << R.at<double>(2,1) << ", " << R.at<double>(2,2) << ", "
                      // end of logMatrix
                      << t.at<double>(0)   << ", " << t.at<double>(1)   << ", " << t.at<double>(2)   << "\n" << std::flush;
        }

        // TODO: a line that impacts the thesis enormously
        if (ba.getWindowSize() < WINDOWSIZE)
            ba.pushFrame(tunnel2D, tunnel3D, K, pnp.getRotationMatrix(), pnp.getTranslationVector(), cal.getDistortionCoeffs());
        if (ba.getWindowSize() == WINDOWSIZE)
        {
            ba.run();
            ba.eraseFirstFrame();
        }


        // close the file
        if (LOGMODE)
        {
           correspondences.close();
           correspondencesRefined.close();
        }

        com.reportTimer();
        frameIdx++; frameCounter++;
    }

    // close the i/o
    logFile.close();
    logMatrix.close();
    return 0;
}
