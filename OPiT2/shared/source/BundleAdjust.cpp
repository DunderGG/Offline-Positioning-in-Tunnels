#include "BundleAdjust.h"

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <cvsba/cvsba.h>

#include <iostream>

using namespace std;
using namespace cv;

BundleAdjust::BundleAdjust()
{
    this->frameWindows.clear();
}

void BundleAdjust::pushFrame(vector<Point2d> image2D, vector<Point3d> image3D, Mat K, Mat Rvec, Mat tvec, Mat dist)
{
    frameInfo temp;
    temp.imagePoints    = image2D;
    temp.worldPoints    = image3D;
    temp.K              = K;
    temp.R              = Rvec;
    temp.t              = tvec;
    temp.distortCoef    = dist;

    this->frameWindows.push_back (temp);
}

int BundleAdjust::getWindowSize()
{
    return BundleAdjust::frameWindows.size();
}

void BundleAdjust::prepareData(vector<Point3d> &points3D,
                               vector<vector<Point2d> > &pointsImg,
                               vector<vector<int> > &visibility,
                               vector<Mat> &cameraMatrix,
                               vector<Mat> &R,
                               vector<Mat> &T,
                               vector<Mat> &distCoeffs)
{
    // get the nframes
    int nframes = frameWindows.size();
    
    // visibility
    visibility.resize(nframes);
    for (int i=0; i<visibility.size(); i++)
        visibility[i].resize(points3D.size());
    
    //
}

void BundleAdjust::run()
{
    cvsba::Sba sba;

    cvsba::Sba::Params params;
    params.type = cvsba::Sba::MOTIONSTRUCTURE;
    params.iterations = 150;
    params.minError = 1e-10;
    params.fixedIntrinsics = 5;
    params.fixedDistortion = 5;
    params.verbose = true;
    sba.setParams(params);
    
    vector<Point3d> points3D;
    vector<vector<cv::Point2d> > pointsImg;
    vector<vector<int> > visibility;
    vector<Mat> cameraMatrix, distCoeffs, R, T;
    
    Mat rTemp;
    Mat distCoef = (Mat1d(1,5) << 0, 0, 0, 0, 0);
    
    prepareData(ref(points3D), ref(pointsImg), ref(visibility), ref(cameraMatrix), ref(R), ref(T), ref(distCoeffs));

    double repError = sba.run(points3D, pointsImg, visibility, cameraMatrix, R, T, distCoeffs);

    cout << "  reprojection error after bundle adjustment: " << repError << endl;
}
