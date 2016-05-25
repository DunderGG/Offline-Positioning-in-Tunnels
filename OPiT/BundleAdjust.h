#ifndef __BUNDLE_ADJUST_HEADER__
#define __BUNDLE_ADJUST_HEADER__

#include <cvsba/cvsba.h>

#include <opencv2/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>

#include <iostream>

using namespace std;
using namespace cv;

class BundleAdjust
{
public:
    struct frameInfo
    {
        vector<Point2d> imagePoints;
        vector<Point3d> worldPoints;
        Mat             K;
        Mat             R;
        Mat             t;
        Mat             distortCoef;
    };

    BundleAdjust();

    void pushFrame (vector<Point2d>, vector<Point3d>, Mat, Mat, Mat, Mat);
    void run();
    void prepareData(vector<Point3d> &, vector<vector<Point2d> > &, vector<vector<int> > &,
                     vector<Mat> &, vector<Mat> &, vector<Mat> &, vector<Mat> &);
    int  getWindowSize();
private:
    vector<frameInfo> frameWindows;
};

#endif
