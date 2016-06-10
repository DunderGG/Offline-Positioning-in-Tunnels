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

#include "Frame.h"

#include <iostream>

using namespace std;
using namespace cv;

class BundleAdjust
{
public:
    BundleAdjust();
    ~BundleAdjust();

    void run(vector<Frame> &);
    void prepareData(vector<Point3d> &, vector<vector<Point2d> > &, vector<vector<int> > &,
                     vector<Mat> &, vector<Mat> &, vector<Mat> &, vector<Mat> &);
    void updateData(vector<Point3d>, vector<Mat>, vector<Mat>, vector<Frame> &);
    int  getWindowSize();
private:
    cvsba::Sba          sba;
    cvsba::Sba::Params  params;
    vector<Frame>       trackedFrame;
};

#endif
