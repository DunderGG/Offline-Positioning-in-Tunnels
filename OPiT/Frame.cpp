#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "Frame.h"

using namespace std;
using namespace cv;

Frame::Frame()
{
    // do nothing
}

Frame::~Frame()
{
    // destruct nothing
}

void Frame::projectWorldtoCamera()
{
    this->c_matchedWorldPoints.clear();
    for (int i=0; i<this->matchedWorldPoints.size(); i++)
    {
        Mat w_pointTemp = (Mat1d(3,1) << this->matchedWorldPoints[i].x,
                                         this->matchedWorldPoints[i].y,
                                         this->matchedWorldPoints[i].z);
        Mat c_pointTemp = this->R * w_pointTemp + this->t;
        this->c_matchedWorldPoints.push_back (Point3d(c_pointTemp));
    }
}

void Frame::projectCameratoWorld()
{
    this->matchedWorldPoints.clear();
    for (int i=0; i<this->c_matchedWorldPoints.size(); i++)
    {
        Mat c_pointTemp = (Mat1d(3,1) << this->c_matchedWorldPoints[i].x,
                                         this->c_matchedWorldPoints[i].y,
                                         this->c_matchedWorldPoints[i].z);
        Mat w_pointTemp = this->R.inv() * c_pointTemp + ((-(this->R.inv()) * this->t));

        this->matchedWorldPoints.push_back (Point3d(w_pointTemp));
    }
}
