#ifndef __FRAME_H_INCLUDED_
#define __FRAME_H_INCLUDED_

#include <iostream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>

#include "Common.h"

using namespace std;
using namespace cv;

class Frame
{
public:
    // all informations regarding frame data
    int                         frameIdx;                   // frame index
    Mat                         image;                      // image file
    Mat                         descriptors;                // size Nx128,  all descriptors for keypoints
    vector <KeyPoint>           keypoints;                  // size N,      all keypoints from SIFT detector

    vector<vector<DMatch> >     matches;                    // size M,      all matches from feature matching

    vector <Point3d>            matchedWorldPoints;         // size M',     all 3d points from lookup table matching
    vector <Point3d>            c_matchedWorldPoints;       // size K,      all 3d points reprojected points (camera coordinate)
    vector <Point2d>            matchedImagePoints;         // size M',     all 2d points from the matching

    vector <Point3d>            reprojectedWorldPoints;     // size K,      all 3d points reprojected from all keypoints (world coordinate)
    vector <Point3d>            c_reprojectedWorldPoints;   // size K,      all 3d points reprojected points (camera coordinate)
    vector <Point2d>            reprojectedImagePoints;     // size K,      all 2d points that successfully backprojected
    vector <int>                reprojectedIndices;         // size K,      indices for allcorresponding 3d/2d backprojected points

    vector<pair<Point3d, Mat> > _3dToDescriptor;            // size K,      all backprojected 3d points + descriptors

    Mat                         K;                          // 3x3,         intrinsic matrix
    Mat                         R;                          // 3x3,         rotation matrix
    Mat                         R_rodrigues;                // 1x3,         rotation vector (rodrigues)
    Mat                         t;                          // 1x3,         translation matrix
    Mat                         R_invert;                   // 3x3,         this is the transposed of regular R
    Mat                         t_invert;                   // 1x3,         this is the camera position in the world space
    Mat                         t_translation;              // 1x3,         represent the translation from previous camera pose
    Mat                         distCoef;                   // 1x5,         distortion coefficients
    Mat                         cameraPose;                 // 3x4,         inverse matrices

// method
    Frame();
    ~Frame();

    void                        projectWorldtoCamera();     // project world space 3d points into camera space
    void                        projectCameratoWorld();     // project camera space 3d points into world
    
private:
};

#endif
