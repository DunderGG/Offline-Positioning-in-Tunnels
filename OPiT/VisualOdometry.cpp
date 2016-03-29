#include "VisualOdometry.h"

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>

#include <iostream>

using namespace std;
using namespace cv;

#define STARTFRAME 200

/*
    A class for implementing visual odometry.

    it is based on the visual odometry tutorial by David Scaramuzza.

    this particular version will use 2d-to-2d correspondences gathered

    by performing feature detection between two aligned images.

    by doing this, essential matrix could be derived and R|t matrix

    could be decomposed. It is used to estimate the camera pose
*/

// constructor, build the camera matrix
VO::VO()											// WE HAVE TO TAKE DISTORTION INTO ACCOUNT!
{																//							    Camera matrix
	cameraMatrix = Mat(3, 3, CV_64FC1, Scalar::all(0));			//								___		  ___
	cameraMatrix.at<double>(0, 0) = 1432;						// Focal length X				| fx  0  cx |
	cameraMatrix.at<double>(1, 1) = 1432;						// Focal length Y				| 0  fy  cy |
	cameraMatrix.at<double>(0, 2) = 640;						// Principal point X			| 0   0   1 |
	cameraMatrix.at<double>(1, 2) = 481;						// Principal point Y			---		  ---
	cameraMatrix.at<double>(2, 2) = 1.0;						// Just a 1 cause why not
}

void VO::visualodometry()
{
    // start the visual odometry
    
    // 1. load the first two images
    char filename1[100];
    char filename2[100];
    
    sprintf(filename1, "%simg_%05d.png", imagePath, STARTFRAME);
    sprintf(filename2, "%simg_%05d.png", imagePath, STARTFRAME+1);
    
    Mat firstImage = imread(filename1);
    Mat secondImage = imread(filename2);
    
    // 2. work on grayscale images only
    cvtColor(firstImage, firstImage, COLOR_BGR2GRAY);
    cvtColor(secondImage, secondImage, COLOR_BGR2GRAY);
    
    // 3. feature detection on first and second images
    vector<Point2f> points1, points2;        //vectors to store the coordinates of the feature points
    featureDetection(firstImage, points1, secondImage, points2, VO_METHOD_SIFT);
    
    // 4. find camera poses alongside the track
    findPoses(secondImage, points1, points2);
}

void VO::findPoses(cv::Mat lastImage, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
{
    // find the first extrinsic matrix
    E = cv::findEssentialMat(points2, points1, cameraMatrix, method, confidence, 1.0, mask);
    recoverPose(E, points2, points1, cameraMatrix, rVec, tVec);

    Mat currImg, prevImg;
    Mat currE, currRVec, currTVec;
    
    vector <Point2f> prevPoints, currPoints;
    
    char nextimage[100];
    char text[100];
    int fontFace = FONT_HERSHEY_PLAIN;
    double fontScale = 1;
    int thickness = 1;
    cv::Point textOrg(10, 50);

    Mat matOut = Mat::zeros(960, 400, CV_64FC3);;
    
    // window
    namedWindow("VO trajectory", WINDOW_AUTOSIZE);
    
    // load the previous image
    prevImg = lastImage;
    
    for (int i=STARTFRAME+2; i<1000; i++)
    {
        // load the current image
        sprintf(nextimage, "%simg_%05d.png", imagePath, i);
        currImg = imread(nextimage);
        
        // working with grayscale only
        cvtColor(currImg, currImg, COLOR_BGR2GRAY);
        
        // feature detection
        featureDetection(prevImg, prevPoints, currImg, currPoints, VO_METHOD_SIFT);
        
        // continuous process on finding essential matrix of the image sequences
        currE = cv::findEssentialMat(currPoints, prevPoints, cameraMatrix, method, confidence, 1.0, mask);
        recoverPose(currE, currPoints, prevPoints, cameraMatrix, currRVec, currTVec);
        
        // process
        cout << "translation vector at frame-" << i << ":\n" << currTVec << endl;
        
        // update the accumulated poses
        tVec = tVec + 0.8 * rVec * currTVec;
        rVec = currRVec * rVec;
        
        // update state (for FAST, because it's using lukas-kanade tracking)
        prevImg = currImg;
        prevPoints = currPoints;
        
        // start drawing the trajectory
        int x = int(tVec.at<double>(0)) + 200;
        int y = int(tVec.at<double>(2)) + 100;
        Point2d currPose(x,y);
        
        cv::circle(matOut, currPose, 1, CV_RGB(255,0,0), -1);
        cv::rectangle(matOut, Point(10, 30), Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
        sprintf(text, "x = %.4fm, y = %.4fm, z = %.4fm", tVec.at<double>(0), tVec.at<double>(1), tVec.at<double>(2));
        cv::putText(matOut, text, textOrg, fontFace, fontScale, Scalar(0,255,0), thickness, 8);
        
        imshow("VO trajectory", matOut);
        
        waitKey(1);
    }
}

void VO::featureDetection(cv::Mat img1, std::vector<cv::Point2f>& points1, cv::Mat img2, std::vector<cv::Point2f>& points2, int vo_method)
{
    
    // fast
    if (vo_method == VO_METHOD_FAST)
    {
        fastDetection(img1, points1, img2, points2);
    }
    // sift
    else if (vo_method == VO_METHOD_SIFT)
    {
        siftDetection(img1, points1, img2, points2);
    }
    // akaze
    else if (vo_method == VO_METHOD_AKAZE)
    {
        akazeDetection(img1, points1, img2, points2);
    }
    
    drawPoints(img2, points1, points2);

}

// adapted from AVI SINGH
void VO::fastDetection(cv::Mat img1, std::vector<cv::Point2f>& points1, cv::Mat img2, std::vector<cv::Point2f>& points2)
{
    vector<KeyPoint> keypoints1, keypoints2;
    vector<uchar> status;
    
    cv::FAST(img1, keypoints1, fast_threshold, nonMaxSuppression);
    cv::KeyPoint::convert(keypoints1, points1, vector<int>());
    
    vector<float> err;
    Size winSize=Size(21,21);
    TermCriteria termcrit=TermCriteria(TermCriteria::COUNT+TermCriteria::EPS, 30, 0.01);
    
    calcOpticalFlowPyrLK(img1, img2, points1, points2, status, err, winSize, 3, termcrit, 0, 0.001);
    
    //getting rid of points for which the KLT tracking failed or those who have gone outside the frame
    int indexCorrection = 0;
    for( int i=0; i<status.size(); i++)
    {
        Point2f pt = points2.at(i- indexCorrection);
        
        if ((status.at(i) == 0)||(pt.x<0)||(pt.y<0))
        {
            if((pt.x<0)||(pt.y<0))
            {
                status.at(i) = 0;
            }
            
            points1.erase (points1.begin() + (i - indexCorrection));
            points2.erase (points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void VO::siftDetection(cv::Mat img1, std::vector<cv::Point2f>& points1, cv::Mat img2, std::vector<cv::Point2f>& points2)
{
    Ptr<Feature2D> siftDetector = xfeatures2d::SIFT::create(0, octave_layer, contrast_threshold, edge_threshold, sigma);
    
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;
    
    siftDetector->detectAndCompute(img1, noArray(), keypoints1, descriptor1);
    siftDetector->detectAndCompute(img2, noArray(), keypoints2, descriptor2);
    
    std::vector<std::vector<cv::DMatch> > matches;
    vector<KeyPoint> matched1, matched2;
    cv::BFMatcher matcher;
    
    matcher.knnMatch(descriptor1, descriptor2, matches, 2);  // Find two nearest matches
    
    for (int i = 0; i < matches.size(); ++i)
    {
        DMatch first = matches[i][0];
        float dist1 = matches[i][0].distance;
        float dist2 = matches[i][1].distance;
        
        if(dist1 < sift_matching_ratio * dist2)
        {
            matched1.push_back(keypoints1[first.queryIdx]);
            matched2.push_back(keypoints2[first.trainIdx]);
        }
    }
    
    // convert
    cv::KeyPoint::convert(matched1, points1, vector<int>());
    cv::KeyPoint::convert(matched2, points2, vector<int>());
}

void VO::surfDetection()
{

}

void VO::akazeDetection(cv::Mat img1, std::vector<cv::Point2f>& points1, cv::Mat img2, std::vector<cv::Point2f>& points2)
{
    Ptr<AKAZE> akaze = AKAZE::create();
    
    vector<KeyPoint> keypoints1, keypoints2;
    Mat descriptor1, descriptor2;
    
    akaze->detectAndCompute(img1, noArray(), keypoints1, descriptor1);
    akaze->detectAndCompute(img2, noArray(), keypoints2, descriptor2);
    
    // matching using nearest-neighbor for two keypoints
    cv::BFMatcher matcher(NORM_HAMMING);
    vector< vector <DMatch> > nn_matches;
    
    matcher.knnMatch(descriptor1, descriptor2, nn_matches, 2);
    
    // selecting the best matched
    vector<KeyPoint> matched1, matched2;
    
    // best matches
    for(size_t i = 0; i < nn_matches.size(); i++)
    {
        DMatch first = nn_matches[i][0];
        float dist1 = nn_matches[i][0].distance;
        float dist2 = nn_matches[i][1].distance;
        
        if(dist1 < nn_match_ratio * dist2)
        {
            matched1.push_back(keypoints1[first.queryIdx]);
            matched2.push_back(keypoints2[first.trainIdx]);
        }
    }
    
    // convert
    cv::KeyPoint::convert(matched1, points1, vector<int>());
    cv::KeyPoint::convert(matched2, points2, vector<int>());
}

void VO::drawPoints(cv::Mat img, std::vector<cv::Point2f>& points1, std::vector<cv::Point2f>& points2)
{
    Mat windowOutput;
    img.copyTo(windowOutput);
    
    namedWindow("VO grayscale", WINDOW_NORMAL);
    
    vector <KeyPoint> keypoints2;
    cv::KeyPoint::convert(points2, keypoints2);
    
    drawKeypoints(img, keypoints2, windowOutput, CV_RGB(255,0,0), 4);
    
    for (int i=0; i<points1.size(); i++)
    {
        cv::line(windowOutput, points2[i], points1[i], CV_RGB(255, 255, 255));
    }
    
    imshow("VO grayscale", windowOutput);
}

void VO::initParameter()
{
    iterationsCount = 100;
    reprojectionError = 8;
    confidence = 0.99;
    method = cv::RANSAC;
    
    // fast
    fast_threshold = 20;
    nonMaxSuppression = true;
    
    // surf
    min_hessian = 500;
    octave_layer = 3;
    contrast_threshold = 0.04;
    edge_threshold = 10;
    sigma = 1.6;
    
    // sift
    sift_matching_ratio = 0.8;
    
    // matching param
    inlier_threshold = 2.5f; // Distance threshold to identify inliers
    nn_match_ratio = 0.8f;   // Nearest neighbor matching ratio
}

void VO::setImagePath(char *pathname)
{
    strcpy(imagePath, pathname);
}
