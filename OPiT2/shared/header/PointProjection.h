#ifndef __POINTPROJECTION_H_INCLUDED__   // if x.h hasn't been included yet...
#define __POINTPROJECTION_H_INCLUDED__
#include <opencv2/features2d.hpp>
#include <initializer_list>
#include <iostream>
class PointProjection
{
public:
	void foo(std::vector<cv::Point3f> worldPoints, cv::Mat rvec, cv::Mat tvec, cv::Mat cameraMatrix, cv::Mat distCoeffs);
	std::vector<cv::Point2f> getProjectedImagePoints();
	cv::Mat getProjectionJacobian();
private:
	std::vector<cv::Point2f> projectionImagePoints;
	cv::Mat projectionJacobian;
};

#endif
