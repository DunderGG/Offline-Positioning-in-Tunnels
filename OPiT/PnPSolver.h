#ifndef __PNPSOLVER_H_INCLUDED__   // if x.h hasn't been included yet...
#define __PNPSOLVER_H_INCLUDED__

#include <opencv2/features2d.hpp>
#include <initializer_list>
#include <iostream>

class PnPSolver
{
public:
	PnPSolver();
	PnPSolver(cv::Mat CM);

	void setWorldPoints();
	void setWorldPoints(std::vector<cv::Point3f> WP);
	std::vector<cv::Point3f> getWorldPoints();

	void setImagePoints();
	void setImagePoints(std::vector<cv::Point2f> IP);
	std::vector<cv::Point2f> getImagePoints();

	cv::Mat getCameraMatrix();
	cv::Mat getCameraPose();
	cv::Mat getCameraPosition();

	cv::Mat getRotationVector();
	cv::Mat getTranslationVector();

	cv::Mat getRotationMatrix();
	cv::Mat getTranslationMatrix();

	int foo(int verbal, cv::Mat distCoeffs);

	cv::Mat cameraPose34;

private:
	cv::Mat cameraMatrix;

	std::vector<cv::Point2f> imagePoints, imagePoints2;
	std::vector<cv::Point3f> worldPoints;


	cv::Mat rVec, tVec;
	cv::Mat rMat, tMat;
	cv::Mat cameraPose, cameraPosition;

};

#endif
