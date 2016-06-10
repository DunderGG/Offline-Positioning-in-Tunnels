#include <iostream>
#include <vector>
#include <fstream>
#include <chrono>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "Common.h"
#include "FeatureDetection.h"

using namespace std;
using namespace cv;

#define FRAMELEN        300
#define STARTFRAME      270
#define ENDFRAME        STARTFRAME + FRAMELEN
#define SIFT_OCTAVE     3
#define SIFT_CONTRAST   0.04f
#define SIFT_CONTRAST2  0.01f
#define SIFT_EDGE       10.0f
#define SIFT_EDGE2      20.0f
#define SIFT_SIGMA      1.6f
#define SIFT_RATIO      0.8f

const string imgPath     = "/Users/januaditya/Thesis/exjobb-data/volvo/tunnel-frames/";

int main ()
{
    FeatureDetection fdet, fdet2;
    Common com;

    fdet.setSiftParam(SIFT_OCTAVE, SIFT_CONTRAST, SIFT_EDGE, SIFT_SIGMA, SIFT_RATIO);
    fdet2.setSiftParam(SIFT_OCTAVE, SIFT_CONTRAST2, SIFT_EDGE2, SIFT_SIGMA, SIFT_RATIO);

    ofstream logFile;

    com.createDir("log-sift-features");
    logFile.open("./log-sift-features/logFeatures.txt", std::ios::out);

    int startFrame = STARTFRAME;
    int endFrame   = ENDFRAME;

    int count = 0;
    int totaldetectedkptslow = 0;
    int totaldetectedkptshigh = 0;
    char currImgPath[100];
    while (startFrame < endFrame)
    {
        cout << "crunching frame-" << startFrame << endl;
        vector<KeyPoint> detectedkpts;

        sprintf(currImgPath, "%simg_%05d.png", imgPath.c_str(), startFrame);
        Mat img = imread(currImgPath);

        Mat imgClone = img.clone();
        Mat mask = Mat::zeros(imgClone.size(), CV_8U);
        Mat roi (mask, Rect(0, 0, imgClone.cols, imgClone.rows*7/8));
        roi = Scalar(255, 255, 255);

        fdet.siftDetector(img, detectedkpts);
        logFile << detectedkpts.size() << ", " << std::flush;

        fdet.siftDetector(imgClone, detectedkpts, mask);
        logFile << detectedkpts.size() << ", " << std::flush;

        if (count < 165)
        {
            totaldetectedkptslow += detectedkpts.size();
        }


        fdet2.siftDetector(img, detectedkpts);
        logFile << detectedkpts.size() << ", " << std::flush;

        fdet2.siftDetector(imgClone, detectedkpts, mask);
        logFile << detectedkpts.size() << std::flush << endl;

        if (count < 165)
        {
            totaldetectedkptshigh += detectedkpts.size();
        }

        if (count == 165)
        {
            cout << "avg low kpts for first " << count << "frames is " << (double)(totaldetectedkptslow/count) << endl;
            cout << "avg hig kpts for first " << count << "frames is " << (double)(totaldetectedkptshigh/count) << endl;
        }

        count++;
        startFrame++;
    }

    logFile.close();

    return 0;
}
