#include <iostream>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    VideoCapture cap("http://192.168.243.173:8080/video/mjpeg");

    if (!cap.isOpened())
    {
       cout << "Cannot open the web cam" << endl;
       return -1;
    }

    int iLowH = 170;
    int iHighH = 179;

    int iLowS = 150; 
    int iHighS = 255;

    int iLowV = 60;
    int iHighV = 255;

    int iLastX = -1; 
    int iLastY = -1;

    Mat imgTmp;
    cap.read(imgTmp); 

    time_t start,end;
    time (&start);

    int frames = 0;

    while (true) {
        Mat imgOriginal;
        bool bSuccess = cap.read(imgOriginal); 
        if (!bSuccess) {      
            cout << "Cannot read a frame from video stream" << endl;
            break;
        }
        Mat imgHSV;

        cvtColor(imgOriginal, imgHSV, COLOR_BGR2HSV);

        Mat imgThresholded;
        inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded);

        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours(imgThresholded, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);

        //drawContours(imgThresholded, contours[0], -1, (255, 0, 255), 3);

        Moments oMoments = moments(imgThresholded);
        double dM01 = oMoments.m01;
        double dM10 = oMoments.m10;
        double dArea = oMoments.m00;

        if (dArea > 10000) {
            int posX = dM10 / dArea;
            int posY = dM01 / dArea; 

            if (iLastX >= 0 && iLastY >= 0 && posX >= 0 && posY >= 0)
                rectangle(imgOriginal, Point(posX, posY), Point(iLastX, iLastY), Scalar(0,1,1), -1);
            
            iLastX = posX;
            iLastY = posY;
        }
        imshow("Thresholded Image", imgThresholded);
        imshow("Original", imgOriginal);
        if (waitKey(30) == 27) {
            cout << "esc key is pressed by user" << endl;
            break; 
        }
        frames++;
    }

    time (&end);
    double dif = difftime (end,start);
    printf("FPS %.2lf seconds.\r\n", (frames / dif));

    return 0;
}
