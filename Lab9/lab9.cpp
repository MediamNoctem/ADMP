#include <iostream>
#include <time.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

Point* findRedObject(Mat img) {
    int h = img.rows;
    int w = img.cols;
    int x0 = h;
    int y0 = -1;
    int x1 = 0;
    int y1 = 0; 

    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            if ((int)img.at<uchar>(y,x) == 255) {
                if (y0 < 0) y0 = y;
                if (x0 > x) x0 = x;
                if (y1 < y) y1 = y;
                if (x1 < x) x1 = x;
            }
        }

    Point* p = new Point[2];
    p[0] = Point(x0, y0);
    p[1] = Point(x1, y1);
    return p;
}

int main(int argc, char** argv) {
    VideoCapture cap(0);

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

    // Mat imgOriginal;
    // imgOriginal = imread("rose.jpg", 1);

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

        Point* p = findRedObject(imgThresholded);

        rectangle(imgOriginal, p[0], p[1], Scalar(0,1,1), -1);

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
