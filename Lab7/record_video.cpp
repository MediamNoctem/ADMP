#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
      
using namespace cv;
using namespace std;

int main(int argc, char** argv)
{
    Mat src;
    VideoCapture cap("http://192.168.148.36:8080/video/mjpeg");
    if (!cap.isOpened()) {
        cout << "Error! Unable to open camera\n";
        return -1;
    }

    cap >> src;

    VideoWriter writer;
    int codec = VideoWriter::fourcc('M', 'J', 'P', 'G');
    double fps = 25.0;
    string filename = "./live.avi";
    writer.open(filename, codec, fps, src.size());

    if (!writer.isOpened()) {
        cout << "Could not open the output video file for write\n";
        return -1;
    }

    for (;;)
    {
        if (!cap.read(src)) {
            cout << "Error! Blank frame grabbed\n";
            break;
        }
        writer.write(src);
        imshow("Live", src);
        if (waitKey(5) >= 0)
            break;
    }
    return 0;
}

// g++ record_video.cpp -o record_video `pkg-config --cflags --libs opencv4`
// ./record_video
