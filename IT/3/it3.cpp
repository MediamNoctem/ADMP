#include <opencv2/opencv.hpp>
#include <opencv2/core/types.hpp>
#include "kcftracker.hpp"
#include <string>

using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
	bool HOG = true;
	bool FIXEDWINDOW = false;
	bool MULTISCALE = true;
	bool SILENT = true;
	bool LAB = false;

	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);

	VideoCapture video("video.mp4");

    Mat frame;

    video >> frame;

    VideoWriter writer;
    int codec = VideoWriter::fourcc('M', 'P', '4', 'V');
    double fps = 60.0;
    string filename = "./object_tracking.mp4";
    writer.open(filename, codec, fps, frame.size());

    if (!writer.isOpened()) {
        cout << "Could not open the output video file for write\n";
        return -1;
    }

    if(!video.isOpened()) {
        cout << "Could not read video file" << endl;
        return -1;
    }
 
    bool ok = video.read(frame); 
    Rect trackingBox = selectROI(frame, false); 
    rectangle(frame, trackingBox, Scalar( 0, 255, 0 ), 2, 1 ); 

    imshow("Tracking", frame); 
	tracker.init( trackingBox, frame );

	while(true)
    {     
        video >> frame;

        if (frame.empty())
            break;

        double timer = (double)getTickCount();
		Rect result = tracker.update(frame);
        bool ok = !result.empty();
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        if (ok) {
            rectangle(frame, result, Scalar( 0, 255, 0 ), 2, 1 );
        }
        else {
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
         
        putText(frame, "KCF Tracker (our)", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
        putText(frame, "FPS : " + to_string(fps), Point(100,50), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50), 2);

        writer.write(frame);
        imshow("Tracking", frame);
         
        if(waitKey(1) == 27) {
            break;
        }
    } 
    video.release();
    destroyAllWindows();
    
    return 0;
}