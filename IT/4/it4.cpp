#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/tracking/tracking_legacy.hpp>
#include <opencv2/core/types.hpp>
#include <string>
 
using namespace cv;
using namespace std;

int main(int argc, char **argv)
{
    // List of tracker types in OpenCV 3.4.1
    // "GOTURN"
    string trackerTypes[7] = {"BOOSTING", "MIL", "KCF", "TLD","MEDIANFLOW", "MOSSE", "CSRT"};
    string trackerType = trackerTypes[6];
    Ptr<Tracker> tracker;
 
    if (trackerType == "BOOSTING")
        tracker = legacy::upgradeTrackingAPI(legacy::TrackerBoosting::create());
    if (trackerType == "MIL")
        tracker = TrackerMIL::create();
    if (trackerType == "KCF")
        tracker = TrackerKCF::create();
    if (trackerType == "TLD")
        tracker = legacy::upgradeTrackingAPI(legacy::TrackerTLD::create());
    if (trackerType == "MEDIANFLOW")
        tracker = legacy::upgradeTrackingAPI(legacy::TrackerMedianFlow::create());
    if (trackerType == "MOSSE")
        tracker = legacy::upgradeTrackingAPI(legacy::TrackerMOSSE::create());
    if (trackerType == "CSRT")
        tracker = TrackerCSRT::create();
    /*if (trackerType == "GOTURN")
        tracker = TrackerGOTURN::create();*/
    
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
    tracker->init(frame, trackingBox);
     
    while(true)
    {     
        video >> frame;

        if (frame.empty())
            break;

        double timer = (double)getTickCount();
        bool ok = tracker->update(frame, trackingBox);
        float fps = getTickFrequency() / ((double)getTickCount() - timer);
         
        if (ok) {
            rectangle(frame, trackingBox, Scalar( 0, 255, 0 ), 2, 1 );
        }
        else {
            putText(frame, "Tracking failure detected", Point(100,80), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0,0,255),2);
        }
         
        putText(frame, trackerType + " Tracker", Point(100,20), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(50,170,50),2);
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
