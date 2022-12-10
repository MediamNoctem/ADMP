#include <iostream>
#include <opencv2/opencv.hpp>
      
using namespace cv;
using namespace std;

Scalar HSVColorDetection(uchar h)
{
    Scalar pixel;
    if (h >= 0 && h <= 30 || h > 150 && h <= 180)
        pixel = Scalar(0,0,255);
    else 
        if (h > 30 && h <= 90)
            pixel = Scalar(0,255,0);
        else 
            if (h > 90 && h <= 150)
                pixel = Scalar(255,0,0);
    return pixel;
}

int main(int argc, char** argv)
{
    Mat image, output_image; 
    Point p1, p2;
    Scalar color;
    int x, y, width = 6;

    image = imread("raspberry.jpg", 1);
    cvtColor(image, output_image, COLOR_BGR2HSV);

    y = image.rows/2; 
    x = image.cols/2;

    p1 = Point(x - width, y - width * 4);
    p2 = Point(x + width, y + width * 4);

    color = HSVColorDetection(output_image.at<Vec3b>(y,x)[0]);

    rectangle(image, p1, p2, color, -1);

    p1 = Point(x - width * 4, y - width);
    p2 = Point(x + width * 4, y + width);

    rectangle(image, p1, p2, color, -1);

    namedWindow("Image", WINDOW_AUTOSIZE);  
    namedWindow("Image_HSV", WINDOW_AUTOSIZE);  
    
    imshow("Image", image);
    imshow("Image_HSV", output_image);
	
    waitKey(0);
    return 0;
}
