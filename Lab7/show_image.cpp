#include <iostream> 
#include <opencv2/opencv.hpp>  
      
using namespace std;
using namespace cv;  
      
int main(int argc, char** argv)  
{  
    Mat image;  
    image = imread("owl.jpg", 1);
      
    if (!image.data)
    {  
	    cout << "No image data \n";
	    return -1;
    }
    namedWindow("Image", WINDOW_AUTOSIZE);  
    imshow("Image", image);
	
	waitKey(0);
    return 0;
} 

// g++ show_image.cpp -o show_image `pkg-config --cflags --libs opencv4`
// ./show_image
