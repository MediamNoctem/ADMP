#include <stdio.h>  
#include <opencv2/opencv.hpp>  
      
using namespace cv;  
      
int main(int argc, char** argv)  
{  
    Mat image;  
    image = imread("owl.jpg", 1 );
      
    if (!image.data)
    {  
	    printf("No image data \n");
	    return -1;
    }
    namedWindow("Image", WINDOW_AUTOSIZE);  
    imshow("Image", image);
	
    //if (waitKey(0) && 0xFF == 27) destroyWindow("Image");

	waitKey(0);
    return 0;
} 
