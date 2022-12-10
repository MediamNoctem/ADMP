#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace cv;
using namespace std;

int round_angle(float sum_gx, float sum_gy)
{
	float tangens = tan(sum_gy / sum_gx);
	int fi = -1;

	if (((sum_gx > 0) && (sum_gy < 0) && (tangens < -2.414)) || ((sum_gx < 0) && (sum_gy < 0) && (tangens > 2.414)))
        fi = 0;
    else
    	if ((sum_gx > 0) && (sum_gy < 0) && (tangens < -0.414))
        	fi = 1;
	    else
	    	if (((sum_gx > 0) && (sum_gy < 0) && (tangens > -0.414)) || ((sum_gx > 0) && (sum_gy > 0) && (tangens < 0.414)))
	        	fi = 2;
		    else
		    	if ((sum_gx > 0) && (sum_gy > 0) && (tangens < 2.414))
		        	fi = 3;
			    else
			    	if (((sum_gx > 0) && (sum_gy > 0) && (tangens > 2.414)) || ((sum_gx < 0) && (sum_gy > 0) && (tangens < -2.414)))
			        	fi = 4;
				    else
				    	if ((sum_gx < 0) && (sum_gy > 0) && (tangens < -0.414))
				        	fi = 5;
					    else
					    	if (((sum_gx < 0) && (sum_gy > 0) && (tangens > -0.414)) || ((sum_gx < 0) && (sum_gy < 0) && (tangens < 0.414)))
					        	fi = 6;
						    else
						    	if ((sum_gx < 0) && (sum_gy < 0) && (tangens < 2.414))
						        	fi = 7;
    return fi;
}

int** non_maximum_suppression(int** grad_lenght, int** grad_angle, int** img, int h, int w)
{
	int** img2 = new int*[h];

    for (int i = 0; i < h; i++)
        img2[i] = new int[w];

    for (int i = 0; i < h; i++)
    	for (int j = 0; j < w; j++)
    		img2[i][j] = img[i][j];

	for (int i = 3; i < h - 3; i++)
		for (int j = 3; j < w - 3; j++) {
			img2[i][j] = 0;
			if ((grad_angle[i][j] == 6) || (grad_angle[i][j] == 2))
				if ((grad_lenght[i][j] > grad_lenght[i][j - 1]) && (grad_lenght[i][j] > grad_lenght[i][j + 1]))
					img2[i][j] = 255;
				else
					img2[i][j] = 0;
			else
				if ((grad_angle[i][j] == 4) || (grad_angle[i][j] == 0))
					if ((grad_lenght[i][j] > grad_lenght[i - 1][j]) && (grad_lenght[i][j] > grad_lenght[i + 1][j]))
						img2[i][j] = 255;
					else
						img2[i][j] = 0;
				else
					if ((grad_angle[i][j] == 5) || (grad_angle[i][j] == 1))
						if ((grad_lenght[i][j] > grad_lenght[i + 1][j - 1]) && (grad_lenght[i][j] > grad_lenght[i - 1][j + 1]))
							img2[i][j] = 255;
						else
							img2[i][j] = 0;
					else
						if ((grad_angle[i][j] == 7) || (grad_angle[i][j] == 2))
							if ((grad_lenght[i][j] > grad_lenght[i - 1][j - 1]) && (grad_lenght[i][j] > grad_lenght[i + 1][j + 1]))
								img2[i][j] = 255;
							else
								img2[i][j] = 0;
		}
	return img2;
}

int** double_filtration(int max_grad_lenght, int** grad_lenght, int** img, int h, int w)
{
	int** img2 = new int*[h];

    for (int i = 0; i < h; i++)
        img2[i] = new int[w];

    for (int i = 0; i < h; i++)
    	for (int j = 0; j < w; j++)
    		img2[i][j] = img[i][j];

	int low_level = max_grad_lenght / 5;
	int high_level = max_grad_lenght / 2;

	for (int i = 4; i < h - 4; i++)
		for (int j = 4; j < w - 4; j++)
			if (img2[i][j] == 255)
				if (grad_lenght[i][j] < low_level)
					img2[i][j] = 0;
				else
					if (grad_lenght[i][j] < high_level)
					{
						img2[i][j] = 0;
						for (int k = 0; k < 8; k++)
							for (int g = 0; g < 8; g++)
								if ((k != 4) && (g != 4) && (img2[i - 4 + k][j - 4 + g] == 255))
									img2[i][j] = 255;
					}
	return img2;
}

void method_Canny(string path)
{
	Mat gray_image_mat = imread(path, CV_8UC1);	

    GaussianBlur(gray_image_mat, gray_image_mat, Size(7,7), 11, 11);

	int h = gray_image_mat.rows;
    int w = gray_image_mat.cols;
    float fi;
    int** gray_image = new int*[h];
    int** grad_lenght = new int*[h];
    int** grad_angle = new int*[h];

    for (int i = 0; i < h; i++) {
        gray_image[i] = new int[w];
        grad_lenght[i] = new int[w];
        grad_angle[i] = new int[w];
    }

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++) {
            gray_image[i][j] = (int)gray_image_mat.at<uchar>(i,j);
            grad_lenght[i][j] =  0;
            grad_angle[i][j] =  0;
        }

    int** gx = new int*[3];
    int** gy = new int*[3];

    for (int i = 0; i < 3; i++) {
    	gx[i] = new int[3];
    	gy[i] = new int[3];
    }

    gx[0][0] = -1;
    gx[0][1] = 0;
    gx[0][2] = 1;

    gx[1][0] = -2;
    gx[1][1] = 0;
    gx[1][2] = 2;

    gx[2][0] = -1;
    gx[2][1] = 0;
    gx[2][2] = 1;

    gy[0][0] = -1;
    gy[0][1] = -2;
    gy[0][2] = -1;

    gy[1][0] = 0;
    gy[1][1] = 0;
    gy[1][2] = 0;

    gy[2][0] = 1;
    gy[2][1] = 2;
    gy[2][2] = 1;

    int len_gx = 3;
    float max_grad_lenght = 0;

    for (int i = len_gx / 2; i < h - len_gx % 2; i++)
    	for (int j = len_gx / 2; j < w - len_gx % 2; j++) {
    		float sum_gx = 0, sum_gy = 0;

    		for (int k = 0; k < len_gx; k++)
    			for (int g = 0; g < len_gx; g++) {
    				sum_gx += gx[k][g] * gray_image[k - len_gx / 2 + i][g - len_gx / 2 + j];
                    sum_gy += gy[k][g] * gray_image[k - len_gx / 2 + i][g - len_gx / 2 + j];
    			}
    		float grad_lenght_temp = sqrt(sum_gx * sum_gx + sum_gy * sum_gy);
    		if (sum_gx == 0.0)
                sum_gx = 0.0000001;
            fi = round_angle(sum_gx, sum_gy);
            grad_lenght[i][j] = grad_lenght_temp;
            grad_angle[i][j] = fi;
            max_grad_lenght = max(max_grad_lenght, grad_lenght_temp);
    	}

    gray_image = non_maximum_suppression(grad_lenght, grad_angle, gray_image, h, w);
    gray_image = double_filtration(max_grad_lenght, grad_lenght, gray_image, h, w);

    namedWindow("Display window", WINDOW_AUTOSIZE);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++)
            gray_image_mat.at<uchar>(i,j) = (uchar)gray_image[i][j];
    }

    imshow("Display window", gray_image_mat);
    waitKey(0);
    destroyAllWindows();
}

int main(int argc, char** argv)
{
	method_Canny("husky.jpg");
    return 0;
}