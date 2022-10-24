#include <iostream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

# define PI           3.14159265358979323846
      
using namespace cv;
using namespace std;

float sum_matrix(float** m, int h, int w)
{
    float sum = 0;
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            sum += m[i][j];
    return sum;
}

float** mult_matrix_elem_by_elem(int** m1, float** m2, int h, int w)
{
    float** m = new float*[h];
    for (int i = 0; i < h; i++)
        m[i] = new float[w];
    
    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            m[i][j] = m1[i][j] * m2[i][j];
    return m;
}

int** submatrix(int** m, int start_i, int start_j, int count)
{
    int** sbm = new int*[count];

    for (int i = 0; i < count; i++)
        sbm[i] = new int[count];

    for (int i = start_i; i < start_i + count; i++)
        for (int j = start_j; j < start_j + count; j++)
            sbm[i - start_i][j - start_j] = m[i][j];
    return sbm;
}

float** convolution_matrix(int size, float sigma, int a, int b)
{
    float** gauss = new float*[size];

    for (int i = 0; i < size; i++) {
        gauss[i] = new float[size];
        for (int j = 0; j < size; j++) {
            gauss[i][j] = 0;
        }
    }

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++) {
            gauss[i][j] = (1 / (2 * PI * pow(sigma, 2))) * exp(-((pow((i - a), 2) + pow((j - b), 2)) / (2 * pow(sigma, 2))));
        }

    float sum = sum_matrix(gauss, size, size);

    for (int i = 0; i < size; i++)
        for (int j = 0; j < size; j++)
            gauss[i][j] /= sum;

    return gauss;
}

int** mult_for_gauss_blur(int** img, int height, int width, float** conv_matrix, int size)
{
    int margin = size / 2;

    for (int i = 0; i < height - margin * 2; i++)
        for (int j = 0; j < width - margin * 2; j++)
            img[margin + i][margin + j] = round(sum_matrix(mult_matrix_elem_by_elem(submatrix(img, i, j, size), conv_matrix, size, size), size, size));
    return img;
}

void gaussian_blur(int size_conv_matrix, float sigma, int a, int b, string path)
{
    float** conv_matrix = convolution_matrix(size_conv_matrix, sigma, a, b);

    Mat gray_image_mat = imread(path, CV_8UC1);
    int h = gray_image_mat.rows;
    int w = gray_image_mat.cols;
    int** gray_image = new int*[h];

    for (int i = 0; i < h; i++)
        gray_image[i] = new int[w];

    for (int i = 0; i < h; i++)
        for (int j = 0; j < w; j++)
            gray_image[i][j] = (int)gray_image_mat.at<uchar>(i,j);

    gray_image = mult_for_gauss_blur(gray_image, h, w, conv_matrix, size_conv_matrix);
    namedWindow("Display window", WINDOW_AUTOSIZE);

    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++)
            gray_image_mat.at<uchar>(i,j) = gray_image[i][j];
    }

    imshow("Display window", gray_image_mat);
    waitKey(0);
    destroyAllWindows();
}

int main(int argc, char** argv)
{
    int conv_matrix_size = 21;
    float sigma = 0.3 * ((conv_matrix_size - 1) * 0.5 - 1) + 0.8;
    gaussian_blur(conv_matrix_size, sigma, (conv_matrix_size - 1) / 2, (conv_matrix_size - 1) / 2, "husky.jpg");
    return 0;
}

// g++ lab2_4.cpp -o lab2_4 `pkg-config --cflags --libs opencv4`
// ./lab2_4
