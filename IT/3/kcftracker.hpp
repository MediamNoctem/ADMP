#pragma once

#include "tracker.h"

#ifndef _OPENCV_KCFTRACKER_HPP_
#define _OPENCV_KCFTRACKER_HPP_
#endif

class KCFTracker : public Tracker
{
public:
    KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

    virtual void init(const cv::Rect &roi, cv::Mat image);
    
    // Обновить позицию на основе нового кадра.
    virtual cv::Rect update(cv::Mat image);

    float interp_factor; // коэффициент линейной интерполяции для адаптации
    float sigma; // пропускная способность ядра Гаусса
    float lambda; // регуляризация
    int cell_size; // размер ячейки HOG
    int cell_sizeQ; // =cell_size^2, чтобы избежать повторных операций
    float padding; // дополнительная область, окружающая цель
    float output_sigma_factor; // пропускная способность цели по Гауссу
    int template_size; // размер шаблона в пикселях, 0 для использования размера ROI
    float scale_step; // шаг масштабирования для многомасштабной оценки, 1, чтобы отключить его
    float scale_weight;  // для снижения показателей обнаружения по другим шкалам для дополнительной стабильности

protected:
    // Обнаружить объект в текущем кадре.
    cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value);

    // Обучить трекер с одним изображением.
    void train(cv::Mat x, float train_interp_factor);

    // Вычислить гауссово ядро с SIGMA пропускной способности для всех относительных сдвигов между входными изображениями X и Y, которые оба должны быть MxN. Они также должны быть периодическими (т.е. предварительно обработанными с помощью окна косинуса).
    cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);

    // Создать Гауссов пик. Функция вызывается только в первом кадре.
    cv::Mat createGaussianPeak(int sizey, int sizex);

    // Obtain sub-window from image, with replication-padding and extract features
    cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);

    // Инициализировать окно Ханнинга. Функция вызывается только в первом кадре.
    void createHanningMats();

    // Вычислить пик субпикселя для одного измерения.
    float subPixelPeak(float left, float center, float right);

    cv::Mat _alphaf;
    cv::Mat _prob;
    cv::Mat _tmpl;
    cv::Mat _num;
    cv::Mat _den;
    cv::Mat _labCentroids;

private:
    int size_patch[3];
    cv::Mat hann;
    cv::Size _tmpl_sz;
    float _scale;
    int _gaussian_size;
    bool _hogfeatures;
    bool _labfeatures;
};
