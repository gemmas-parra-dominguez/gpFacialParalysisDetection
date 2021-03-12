#ifndef GP1TOBE_MAIN_H_INCLUDED
#define GP1TOBE_MAIN_H_INCLUDED

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/opencv_modules.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <math.h>
#include "gpmisc.h"

#define FIRST_LANDMARK 17
#define TOTAL_LANDMARK 51
#define TOTAL_SMR 29

const string test_results_path("test_results/");
const string test_image_path("test_data/");
const string face_detector("haarcascade_frontalface_alt2.xml");
const string image_ext(".jpg");
const string csv_name(".csv");
/**/
void gpComputeSRM(int num_test_images, string test_image_root, int init_img);
Mat gpFaceReg(Mat &image_face, Mat &data_keypts, Mat &mat_landmarks);
float gpEuclideanDist (float x_land, float y_land, float x_key, float y_key);
void gpPtsExt(Mat &mat_landmarks, Mat &coord_vector);
void gpGetSMR(Mat &norm_vector, Mat &face_smr_data);
void gpGetFMM(Mat &norm_vector, Mat &face_smr_data);

#endif // GP1TOBE_MAIN_H_INCLUDED
