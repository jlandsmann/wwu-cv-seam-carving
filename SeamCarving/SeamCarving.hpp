#pragma once
#include <iostream>
#include <filesystem>
#include <fstream>
#include <regex>
#include "opencv2/opencv.hpp"
using namespace cv;
using namespace std;

Mat seamCarving(Mat img, double ratioCols = 1., double ratioRows = 1.);
Mat seamCarving(Mat img,int newCols=0,int newRows=0);
Mat carve(Mat img, int newCols);
vector<int> findSeam(Mat& energy);
float energy(Mat& img, int x, int y);
Mat energyImage(Mat& img);
void energyImage(Mat& img, Mat& oldEnergyImg,vector<int>& seam);