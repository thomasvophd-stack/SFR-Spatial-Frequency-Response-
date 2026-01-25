// App to calculate SFR from images. Based off SFRMat...but free and open source.
// Thomas Vo

#include <iostream>
#include <fstream>
#include <string.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <math.h>
#include <typeinfo>
#include <vector>
#include <algorithm>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <filesystem>

#include <sys/stat.h>
#include <unistd.h>

#include "../include/sfr_utility.h"
#include "../include/sfr_utility_polyfit.h"
#include "json/json.h"

using namespace std;

inline bool exists_test (const std::string& name) {
  struct stat buffer;   
  return (stat (name.c_str(), &buffer) == 0); 
}

void print(std::vector<double> const &input)
{
    for (int i = 0; i < input.size(); i++) {
        std::cout << input.at(i) << ' ';
    }
}

int main(){
    /*
    This module provesses one image with multiple ROIs to calculate SFR.
    The image is read in using OpenCV.
    The ROIs are read in from a .csv file.
    Analysis is performed on each ROI to calculate the SFR using the get_sfr function from sfr_utility.cpp.
    The output SFR data is written to a Json file. 
    */

    // 3 micron pixel size
    // float pixel_size = 3 * 0.001;
    float pixel_size = 1.0;
    float freq_Ny4 = 1/(2*pixel_size)/4;
    
    // Used opencv to read png image.
    string dir = "/Users/thomasvo/Documents/GitHub/SFR-Spatial-Frequency-Response-/data/"; // folder containing ROI and input image
    string file_img = "images/test.png"; // The raw image
    string file_roi = "config/rois.csv"; // ROI coordinates
    
    cout << "\n" << "Target directory: \n" << dir << "\n" << endl;

    string dir_img = dir + file_img;
    string dir_roi = dir + file_roi;
    
    cout << "Open Image:\n " << dir_img << endl;
    
    // Open Image
    cv::Mat img = cv::imread(dir_img, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Cannot find image or open image." << endl;
    }

    // The upper left is the origin for each of the ROIs
    vector<int> roi_x, roi_y, roi_x_del, roi_y_del;
    string line, num;

    // Open ROI file with fstream
    fstream fin;
    string sn = file_roi.substr(file_roi.find("/")+1, file_roi.find(".")-file_roi.find("/")-1);
    fin.open(dir_roi, ios::in);
    cout << "\nROI Location: \n" << dir_roi << "\n" << endl;

    bool isFirstLine = 1;
    while(getline(fin, num, ',')){
        // To check if there is a \357 \273 \277 prefix, which indicates UTF-8, at the beginning of the .csv file
        if (isFirstLine == true && num[0] == '\357') {
            num.erase(0,3);
            isFirstLine = false;
        }
        roi_x.push_back(stoi(num));
        getline(fin, num, ',');
        roi_y.push_back(stoi(num));
        getline(fin, num, ',');
        roi_x_del.push_back(stoi(num));
        getline(fin, num, '\n');
        roi_y_del.push_back(stoi(num));
    }

    // Get the ROI from the image
    int  n=roi_x.size();
    vector<double> mtf_at_Ny4_poly;
    vector<double> mtf_at_Ny4_intp;

    cout << "roi_x size = " << n << "\n";

    // Create image string::path for ROIs
    std::string path(dir+"images/" + sn + "/");
    cout << "\n" << path << "\n";
    // Crete char copy of string.
    int ns = path.length();
    char char_array[ns + 1];
    strcpy(char_array, path.c_str());

    // Create ROI path and chek if successful.
    int check;
    // check = mkdir(char_array, 0777);
    
    if (exists_test(char_array)){
        printf("\n Directory Exists. populating ");
        printf("%s", char_array);
        cout << "\n";
    }
    else {
        printf("\n Directory does not exist creating...");
        check = mkdir(char_array, 0777);
    }

    for(int i=0; i<n; i++){
        // Use openCV to crop the image to each ROI.
        cv::Rect rect(roi_x[i], roi_y[i], roi_x_del[i]+1, roi_y_del[i]+1);
        cv::Mat roiRef(img, rect);
        cv::Mat img_roi;
        roiRef.copyTo(img_roi);

        /* // Test image display of ROI. 
        if(i==1){
            while(true){
            cv::imshow("First Rotation of Gray", img_roi);
            if(cv::waitKey(30)>=0) break;
            }
        }
        */

       string filename(dir);
       filename += "images/" + sn + "/img_roi_" + to_string(i+1) + ".png";

       //cout << filename << "\n";
       cv::imwrite(filename, img_roi);
       
       vector<double> roi = (vector<double>) (img_roi.reshape(1,1));
       
       /*
       if (i==1){
        print(roi);
       }
       */

        vector<double> esf, psf, mtf, freq;

        /* MAIN FUNCTION */
        cout << "Filename " << filename;
        
        get_sfr(pixel_size, roi, img_roi.rows, img_roi.cols, esf, psf, mtf, freq, filename);
        
        /* Polyfit to find MTF@Ny/4 */
        int n_poly = 10;
        vector<double> coeff(n_poly);
        /*polyfit(freq, mtf, coeff, n_poly);*/

        double sfr_Ny4 = coeff[0];
        for (int i = 1; i <= n_poly; i++){
            sfr_Ny4 += coeff[i] * pow(freq_Ny4, i);
        }
        mtf_at_Ny4_poly.push_back(sfr_Ny4);
        cout << "ROI " << i+1 << ": MTF at Ny/4 (poly): " << sfr_Ny4 << endl;

            // Create Json File
        Json::Value data_file_out;   

        // Initiate parameters
        Json::Value vec(Json::arrayValue);
        Json::Value esf_out(Json::arrayValue);
        Json::Value psf_out(Json::arrayValue);
        Json::Value first_deriv(Json::arrayValue);
        Json::Value mtf_out(Json::arrayValue);
        Json::Value freq_out(Json::arrayValue);
        Json::Value roi_out(Json::arrayValue);

        for (int i =0; i<roi.size(); i++){
            vec.append(roi[i]);
        }
        for (int i= 0; i< esf.size(); i++){
            esf_out.append(esf[i]);
        }
        for (int i= 0; i< psf.size(); i++){
            psf_out.append(psf[i]);
        }
        for (int i= 0; i< mtf.size(); i++){
            mtf_out.append(mtf[i]);
        }
        for (int i =0; i<freq.size(); i++){
            freq_out.append(freq[i]);
        }
        for (int i =0; i<roi.size(); i++){
            roi_out.append(roi[i]);
        }

        data_file_out["Metadata"]["Filename"] = filename;
        data_file_out["Metadata"]["Number of Rows"] = img_roi.rows;
        data_file_out["Metadata"]["Number of Columns"] = img_roi.cols;
        data_file_out["Data"]["Image Stream"] = vec;
        data_file_out["Data"]["Raw Image"] = roi_out;
        data_file_out["Data"]["ESF"] = esf_out;
        data_file_out["Data"]["PSF"] = psf_out;
        data_file_out["Data"]["MTF"] = mtf_out;
        data_file_out["Data"]["Freq"] = freq_out;

        // Write out the Json map to data file.
        std::ofstream file_id( (filename + ".txt" ).c_str() );
        Json::StyledWriter styledWriter;
        file_id << styledWriter.write(data_file_out);
        file_id.close();

    }


}