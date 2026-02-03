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

struct ROI {
    int x;
    int y;
    int x_del;
    int y_del;
};

std::vector<ROI> load_rois_from_json(const std::string& filepath) {
    /*
    Opens a JSON file and reads in ROIs defined as arrays of four integers:
    [x, y, x_del, y_del]

    @param filepath: Path to the JSON file.
    @return a vector of ROI structs.
    */

    // Open the JSON file
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filepath << std::endl;
        throw std::runtime_error("Could not open file: " + filepath);
    }

    // Define JSON root, parser objects, and error string
    Json::Value root;
    Json::CharReaderBuilder reader;
    std::string errs;

    // Parse the JSON file
    if (!Json::parseFromStream(reader, file, &root, &errs)) {
        std::cerr << "Error parsing JSON: " << errs << std::endl;
        std::runtime_error("Failed to parse JSON: " + errs);

    }

    // Ensure rois key exists and is an array
    if (!root.isMember("rois") || !root["rois"].isArray()) {
        throw std::runtime_error("'rois' key is missing or is not an array in the JSON file.");
    }

    // Read in rois from json
    const Json::Value& rois_json = root["rois"];
    std::vector<ROI> rois;
    rois.reserve(rois_json.size());

    for (Json::ArrayIndex i = 0; i < rois_json.size(); ++i) {
        const Json::Value& roi = rois_json[i];

        if (!roi.isArray() || roi.size() != 4){
            throw std::runtime_error("Each ROI must be an array of four integers." + std::to_string(i));
        }

        ROI r;
        r.x = roi[0].asInt();
        r.y = roi[1].asInt();
        r.x_del = roi[2].asInt();
        r.y_del = roi[3].asInt();

        rois.push_back(r);
    }

    return rois;
}


int main(){
    /*
    This module processes one image with multiple ROIs to calculate SFR.
    The image is read using OpenCV.
    The ROIs are read from a config file
    */


    // Used opencv to read png image.
    string dir = "/Users/thomasvo/Documents/GitHub/SFR-Spatial-Frequency-Response-/data/"; // folder containing ROI and input image
    string file_img = "images/test.png"; // The raw image
    string file_config = "config/config.json"; // Config file
    string sn = "TestImage"; // Placeholder for sample name
    string config_dir = dir + file_config;
    string img_dir = dir + file_img;

    // Open the JSON file
    cout << "--- Opening JSON File ---" << std::endl;
    std::ifstream json_file(config_dir, std::ifstream::binary);
    if (!json_file.is_open()) {
        std::cerr << "Error: Could not open the file." << std::endl;
        return EXIT_FAILURE;
    }

    // Parse JSON content
    Json::Value root;
    try {
        json_file >> root;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    // Access the data
    std::cout << "--- Reading Data ---" << std::endl;

    // Extract pixel_size
    if (root.isMember("pixel_size")) {
        float pixel_size = root["pixel_size"].asFloat();
        std::cout << "Pixel Size: " << pixel_size << std::endl;
    }
    else {
        std::cerr << "Error: 'pixel_size' not found in JSON." << std::endl;
        return EXIT_FAILURE;
    }
    float pixel_size = root["pixel_size"].asFloat();
    float freq_Ny4 = 1/(2*pixel_size)/4;
    cout << "Nyquist = " << freq_Ny4;

    // Extract ROIs
    try {
        // Try loading rois
        auto rois = load_rois_from_json(config_dir);
        
        // test print loaded rois
        for (const auto& roi : rois) {
            std::cout << "Loaded ROI - x: " << roi.x << ", y: " << roi.y
                      << ", x_del: " << roi.x_del << ", y_del: " << roi.y_del << std::endl;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error loading ROIs: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    // Actually Load ROIs
    auto rois = load_rois_from_json(config_dir);
    // The upper left is the origin for each of the ROIs
    vector<int> roi_x, roi_y, roi_x_del, roi_y_del;
    string line, num;

    int counter = 0;
    for (const auto& roi: rois){
        roi_x.push_back(roi.x);
        roi_y.push_back(roi.y);
        roi_x_del.push_back(roi.x_del);
        roi_y_del.push_back(roi.y_del);

    }

    // Open image using OpenCV
    cout << "Open Image:\n " << img_dir << endl;
    cv::Mat img = cv::imread(img_dir, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        cout << "Cannot find image or open image." << endl;
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

        string filename(dir);
        filename += "images/" + sn + "/img_roi_" + to_string(i+1) + ".png";

        cv::imwrite(filename, img_roi);       
        vector<double> roi = (vector<double>) (img_roi.reshape(1,1));
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