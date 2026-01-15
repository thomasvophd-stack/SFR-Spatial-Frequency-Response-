//
//  sfr_utility.h
//  SFR_process
//
//
//

//Header Guards
#ifndef sfr_utility_h
#define sfr_utility_h

#include <vector>
using namespace std;

/**
 * Main Function for SFR processing given a ROI.
 *
 * @param del Pixel Size.
 * @param roi ROI arranged in row first order. (r, c) <-> roi[r*npix + c].
 * @param nlin Number of rows of ROI.
 * @param npix Number of columns of ROI.
 *
 * @param esf Outputs ESF (Edge Spread Function).
 * @param mtf Outputs MTF (Modulation Transfer Function).
 * @param freq Outputs the x axis for MTF.
 * @param filename Filename
 */
void get_sfr(double del, vector<double>roi, int nlin, int npix, vector<double>&esf, vector<double>&psf,  vector<double>&mtf, vector<double>&freq, string filename);

/*
* Determine if the ROI needs to be rotated. Same with rotatev2.m in sfrmat3.
*
* 
 */
void rotate(vector<double>& roi, int& nlin, int& npix, int& rflag);

/*
Rotate the ROI anti-clockwisely for 90 degrees. Same with rotate90.m in sfrmat3.
 */
void rotate90(vector<double>& roi, int& nlin, int& npix);

/*
Get (col)th column of roi.
 */
vector<double> get_col(vector<double>&roi, int nlin, int npix, int col);

/*
Get (row)th row of roi.
 */
vector<double> get_row(vector<double>& roi, int nlin, int npix, int row);

/*
Get centroid index location of the vector.
*/
double centroid(vector<double>& x);

/**
 * Function generates a general asymmetric Hamming-type window array. Same with ahamming.m in sfrmat3.
 * @param n length of array
 * @param mid peak of window
 * @return window array with a size of n.
 */
vector<double> ahamming(int n, double mid);

/**
 * Computes first derivative via FIR (1xn) filter. Edge effects are suppressed and vector size is preserved. Same with deriv1.m in sfrmat3.
 */
vector<double> deriv1(const vector<double>& a, int nlin, int npix, const vector<double>& fil);

/**
 * Source: https://www.mathsisfun.com/data/least-squares-regression.html
 * Use Least Square method to linearly fit data array data_x and data_y.
 * @param data_n length of data_x / data_y
 * @param vResult vResult[0] = slope, vResult[1] = intercept, vResult[2] = r^2
 */
void LineFitLeastSquares(vector<double>& data_x, vector<double>&data_y, int data_n, vector<double> &vResult);

/**
 * Oversample and project the data in array bb along the linear fitted edge. Almost same with project.m in strmat3, with modification for intercept and a vacant bin.
 * @param bb Input roi.
 * @param slope From the linear fitting of edge.
 * @param fac Oversampling factor. Used 4 in code.
 * @return ESF
 */
vector<double> project(vector<double> bb, int nlin, int npix, double slope, int fac);

/**
 * Shift a 1D array to a center. Same with cent.m in sfrmat3.
 */
vector<double> cent(vector<double> a, double center);

double sum_vec(vector<double>&v);
double mean_vec(vector<double>&v);

/**
 * Correction for MTF of derivative (difference) filter. Same with fir2fix.m in sfrmat3.
 * @param n frequency data length (Nyquist frequency)
 * @param m length of differentiation kernel
 * @return MTF correection array with a length of n
 */
vector<double> fir2fix(int n, int m);

/**
 * Source: https://stackoverflow.com/questions/9394867/c-implementation-of-matlab-interp1-function-linear-interpolation
 * Linear interpolation. Given a vector of x and y, input a vector of x_new and output the interpolated result.
 */
vector< double > interp1( vector< double > &x, vector< double > &y, vector< double > &x_new );
int findNearestNeighbourIndex( double value, vector< double > &x );

#endif