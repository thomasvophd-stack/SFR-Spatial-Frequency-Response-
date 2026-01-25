#include <iostream>
#include <vector>
#include <math.h>
#include "../include/sfr_utility.h"
#include "../third_party/kiss_fft130/kiss_fft.h"
#include "float.h"

#include <fstream>
#include <vector>

#include <iomanip>

using namespace std;


void get_sfr(double del, vector<double> roi, int nlin, int npix, vector<double>&esf, vector<double>&psf, vector<double>&mtf, vector<double>&freq, string filename){
    /*
    Main SFR function
    */
    cout << "\n" << "Running SFR Subroutine...";
    int nbin = 4;
    
    // Easiest to scan vertical edges, so rotate all horizontals.
    int rflag = 0;
    rotate(roi, nlin, npix, rflag);

    cout<< "Rotation Flag is " << rflag << "\n";
    vector<double> loc(nlin);
    vector<double> fil1 = {0.5, -0.5};
    vector<double> fil2 = {0.5, 0, -0.5};

    // Easiest to measure Edge SFR from dark to light.
    // So reverse if light to dark.
    double tleft = 0;
    double tright = 0;
    for (int i = 0; i < 5; i++) {
        vector<double> temp1 = get_col(roi, nlin, npix, i);
        vector<double> temp2 = get_col(roi, nlin, npix, npix-1-i);
        tleft += sum_vec(temp1);
        tright += sum_vec(temp2);
    }
    
    if (tleft > tright) {
        fil1 = {-0.5, 0.5};
        fil2 = {-0.5, 0, 0.5};
    }

    // Smoothing window for the initial edge location estimation -
    // Apply the hamming window to each line of ROI
    vector<double> win1 = ahamming(npix, (npix+1)/2);

    //Find Edge
    vector<double> c_temp = deriv1(roi, nlin, npix, fil1);

    for (int i = 0; i < nlin; i++){
        vector<double> apply_win1(npix);
        for (int j = 0; j < npix; j++){
            apply_win1[j] = c_temp[i*npix + j] * win1[j];
        }
        loc[i] = centroid(apply_win1) + 0.5;
    }

    // Create counter vector
    vector <double> index(nlin);
    for (int i=0; i<nlin; i++){
        index[i]=i;
    }

    /* Linear fit for an initial edge detection.*/
    vector<double> fitResult;
    LineFitLeastSquares(index, loc, nlin, fitResult);

    vector<double> place(nlin);
    for (int i = 0; i < nlin; i++){
        //place[i] = fitResult[0] * i + fitResult[1]; 
        place[i] = fitResult[0] * (i+1) + fitResult[1];
        vector<double> win2 = ahamming(npix, place[i]);
        vector<double> apply_win2(npix);
        for (int j = 0; j < npix; j++){
            apply_win2[j] = c_temp[i*npix + j] * win2[j];
        }
        loc[i] = centroid(apply_win2) + 0.5;
    }

    vector<double> fitResult2;
    LineFitLeastSquares(index, loc, nlin, fitResult2);
    cout << "slope: " << fitResult2[0] << "  intercept: " << fitResult2[1] << endl;
    
    /* Correct sampling interval for sampling parallel to edge */
    double delfac = cos(atan(fitResult2[0]));
    del = del * delfac;

    int nn = floor(npix * nbin);
    int nn2 = floor(nn/2) + 1;
    int nn2out = round(nn2 /2.0);
    // double nfreq = 1/(2*del);
 
    /* Get 4x super sampled ESF*/
    esf = project(roi,  nlin, npix, fitResult2[0], nbin);

    cout << "ESF size is " << esf.size() << "\n";

    /* compute first derivative via FIR (1x3) filter fil2 */
    psf = deriv1(esf, 1, nn, fil2);
    
    /* centering PSF */
    double mid = centroid(psf);
    vector<double> psf_center = cent(psf, round(mid));
    
    /* centered Hamming window */
    vector<double> win = ahamming(nbin*npix,(nbin*npix+1)/2.0);
    
    /* Apply centerd Hamming window */
    for (int i=0; i<win.size(); i++){
        psf_center[i] = psf_center[i] * win[i];
    }
    
    /* Transform, scale and correct for FIR filter response */
    /* Used KISS FFT for 1D fft https://github.com/mborgerding/kissfft */
    kiss_fft_cfg cfg = kiss_fft_alloc(nn, 0, 0, 0);
    kiss_fft_cpx cx_in[nn];
    kiss_fft_cpx cx_out[nn];
    for (int i = 0; i<nn; i++){
        cx_in[i].r = psf_center[i];
        cx_in[i].i = 0;
    }
    kiss_fft(cfg , cx_in , cx_out);
    vector<double> temp(nn);
    for (int i = 0; i<nn; i++){
        temp[i] = sqrt(cx_out[i].r * cx_out[i].r + cx_out[i].i * cx_out[i].i);
    }
    
    /* dcorr corrects SFR for response of FIR filter */
    vector<double> dcorr = fir2fix(nn2, 3);
    for (int i=0; i<nn2out; i++){
        mtf.push_back(temp[i]/temp[0] * dcorr[i]);
    }
    
    /* freq is the x axis for MTF*/
    for (int i = 0; i<nn2out; i++){
        freq.push_back(nbin * i / (del * nn));
    }

}

void rotate(vector<double>& roi, int& nlin, int& npix, int& rflag){
    /* Limit test area. */
    int nn = 2;
    vector<double> r1 = get_row(roi, nlin, npix, nlin-1-nn);
    vector<double> r2 = get_row(roi, nlin, npix, nn);
    vector<double> c1 = get_col(roi, nlin, npix, npix-1-nn);
    vector<double> c2 = get_col(roi, nlin, npix, nn);
    
    double testv = abs(mean_vec(r1) - mean_vec(r2));
    double testh = abs(mean_vec(c1) - mean_vec(c2));
    if (testv > testh) {
        rflag = 1;
        rotate90(roi, nlin, npix);
    }
    return;
}

void rotate90(vector<double>& roi, int& nlin, int& npix){
    vector<double> roi_new(roi.size());
    for (int r = 0; r < nlin; r++){
        for (int c = 0; c < npix; c++){
            roi_new[nlin*(npix -1 - c)+r] = roi[npix * r + c];
        }
    }
    roi = roi_new;
    int tmp = npix;
    npix = nlin;
    nlin = tmp;
    return;
}

vector<double> get_col(vector<double>&roi, int nlin, int npix, int col){
    vector<double> col_vec;
    for (int i = 0; i < nlin; i++){
        col_vec.push_back(roi[npix * i + col]);
    }
    return col_vec;
}

vector<double> get_row(vector<double>&roi, int nlin, int npix, int row){
    vector<double> row_vec;
    for (int i = 0; i < npix; i++){
        row_vec.push_back(roi[row * npix + i]);
    }
    return row_vec;
}

double mean_vec(vector<double>&v){
    return (sum_vec(v)/v.size());
}

double sum_vec(vector<double>&v){
    double s=0;
    for (double element:v){
        s = s + element;
    }
    return s;
}

double centroid(vector<double>& x){
    double sumx = sum_vec(x);
    double temp = 0.0;
    for (int i = 0; i < x.size(); i++){
        temp += i*x[i];
    }
    return temp / sumx;
}

vector<double> ahamming(int n, double mid){
    /*
    Create a hamming window given a length and middile point.
    */
    vector<double> data(n);
    double wid1 = mid - 1;
    double wid2 = n - mid;
    double wid = max(wid1, wid2);
    for (int i = 0; i < n; i++) {
        double arg = i+1 - mid;
        data[i] = cos(M_PI * arg / wid);
        data[i] = 0.54 + 0.46 * data[i];
    }
    return data;
}

vector<double> deriv1(const vector<double>& a, int nlin, int npix, const vector<double>& fil){
    vector<double> b(nlin * npix);
    int nn = fil.size();
    vector<double> fil2cov(fil);
    reverse(fil2cov.begin(), fil2cov.end());
    for(int r = 0; r < nlin; r++){
        for(int i = nn - 1; i < npix; i++){
            for(int j = 0; j < nn; j++){
                b[r*npix + i] += fil2cov[j] * a[r*npix + i - (nn-1) + j];
            }
        }
        b[r*npix + nn-2] = b[r*npix + nn-1];
    }
    return b;
}

/*
// y = vResult[0]*x + vResult[1]
void LineFitLeastSquares(vector<double>& data_x, vector<double>& data_y, int data_n, vector<double>& vResult)
{
    float A = 0.0;
    float B = 0.0;
    float C = 0.0;
    float D = 0.0;
    float E = 0.0;
    float F = 0.0;
 
    for (int i=0; i<data_n; i++)
    {
        A += data_x[i] * data_x[i];
        B += data_x[i];
        C += data_x[i] * data_y[i];
        D += data_y[i];
    }
 
    // calculate a and b
    float a, b, temp = 0;
    temp = (data_n * A - B * B);
    if( temp )   // in case the denominator is zero
    {
        a = (data_n*C - B*D) / temp;
        b = (A*D - B*C) / temp;
    }
    else
    {
        a = 1;
        b = 0;
    }
 
    // calculate correlation coefficient
    float Xmean, Ymean;
    Xmean = B / data_n;
    Ymean = D / data_n;
 
    float tempSumXX = 0.0, tempSumYY = 0.0;
    for (int i=0; i<data_n; i++)
    {
        tempSumXX += (data_x[i] - Xmean) * (data_x[i] - Xmean);
        tempSumYY += (data_y[i] - Ymean) * (data_y[i] - Ymean);
        E += (data_x[i] - Xmean) * (data_y[i] - Ymean);
    }
    F = sqrt(tempSumXX) * sqrt(tempSumYY);
 
    float r;
    r = E / F;
 
    vResult.push_back(a);
    vResult.push_back(b);
    vResult.push_back(r*r);
}
*/

void LineFitLeastSquares(vector<double>& data_x, vector<double>& data_y, int data_n, vector<double>& vResult)
{
    float sumx = 0.0; // Sum of x vector
    float sumy = 0.0; // Sum of y vetor
    float sumxy = 0.0; // Sum of x*y
    float sumx2 = 0.0; // Sum of x**2
    float sumy2 = 0.0; // Sum of y**2
    
    float m = 0.0;
    float b = 0.0;
    float r = 0.0;
    for (int i =0; i< data_n; i++){
        sumx  += data_x[i];       
        sumx2 += pow(data_x[i],2);  
        sumxy += data_x[i] * data_y[i];
        sumy  += data_y[i];      
        sumy2 += pow(data_y[i],2); 
    }

    float denom = (data_n * sumx2 - pow(sumx,2));
    
    m = (data_n * sumxy  -  sumx * sumy) / denom;
    b = (sumy * sumx2  -  sumx * sumxy) / denom;

    r = (sumxy - sumx * sumy / data_n) / sqrt(
        (sumx2 - pow(sumx,2)/data_n)*(sumy2 - pow(sumy,2)/data_n)
        );

    vResult.push_back(m);
    vResult.push_back(b);
    vResult.push_back(r*r);

    cout<<"r*r";
}



vector<double> project(vector<double> bb, int nlin, int npix, double slope, int fac) {
    int nn = npix * fac;
    slope = 1/slope;
    int offset =  round(fac * (0 - (nlin - 1)/slope ));
    int del = abs(offset);

    cout <<"Offset is " << offset << "\n";
    if (offset>0) {offset=0;}
    vector<double> barray_cnt(nn+del+100);
    vector<double> barray_val(nn+del+100);
    
    /* project and binning */
    for (int n = 1; n < npix+1; n++){
        for (int m = 1; m < nlin+1; m++){
            int x = n - 1;
            int y = m - 1;
            int ling =  ceil((x  - y/slope)*fac) + 1 - offset;
            barray_cnt[ling-1] += 1;
            barray_val[ling-1] += bb[npix * y + x];
        }
    }

    vector<double> point(nn);
    int start = round(0.5*del);
    
    /* Check for zero counts. */
    int nz = 0;
    for (int i = start; i < start+nn; i++){
        if (barray_cnt[i] == 0){
            nz += 1;
            if (i == 0) {
                barray_cnt[i] = barray_cnt[i+1];
                barray_val[i] = barray_val[i+1];
            }
            else{
                barray_cnt[i] = (barray_cnt[i-1] + barray_cnt[i+1]) / 2;
                barray_val[i] = (barray_val[i-1] + barray_val[i+1]) / 2;
            }
        }
    }
    
    for (int i = 0; i < nn; i++) {
        point[i] = barray_val[i+start] / barray_cnt[i+start];
    }

    return point;
}

vector<double> cent(vector<double> a, double center) {
    int n = a.size();
    vector<double> b(n);
    int mid = round((n+1)/2);
    int del = round(center - mid);
    
    if (del > 0) {
        for (int i = 0; i < n-del; i++){
            b[i] = a[i + del];
        }
    }
    else {
        for (int i = -del; i < n; i++){
            b[i] = a[i + del];
        }
    }
    return b;
}

vector<double> fir2fix(int n, int m){
    vector<double> correct(n);
    m -= 1;
    correct[0] = 1;
    for (int i = 2; i < n+1; i++){
        correct[i-1] = abs((M_PI * i * m / (2 * (n+1))) / sin(M_PI * i * m / (2 * (n+1))));
        correct[i-1] = 1 + (correct[i-1]-1);
        if (correct[i-1] > 10) {
            correct[i-1] = 10;
        }
        
    }
    return correct;
}
