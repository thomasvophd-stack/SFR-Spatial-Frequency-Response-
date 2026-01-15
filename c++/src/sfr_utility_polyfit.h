//
//  sfr_utility_polyfit.h
//  OpenCVTest
//
// 
//

#ifndef sfr_utility_polyfit_h
#define sfr_utility_polyfit_h

#include <Eigen/QR>
#include <vector>

/**
 * Source: http://svn.clifford.at/handicraft/2014/polyfit/polyfit.cc
 * @param coeff Outputs the coeff. For example, if order == 3, coeff.size( ) = 4, y = coeff[0] + coeff[1]*x + coeff[2]*x^2 + coeff[3]*x^3
 */
void polyfit(const std::vector<double> &xv, const std::vector<double> &yv, std::vector<double> &coeff, int order)
{
    Eigen::MatrixXd A(xv.size(), order+1);
    Eigen::VectorXd yv_mapped = Eigen::VectorXd::Map(&yv.front(), yv.size());
    Eigen::VectorXd result;

    assert(xv.size() == yv.size());
    assert(xv.size() >= order+1);

    // create matrix
    for (size_t i = 0; i < xv.size(); i++)
    for (size_t j = 0; j < order+1; j++)
        A(i, j) = pow(xv.at(i), j);

    // solve for linear least squares fit
    result = A.householderQr().solve(yv_mapped);

    coeff.resize(order+1);
    for (size_t i = 0; i < order+1; i++)
        coeff[i] = result[i];
}

#endif /* sfr_utility_polyfit_h */
