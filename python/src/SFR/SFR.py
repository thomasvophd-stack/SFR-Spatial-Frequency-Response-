import json
import numpy as np
import pandas as pd
import cv2
import scipy

import matplotlib.pyplot as plt


class SFR:
    def __init__(self, image_path, config_path):
        """
        Initialize the SFR class with star formation rate data.

        Parameters:
        image_path (str): Path to the image file.
        config_path (str): Path to the JSON configuration file.


        """
        # Load additional configuration
        sfr_config = self.load_config(config_path)
        self.pixel_size = sfr_config['pixel_size']
        self.rois = sfr_config['rois']
        
        self.bin = 4  # Number of bins for SFR computation

        # load image if needed
        self.image_path = image_path
        self.image = self.load_image(self.image_path)
    
    @staticmethod
    def load_config(config_path):
        """
        Load the SFR configuration from a JSON file.

        Parameters:
        config_path (str): Path to the JSON configuration file.

        Returns:
        dict: Configuration parameters for SFR.
        """
        try:
            # Open the file in read mode ('r')
            with open(config_path, 'r', encoding='utf-8') as file:
                # Load the JSON data from the file
                config = json.load(file)

        except FileNotFoundError:
                print("Error: The file 'data.json' was not found.")
        except json.JSONDecodeError as e:
                print(f"Error: Failed to decode JSON from the file: {e}")

        return config
    
    @staticmethod
    def load_image(image_path):
        """
        Load the image from the specified path.

        Parameters:
        image_path (str): Path to the image file.
        """
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image file not found at path: {image_path}")
        return image

    def compute_sfr(self):
        """
        Compute the spatial frequency response (SFR) for the loaded image.

        Returns:
        dict: Computed SFR data.
        """
        # Placeholder for SFR computation logic
        sfr_data = {}

        # Crop ROIs and compute SFR for each
        for idx, roi in enumerate(self.rois):
            cropped_image = self.crop_roi(self.image, roi)
            # Perform SFR computation on cropped_image
            # This is a placeholder for actual SFR computation logic
            sfr_data_for_roi = self.get_sfr_data(cropped_image, cropped_image.shape[0], cropped_image.shape[1])
            sfr_data[f'roi_{idx}'] = sfr_data_for_roi

        return sfr_data

    def get_sfr_data(self, cropped_image, rows, cols):
        """
        Get the SFR data for a cropped image.

        Parameters:
        cropped_image (numpy.ndarray): The cropped image.

        Returns:
        dict: SFR data for the cropped image.
        """
        
        delta = self.pixel_size

        # If image edge is horizontal, flip the SFR data
        roi, nlin, npix, rflag = self.rotate(cropped_image, rows, cols, 0)

        # Orient the edge from dark to light.
        tleft = 0.0
        tright = 0.0
        for i in range(5):
            temp1 = self.get_col(roi, nlin, npix, i)
            temp2 = self.get_col(roi, nlin, npix, npix - 1 - i)
            tleft += np.sum(temp1)
            tright += np.sum(temp2)

        if tleft > tright:
            fil1 = [0.5, 0, -0.5]
            fil2 = [-0.5, 0, 0.5]
        else:
            fil1 = [-0.5, 0, 0.5]
            fil2 = [0.5, 0, -0.5]

        # Create hamming window
        win1 = np.array(self.ahamming(npix, (npix+1)/2.0))
        #print("win1:", win1)

        #plt.figure()
        #plt.plot(win1)
        #plt.title("Hamming Window 1")
        #plt.show()

        # Find Initial Edge Detection
        c_temp = np.array(self.deriv1(roi, nlin, npix, fil1))
        #print("c_temp:", c_temp)
        
        loc = np.zeros(nlin)
        for i in range(nlin):
            row = c_temp[i * npix:(i + 1) * npix]
            apply_win1 = row * win1
            #print("apply_win1:", apply_win1)
            loc[i] = self.centroid(apply_win1) + 0.5

        # Create a counter vector
        index = np.arange(1, nlin + 1)
        #print("index:", index)
        #print("loc:", loc)
        # Fit line to edge location
        fit_result = self.line_fit_least_squares(index, loc)
        #print("Initial fit results: ", fit_result)

        place = np.zeros(nlin)
        for i in range(nlin):
            place[i] = fit_result[0] * (i + 1) + fit_result[1]
            win2 = np.array(self.ahamming(npix, place[i]))
            apply_win2 = c_temp[i * npix:(i + 1) * npix] * win2
            loc[i] = self.centroid(apply_win2) + 0.5
        
        fit_result_2 = self.line_fit_least_squares(index, loc)
        #print("Refined fit results: ", fit_result_2)

        # 
        delfac = np.cos(np.arctan(fit_result_2[0]))
        delta = delta * delfac

        nn = np.floor(npix * self.bin).astype(int)
        nn2 = nn//2 + 1
        nn2_out = round(nn/2)

        # Calcualte ESF
        esf = self.project(roi, nlin, npix, fit_result_2[0], self.bin)
        
        # Calculate PSF
        psf = self.deriv1(esf, 1, nn, fil2)
        mid = self.centroid(psf)
        psf = self.cent(psf, mid)

        # Create centered Hamming window for MTF
        win3 = np.array(self.ahamming(self.bin*npix, (self.bin*npix+1)/2.0))
        psf_win = psf * win3

        # Calculate MTF
        mtf = np.abs(np.fft.fft(psf_win))
        mtf = mtf[:nn]
        mtf = mtf / mtf[0]

        # Frequency axis
        freq = np.linspace(0, 0.5 / delta, nn)

        # Correct for FIR filter effect
        correct = self.compute_correct(nn, self.bin)
        mtf_corrected = mtf * correct

        return esf, psf, mtf_corrected, freq, delta

    def compute_correct(self, n, m):
        """
        Compute the 'correct' weighting array.

        This is a direct translation of the C++ code, preserving:
        - 1-based loop logic
        - index offsets (i-1)
        - clamping to a maximum value of 10
        """

        # Allocate output array (vector<double> correct(n))
        correct = np.zeros(n, dtype=float)

        # Match C++ behavior: decrement m before use
        m -= 1

        # First element is explicitly set
        correct[0] = 1.0

        # Loop corresponds to: for (int i = 2; i < n+1; i++)
        # i runs from 2 to n (inclusive)
        for i in range(2, n + 1):
            # Compute argument used in both numerator and denominator
            arg = (np.pi * i * m) / (2.0 * (n + 1))

            # Compute correction factor
            # abs( (pi * i * m / (2*(n+1))) / sin(...) )
            val = abs(arg / np.sin(arg))

            # Clamp to a maximum value of 10
            if val > 10.0:
                val = 10.0

            # Store result (i-1 because Python is 0-based)
            correct[i - 1] = val

        return correct


    def cent(self, a, center):
        """
        Shift a 1D array so that its 'center' aligns with the middle index.

        This function copies values from the input array into a new array,
        applying an integer shift determined by the difference between the
        desired center location and the array midpoint.

        Parameters
        ----------
        a : array-like
            Input 1D array.
        center : float
            Desired center location (1-based, matching the C++ logic).

        Returns
        -------
        b : numpy.ndarray
            Shifted array of the same length as `a`.
        """

        # Convert input to NumPy array for consistency
        a = np.asarray(a, dtype=float)

        # Length of the array
        n = a.size

        # Output array (initialized to zeros, like default vector<double>)
        b = np.zeros(n, dtype=float)

        # Compute midpoint index using the same formula as C++
        # Note: (n + 1) / 2 is evaluated in floating point, then rounded
        mid = int(round((n + 1) / 2))

        # Integer shift needed to move 'center' to 'mid'
        # Positive del -> shift left
        # Negative del -> shift right
        del_ = int(round(center - mid))

        # -------------------------------------------------------------
        # Apply the shift by copying valid overlapping elements only
        #
        # We avoid out-of-bounds access by limiting loop ranges,
        # exactly as done in the C++ implementation.
        # -------------------------------------------------------------
        if del_ > 0:
            # Shift left: elements move toward lower indices
            for i in range(0, n - del_):
                b[i] = a[i + del_]
        else:
            # Shift right (or no shift if del_ == 0)
            for i in range(-del_, n):
                b[i] = a[i + del_]

        return b

    def project(self, bb, nlin, npix, slope, fac):
        """
        Project a 2D image (flattened into 1D) along a slanted direction,
        bin the values, and return a 1D projected profile.

        Parameters
        ----------
        bb : array-like
            Flattened image of size nlin*npix (row-major order).
        nlin : int
            Number of rows in the image (y dimension).
        npix : int
            Number of columns in the image (x dimension).
        slope : float
            Slope of the projection line (will be inverted internally).
        fac : int
            Oversampling / binning factor along the projection axis.
        """

        # Ensure input is a NumPy array (float for accumulation)
        bb = np.asarray(bb, dtype=float).ravel()

        # Total number of bins in the projected output
        nn = npix * fac

        # Invert slope to match original C++ behavior
        slope = 1.0 / slope

        # Compute offset to shift projection indices so they remain positive
        # This accounts for the maximum vertical extent of the projection
        offset = int(round(fac * (0 - (nlin - 1) / slope)))
        delt = abs(offset)

        #print(f"Offset is {offset}")

        # In C++ code: positive offsets are clipped to zero
        if offset > 0:
            offset = 0

        # Allocate arrays for:
        #  - barray_cnt : number of samples contributing to each bin
        #  - barray_val : sum of pixel values contributing to each bin
        #
        # Extra padding (+100) prevents accidental out-of-bounds access
        size = nn + delt + 100
        barray_cnt = np.zeros(size, dtype=float)
        barray_val = np.zeros(size, dtype=float)

        # ------------------------------------------------------------------
        # Projection and binning
        #
        # Loop over every pixel (x, y) in the image:
        #   - Compute its projected coordinate along a slanted axis
        #   - Convert that coordinate to a bin index ("ling")
        #   - Accumulate count and value in that bin
        # ------------------------------------------------------------------
        for n in range(1, npix + 1):
            for m in range(1, nlin + 1):
                # Convert 1-based loop indices to 0-based pixel coordinates
                x = n - 1
                y = m - 1

                # Project (x, y) onto a slanted axis:
                #   (x - y/slope) gives position along the projection line
                #   fac scales to higher-resolution bins
                #   ceil() matches C++ behavior exactly
                ling = int(np.ceil((x - y / slope) * fac)) + 1 - offset

                # Convert to 0-based array index
                idx = ling - 1
                val = bb[npix * y + x]
                #print("Type(val), np.shape(val)", type(val), np.shape(val))
                # Accumulate contribution
                barray_cnt[idx] += 1.0
                barray_val[idx] += bb[npix * y + x]

        # Output projected profile
        point = np.zeros(nn, dtype=float)

        # Starting index for the valid projection region
        # (centered relative to the offset padding)
        start = int(round(0.5 * delt))

        # ------------------------------------------------------------------
        # Handle bins with zero counts
        #
        # If a bin received no samples:
        #   - Replace it using neighboring bins
        #   - This prevents division by zero later
        # ------------------------------------------------------------------
        nz = 0
        for i in range(start, start + nn):
            if barray_cnt[i] == 0:
                nz += 1
                if i == 0:
                    # Edge case: copy from next bin
                    barray_cnt[i] = barray_cnt[i + 1]
                    barray_val[i] = barray_val[i + 1]
                else:
                    # Average neighboring bins
                    barray_cnt[i] = 0.5 * (barray_cnt[i - 1] + barray_cnt[i + 1])
                    barray_val[i] = 0.5 * (barray_val[i - 1] + barray_val[i + 1])

        # ------------------------------------------------------------------
        # Normalize summed values by counts to get final projection
        # ------------------------------------------------------------------
        for i in range(nn):
            point[i] = barray_val[i + start] / barray_cnt[i + start]

        return point


    def line_fit_least_squares(self, data_x, data_y):
        """
        Perform a least-squares line fit y = m*x + b and compute r^2.

        This is a direct translation of the C++ implementation, using
        explicit summations rather than NumPy's polyfit, to preserve
        numerical behavior and intent.

        Parameters
        ----------
        data_x : array-like
            x-coordinates of data points.
        data_y : array-like
            y-coordinates of data points.

        Returns
        -------
        result : numpy.ndarray
            Array containing [m, b, r_squared]
            where:
                m = slope
                b = intercept
                r_squared = correlation coefficient squared
        """

        # Convert inputs to NumPy arrays
        x = np.asarray(data_x, dtype=float)
        y = np.asarray(data_y, dtype=float)

        #print("Line Fit Input X:", x, "Y:", y)

        # Number of data points
        data_n = len(x)

        # Initialize summation terms (match C++ floats conceptually)
        sumx = 0.0     # Sum of x
        sumy = 0.0     # Sum of y
        sumxy = 0.0    # Sum of x * y
        sumx2 = 0.0    # Sum of x^2
        sumy2 = 0.0    # Sum of y^2

        # Accumulate sums
        for i in range(data_n):
            sumx  += x[i]
            sumx2 += x[i] ** 2
            sumxy += x[i] * y[i]
            sumy  += y[i]
            sumy2 += y[i] ** 2

        #print("Sumx:", sumx, "Sumy:", sumy, "Sumxy:", sumxy, "Sumx2:", sumx2, "Sumy2:", sumy2)

        # Denominator used in both slope and intercept
        denom = data_n * sumx2 - sumx ** 2

        # Least-squares slope (m) and intercept (b)
        m = (data_n * sumxy - sumx * sumy) / denom
        b = (sumy * sumx2 - sumx * sumxy) / denom

        # Correlation coefficient r
        r = (sumxy - (sumx * sumy) / data_n) / np.sqrt(
            (sumx2 - (sumx ** 2) / data_n) *
            (sumy2 - (sumy ** 2) / data_n)
        )

        # Store results in the same order as the C++ vector push_back calls
        result = np.array([m, b, r * r], dtype=float)

        # Match C++ debug output
        #print("r*r")

        return result

    def centroid(self, x):
        """
        Compute the centroid (center of mass) of a 1D array.

        This matches the C++ implementation:
            centroid = sum(i * x[i]) / sum(x)

        Indices are 0-based, exactly as in the C++ loop.

        Parameters
        ----------
        x : array-like
            Input 1D array of weights or intensities.

        Returns
        -------
        float
            Centroid location (0-based index).
        """

        # Convert input to NumPy array
        x = np.asarray(x, dtype=float)

        # Sum of all values (equivalent to sum_vec(x))
        sumx = np.sum(x)
        #print("x.size:", x.size)
        # Accumulate weighted index sum
        temp = 0.0
        for i in range(x.size):
            #print(i)
            temp += i * x[i]

        # Return centroid position
        return temp / sumx


    def ahamming(self, n, mid):
        """
        Create a Hamming window of length n with a specified center position.

        This function mirrors the C++ implementation exactly, including:
        - 1-based indexing logic (i+1)
        - symmetric width determined from the provided midpoint
        - cosine-based Hamming formulation

        Parameters
        ----------
        n : int
            Length of the window.
        mid : float
            Center position (1-based, as in the C++ code).

        Returns
        -------
        data : numpy.ndarray
            Hamming window values of length n.
        """

        # Allocate output array
        data = np.zeros(n, dtype=float)

        # Distance from the midpoint to the left and right edges
        wid1 = mid - 1
        wid2 = n - mid

        # Use the larger distance to maintain symmetry
        wid = max(wid1, wid2)

        # Compute the window values
        for i in range(n):
            # arg uses 1-based indexing to match C++
            arg = (i + 1) - mid

            # Cosine term
            c = np.cos(np.pi * arg / wid)

            # Hamming window formula
            data[i] = 0.54 + 0.46 * c

        return data

    
    def rotate90(self,roi_arr, nlin, npix):
        """Rotates the matrix 90 degrees (clockwise in this implementation)."""
        # Transpose and reverse rows for 90-degree clockwise rotation
        # The C++ code seems to do an anti-clockwise rotation based on indexing,
        # but let's match the effect: new_roi[nlin*(npix-1-c)+r] = old_roi[npix*r + c]
        
        # Pythonic way: Transpose then flip horizontally (or flip vertically then transpose)
        # For a 90 deg clockwise rotation:
        rotated_arr = np.rot90(roi_arr, k=-1) # k=-1 for 90 deg clockwise
        
        # Or manually:
        # rotated_arr = np.empty_like(roi_arr).T # Creates a transposed shape array
        # for r in range(nlin):
        #     for c in range(npix):
        #         rotated_arr[c, nlin - 1 - r] = roi_arr[r, c]
                
        nlin, npix = npix, nlin # Swap dimensions
        return rotated_arr, nlin, npix

    def rotate(self, roi_arr, nlin, npix, rflag):
        """Decides whether to rotate the region of interest (ROI) by 90 degrees."""
        nn = 2
        
        # Ensure indices are valid
        if nlin - 1 - nn < 0 or nn >= nlin or npix - 1 - nn < 0 or nn >= npix:
            print("Warning: Test area indices out of bounds.")
            return roi_arr, nlin, npix, rflag

        # Get test rows and columns
        r1 = self.get_row(roi_arr, nlin, npix, nlin - 1 - nn)
        r2 = self.get_row(roi_arr, nlin, npix, nn)
        c1 = self.get_col(roi_arr, nlin, npix, npix - 1 - nn)
        c2 = self.get_col(roi_arr, nlin, npix, nn)

        # Calculate differences in means
        testv = abs(np.mean(r1) - np.mean(r2))
        testh = abs(np.mean(c1) - np.mean(c2))

        if testv > testh:
            rflag = 1
            roi_arr, nlin, npix = self.rotate90(roi_arr, nlin, npix)
        
        return roi_arr, nlin, npix, rflag
    

    def get_col(self, roi_flat, nlin, npix, col):
        """Equivalent to C++ get_col"""
        # 1. Reshape 1D vector to 2D matrix (nlin x npix)
        # 2. Slice all rows (:) and the specific column
        matrix = np.array(roi_flat).reshape((nlin, npix))
        return matrix[:, col]
    
    def get_row(self, roi_flat, nlin, npix, row):
        """Equivalent to C++ get_row"""
        # 1. Reshape 1D vector to 2D matrix (nlin x npix)
        # 2. Slice the specific row and all columns (:)
        matrix = np.array(roi_flat).reshape((nlin, npix))
        return matrix[row, :]
    
    def crop_roi(self, image, roi):
        """
        Crop the region of interest (ROI) from the image.

        Parameters:
        image (numpy.ndarray): The input image.
        roi (tuple): The ROI defined as (x, y, width, height).

        Returns:
        numpy.ndarray: Cropped ROI image.
        """
        x, y, w, h = roi
        return image[y:y+h, x:x+w]
    
    def deriv1(self, a, nlin, npix, fil):
        print("In deriv1", a.shape, nlin, npix, fil)
        a = np.asarray(a, dtype=float).reshape(nlin, npix)
        fil = np.asarray(fil, dtype=float)

        nn = fil.size
        fil2cov = fil[::-1]

        b = np.zeros_like(a)

        # Vectorized accumulation over filter taps
        for k in range(nn):
            b[:, nn-1:] += fil2cov[k] * a[:, k:npix - (nn-1) + k]

        # Boundary handling (exact C++ behavior)
        if nn >= 2:
            b[:, nn-2] = b[:, nn-1]

        return b.ravel()
    

if __name__ == "__main__":
    # Example usage
    sfr = SFR(r"/Users/thomasvo/Documents/GitHub/SFR-Spatial-Frequency-Response-/data/images/test.png",r"/Users/thomasvo/Documents/GitHub/SFR-Spatial-Frequency-Response-/data/config/config.json")
    sfr_data = sfr.compute_sfr()
    plt.figure()
    for key, (esf, psf, mtf_corrected, freq, delta) in sfr_data.items():
        plt.plot(freq, mtf_corrected, label=key)
    plt.xlabel('Spatial Frequency (cycles/pixel)')
    plt.ylabel('MTF Corrected')
    plt.title('Spatial Frequency Response (SFR)')
    plt.legend()
    plt.show()