'''
The code was rewritten into python based on the matlab code provided by Kaiming He . 
Reference: He, K., Sun, J., & Tang, X. (2010). Guided Image Filtering. Computer Vision – ECCV 2010. Springer Berlin Heidelberg.
           http://kaiminghe.com/publications/eccv10guidedfilter.pdf
'''

import gdal
import numpy as np
import cv2

# Import files
P_rf = gdal.Open(r'D:\Pass\Boka\MLP_test\11_62\test\Comb\UpScAtCor_LC08_L1TP_011062_20150125_20170413_01_T1_B5.rst')  # input image
I_rf = gdal.Open(r'D:\Pass\Boka\MLP_test\11_62\test\Comb\UpScAtCor_LC08_L1TP_011062_20150125_20170413_01_T1_B5.rst')  # guide image


# Parameters of the scene
pj = P_rf.GetProjection()
gt = P_rf.GetGeoTransform()
col = P_rf.GetRasterBand(1).XSize
row = P_rf.GetRasterBand(1).YSize

# Read input as array
P = P_rf.ReadAsArray()
I = I_rf.ReadAsArray()


# box filter
def boxfilter(I, r):
    bf = cv2.boxFilter(I, -1, (r, r))
    return bf

 # Guided filter
 def guidefilter(I, P, r, eps):
     N = boxfilter(np.ones(np.shape(I)), r)

     mean_I = boxfilter(I, r) / N
     mean_P = boxfilter(P, r) / N
     mean_IP = boxfilter(I * P, r) / N
     cov_IP = mean_IP - mean_I * mean_P  # covariance of (I,P) in each local patch

     mean_II = boxfilter(I * I, r) / N
     var_I = mean_II - mean_I * mean_I

     a = cov_IP / (var_I + eps)  # equation 5 in the paper
     b = mean_P - a * mean_I  # equation 6 in the paper

     mean_a = boxfilter(a, r) / N
     mean_b = boxfilter(b, r) / N

     q = mean_a * I + mean_b  # equation 8 in the paper
     return q



# Guided filter parameter setting
# r: filter radius, eps: regularization coefficient
for r in range(5, 32, 4):
    for e in range(50, 400, 100):
            eps = e ** 2
   
            # Output path and name
            output = guidefilter(I, P, r, eps)
            out_path = r'D:\Pass\Boka\MLP_test\11_62\test\guided filter'
            out_name = '\GF_1162_r' + str(r) + 'eps'+str(e)+'_B5.rst'

            # Write the result into an rst file
            rst_driver = gdal.GetDriverByName('RST')
            out_ds = rst_driver.Create(out_path + out_name, col, row, 1, gdal.GDT_Float32) #output data type as float 32 bits
            out_ds.GetRasterBand(1).WriteArray(output)
            out_ds.SetProjection(pj)
            out_ds.SetGeoTransform(gt)
            del out_ds





