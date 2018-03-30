# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 12:54:57 2017

@author: shooper
"""

import sys
import os
import time
import gdal
import math
import pandas as pd
import numpy as np
from sklearn import metrics

from lthacks import createMetadata
from lthacks.stats_functions import get_function
from lthacks.intersectMask import array_to_raster

FUNC_OPTIONS = ['mode',
                'mean',
                'nanmean',
                'nansum']

def aggregate_array(ar, window_size, function, nodata=None):
    
    '''# Clip rows and cols without any data
    in_nrows, in_ncols = ar.shape
    in_nodata_mask = ar == nodata
    row_mask_1d = np.all(in_nodata_mask, axis=1)
    col_mask_1d = np.all(in_nodata_mask, axis=0)
    row_mask = np.repeat(row_mask_1d, in_ncols).reshape(in_nrows, in_ncols)
    col_mask = np.repeat(col_mask_1d, in_nrows).reshape(in_nrows, in_ncols)
    data_nrows = in_nrows - row_mask_1d.sum()
    data_ncols = in_ncols - col_mask_1d.sum()
    ar = ar[~row_mask & ~col_mask].astype(np.float16).reshape(data_nrows, data_ncols)'''
    
    # Buffer the input array so that it has enough rows and cols such that 
    #   ar_buf.shape % window_size == 0
    dtype = ar.dtype
    data_nrows, data_ncols = ar.shape
    ar = ar.astype(np.float32) #Has to be 32 bit because precision of 16 bit is only 3
    buf_nrows = int(math.ceil(data_nrows/float(window_size)) * window_size)
    buf_ncols = int(math.ceil(data_ncols/float(window_size)) * window_size)
    ar_buf = np.full((buf_nrows, buf_ncols), np.nan, dtype=np.float32)
    ul_row = (buf_nrows - data_nrows)/2
    ul_col = (buf_ncols - data_ncols)/2
    ar[ar == nodata] = np.nan # Protect nodata values for calculations
    ar_buf[ul_row : ul_row + data_nrows, ul_col : ul_col + data_ncols] = ar # Center data

    # Aggregate
    out_nrows = buf_nrows/window_size
    out_ncols = buf_ncols/window_size
    ar_out = function(ar_buf.reshape(out_nrows, window_size, out_ncols, window_size),
                      axis=(1,3))
    ar_out[np.isnan(ar_out)] = nodata
    ar_out.astype(dtype)
    
    return ar_out.astype(dtype), ul_row, ul_col
    

def get_gdal_dtype(type_code):
    
    code_dict = {1: gdal.GDT_Byte,
                 2: gdal.GDT_UInt16,
                 3: gdal.GDT_Int16,
                 4: gdal.GDT_UInt32,
                 5: gdal.GDT_Int32,
                 6: gdal.GDT_Float32,
                 7: gdal.GDT_Float64,
                 8: gdal.GDT_CInt16,
                 9: gdal.GDT_CInt32,
                 10: gdal.GDT_CFloat32,
                 11: gdal.GDT_CFloat64
                 }
    
    return code_dict[type_code]


def main(raster_path, window_sizes, function_string, nodata, out_dir=None, mask_path=None, mask_val=0):
       
    t0 = time.time()
    if not os.path.exists(raster_path):
        sys.exit('raster_path specified does not exist: ' + raster_path)
    
    window_sizes = [int(s) for s in window_sizes.split(',')]
    
    if function_string in FUNC_OPTIONS:
        if function_string == 'nansum':
            function = np.nansum
        elif function_string == 'nanmean':
            function = np.nanmean
        else:
            function = get_function(function_string)
    else:
        raise ValueError('function_string not understood: %s. Valid options: %s' % (function_string, '\n\t'.join(FUNC_OPTIONS)))
    
    ds = gdal.Open(raster_path)
    ar = ds.ReadAsArray()
    n_bands = ds.RasterCount # Add fucntionality for multiple bands
    ul_x, x_res, x_rot, ul_y, y_rot, y_res = ds.GetGeoTransform()
    prj = ds.GetProjection()
    driver = ds.GetDriver()
    dtype = get_gdal_dtype(ds.GetRasterBand(1).DataType)
    if nodata:
        nodata = int(nodata)
    elif not nodata:
        nodata = ds.GetRasterBand(1).GetNoDataValue()
    else:
        raise RuntimeError('nodata not specified and not retrivable')
    ds = None
    
    for window_size in window_sizes:
        ar_out, ul_row, ul_col = aggregate_array(ar, window_size, function, nodata)
        out_ul_x = ul_x - ul_col * x_res # Readjust ul coords, which are offset by cells of the input raster
        out_ul_y = ul_y - ul_row * y_res
        out_x_res = x_res * window_size
        out_y_res = y_res * window_size
        out_tx = out_ul_x, out_x_res, x_rot, out_ul_y, y_rot, out_y_res
        
        _, file_ext = os.path.splitext(raster_path)
        tag = '_aggregated_{0}x{0}_{1}{2}'.format(window_size, function_string, file_ext)
        if '_1x1' in raster_path:
            out_path = raster_path.replace('_1x1' + file_ext, tag)
        else:
            out_path = raster_path.replace(file_ext, tag)
        if out_dir:
            out_path = os.path.join(out_dir, os.path.basename(out_path))
        '''if os.path.exists(out_path):
            raise IOError('Path exists: %s' % out_path)'''
        array_to_raster(ar_out, out_tx, prj, driver, out_path, dtype, nodata)
        
        desc = ('Aggregated raster using the {0} of window size of {1} x {1}.' +\
                'Nodata value for the input and output raster was {2}.')\
                .format(function_string, window_size, nodata)
        if mask_path:
            desc += ' Data were masked from %s with value %s.' % (mask_path, mask_val)
        createMetadata(sys.argv, out_path, description=desc)
    
    print '\nRuntime: %.1f seconds\n' % (time.time() - t0)
    

if __name__ == '__main__':
    
    sys.exit(main(*sys.argv[1:]))
    