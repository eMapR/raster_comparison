# -*- coding: utf-8 -*-
"""
Usage:
    aggregate_raster.py <raster_path> <window_sizes> <function_string> [--nodata=<int>] [--out_dir=<str>] [--mask_path=<str>] [--mask_val=<int>]

    example: python aggregate_raster.py /path/to/input_raster.py 2 nanmean -9999
    

    Mutliple window sizes can be set with a comma-separated list.
    example: python aggregate_raster.py /path/to/input_raster.py '2,3,4,5' nanmean -9999

    To mask the input raster before aggregating, you can specify a mask_path and mask_val(defaults to 0)

Required parameters:
    raster_path     path to raster to aggregate
    window_sizes    single int or comma-separate list of ints specifying number
                    of pixels to aggregate
    function_string string indicating what statistic to use to calculate aggregations.
                    Valid options for `function_string' are 'nanmean', 'nansum', and 'mode'.

Options:
    nodata=<int>    integer nodata value of raster_path
    out_dir=<str>   alternative output directory. Default is directory of raster_path
    mask_path=<str> path of raster to use as mask before aggregating. Must have same geotransform and projection as raster_path
    mask_val=<int>  int specifying which pixels to use as mask from mask_path

"""

import sys
import os
import time
import gdal
import math
import warnings
import pandas as pd
import numpy as np
from sklearn import metrics

from lthacks import createMetadata
from lthacks.stats_functions import get_function
from lthacks.intersectMask import array_to_raster

FUNC_OPTIONS = ['mode',
                'nanmean',
                'nansum']

def aggregate_array(ar, window_size, function, nodata=None):

    
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
    if nodata is not None:
        ar[ar == nodata] = np.nan # Protect nodata values for calculations
    ar_buf[ul_row : ul_row + data_nrows, ul_col : ul_col + data_ncols] = ar # Center data

    # Aggregate
    out_nrows = buf_nrows/window_size
    out_ncols = buf_ncols/window_size
    ar_out = function(ar_buf.reshape(out_nrows, window_size, out_ncols, window_size),
                      axis=(1,3))
    if nodata is not None:
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


def main(raster_path, window_sizes, function_string, nodata=None, out_dir=None, mask_path=None, mask_val=0):
       
    t0 = time.time()
    if not os.path.exists(raster_path):
        sys.exit('raster_path specified does not exist: ' + raster_path)
    
    window_sizes = [int(s) for s in window_sizes.split(',')]
    
    if function_string in FUNC_OPTIONS:
        function = get_function(function_string)
    else:
        raise ValueError('function_string not understood: %s. Valid options: %s' % (function_string, '\n\t'.join(FUNC_OPTIONS)))
    
    mask_val = int(mask_val)
    
    ds = gdal.Open(raster_path)
    ar = ds.ReadAsArray()
    n_bands = ds.RasterCount # Add fucntionality for multiple bands
    ul_x, x_res, x_rot, ul_y, y_rot, y_res = ds.GetGeoTransform()
    prj = ds.GetProjection()
    driver = ds.GetDriver()
    dtype = get_gdal_dtype(ds.GetRasterBand(1).DataType)
    if nodata is not None:
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
            if os.path.isdir(out_dir):
                out_path = os.path.join(out_dir, os.path.basename(out_path))
            else:
                os.mkdir(out_dir)

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
    