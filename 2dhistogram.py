#!/usr/bin/env python
'''
\nMake 2D histograms and calculate agreement stats for two rasters.

Usage:
    2dhistogram.py <ref_path> <pred_path> <r_nodata> <p_nodata> <out_txt> [--pred_scale=<float>] [--ax_limit=<int>]
    2dhistogram.py -h | --help

Required parameters:
    
Options:
    -h --help     	     Show this screen.
    --pred_scale=<float>    Scaling factor for the prediction map (float)
    --ax_limit=<int>        limit of x and y axes. Default is max of ref_path or pred_path
'''

import os 
import sys
import time
import docopt
import math
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import seaborn as sns
from osgeo import gdal
from sklearn import metrics
from scipy import stats
from glob import glob
from scipy import odr

from lthacks import createMetadata


def calc_rmse(x, y):
    #import pdb; pdb.set_trace()
    rmse = np.sqrt((((x - y)) ** 2).mean())
    
    return rmse
    

def calc_rmspe(x, y):
    ''' Return root mean square percentage error of y with respect to x'''
    rmspe = np.sqrt(((100 * (x - y)/x) ** 2).mean())
    
    return rmspe


def calc_gmfr_coefficients(x, y):
    ''' Return the coefficents from a geometric mean function relationship 
    equation (i.e., a and b from y = bx + a)
    '''
    r , _ = stats.pearsonr(x, y)
    mean_x = x.mean()
    mean_y = y.mean()
    b = (np.sum((y - mean_y)**2)/np.sum((x - mean_x)**2)) ** (.5)
    a = mean_y - b * mean_x
    
    return a, b


def r2_from_coefficients(x, y, slope, intercept, adjusted=True):

    ss_tot = np.sum((y - y.mean())**2) # Sum of squares
    y_pred = x * slope + intercept # predicted y
    ss_res = np.sum((y - y_pred)**2) # Sum of squares of residuals
    ss_reg = ss_tot - ss_res
    r2 = 1 - ss_res/ss_tot

    if adjusted:
        ''' 
                                     (n - 1)
        adjusted r2 = 1 - (1 - r2) ____________
                                    n - p - 1
        '''
        n_predictors = 1
        n_samples = x.size
        df_reg = n_samples - 1
        df_res = n_samples - n_predictors - 1
        res_variance = ss_res/df_reg
        tot_variance = ss_tot/df_res
        r2 = 1 - (res_variance/tot_variance)

    return r2
    
    
def calc_orthog_regression(obs, pred):
    # From Matt Gregory

    
    def f(b, x):
        return b[0]*x + b[1]
    try:
        # Get slope and intercept from linear regression
        ols_slope, ols_inter = stats.linregress(obs, pred)[:2] # First 2 return are slope and 
        linear = odr.Model(f)
        data = odr.RealData(obs, pred)
        odr_model = odr.ODR(data, linear, beta0=[ols_slope, ols_inter])
        regression = odr_model.run()
    except Exception as e: 
        raise RuntimeError("Error in calc_orthog_regression: %s" % e)
        return None
    
    #calculate the RMSE by comparing 
    newpred = obs * regression.beta[0] + regression.beta[1]
    rmse = np.sqrt(((pred - newpred)**2).mean())
        
    return regression, rmse


def calc_stats(ar_r, ar_p):
    
    rmse = calc_rmse(ar_r, ar_p)
    nrmse = rmse/(ar_r.max() - ar_r.min())
    gmfr_a, gmfr_b = calc_gmfr_coefficients(ar_r, ar_p)
    #r2 = r2_from_coefficients(ar_r, ar_p, gmfr_b, gmfr_a)
    
    # Othogonal distance regression
    odr_reg, odr_rmse = calc_orthog_regression(ar_r, ar_p)
    odr_slope = odr_reg.beta[0] #pretty sure this is the slope
    odr_inter = odr_reg.beta[1] #pretty sure this is the intercept
    odr_ss_res = odr_reg.sum_square
    odr_ss_tot = np.sum((ar_p - ar_p.mean())**2)
    r2 = 1 - odr_ss_res/odr_ss_tot
    
    r, _ = stats.pearsonr(ar_r, ar_p)
    
    
    return rmse, nrmse, r2, r, odr_inter, odr_slope #gmfr_a, gmfr_b
    

def histogram_2d(r_samples, p_samples, out_png, bins=50, title=None, cmap=matplotlib.cm.gray, hexplot=False, vmax=None, xlabel=None, ylabel=None, norm=None):
    #print 'Plotting 2D histogram...'
    t0 = time.time()
    sns.axes_style('dark', rc={'axes.facecolor': 'white', 'axes.linewidth': 1})
    
    # Split plot into 22 subplots and plot hist in all but rightmost 5.
    #  This is to make sure the enlarged colorbar text doesn't overlap the plot.
    n_sub_cols = 22
    n_sub_rows = 1    
    max_val = max(r_samples.max(), p_samples.max())
    with sns.axes_style('white'):#, rc={'axes.facecolor': 'white', 'axes.linewidth': 1}):
        ax = plt.subplot2grid((n_sub_rows, n_sub_cols), (0, 0), colspan=n_sub_cols - 5)

    clip = False
    if vmax:
        clip = True
    else:
        vmax = max([r_samples.max(), p_samples.max()])
    if hexplot:
        if type(bins) != int: 
            print 'WARNING: bins given is not an integer, setting to default 100 equally sized bins...'
        plt.hexbin(r_samples, p_samples, gridsize=bins, norm=colors.LogNorm(), cmap=cmap)#vmax=vmax, clip=clip), cmap=cmap)
    else:
        #norm = colors.PowerNorm(2, vmin=1, clip=True)
        step = int(np.ceil(max_val / float(bins)))
        bins = np.arange(p_samples.min(), max_val + step, step)
        counts, xedges, yedges, img = ax.hist2d(r_samples, p_samples, bins=bins, cmap=cmap, norm=colors.LogNorm())#'''
                # Add minor axes and 1:1 line in gray
        #plt.plot([0, max_val], [0, max_val], '--', lw=2, color='white', alpha=0.5)
        plt.plot([0, max_val], [0, max_val], '--', lw=1, color='k', alpha=0.5)
        plot_min, plot_max = plt.xlim()
        #plot_max =+ 1
        #import pdb; pdb.set_trace()
    
    if xlabel: plt.xlabel(xlabel)
    if ylabel: plt.ylabel(ylabel)

    # PLot color bar without a border and in the rightmost subplot
    with sns.axes_style('dark', rc={'axes.linewidth':0}):
        c_ax = plt.subplot2grid((n_sub_rows, n_sub_cols), (0, n_sub_cols - 1))
        plt.colorbar(mappable=img, cax=c_ax, ticklocation='right')#, pad=-1)
    plt.tick_params(right=False)
    plt.tick_params(axis='y', which='minor', color='none')
    if title: plt.title(title)
    
    sns.despine()
    plt.savefig(out_png, dpi=300)
    
    return ax



def hist2d(ar_r, ar_p, out_png, nbins=100, cmap='gray_r', title=None, xlabel=None, ylabel=None, norm=None):
    
    sns.set_style('white')
    
    limit = max(ar_r.max(), ar_p.max())
    hist, bins, _ = np.histogram2d(ar_r, ar_p, bins=nbins, range=[[0, limit], [0, limit]])
    hist = hist.T[::-1, :]
    mask = hist == 0
    sns.heatmap(hist, cmap=cmap, mask=mask, xticklabels=False, yticklabels=False)
    #import pdb; pdb.set_trace()
    plt.title(title)
    if xlabel: plt.xlabel(xlabel)
    else: plt.xlabel('Lidar')
    if ylabel: plt.ylabel(ylabel)
    else: plt.ylabel('Landtrendr')
    
    labels = np.arange(0, limit, 500)
    tick_positions = labels/float(limit) * nbins
    #tick_positions = range(0, nbins, 10)
    plt.xticks(tick_positions, labels)
    plt.yticks(tick_positions, labels)#'''
    
    #plt.xlim(300/float(limit) * nbins, nbins)
    #plt.ylim(300/float(limit) * nbins, nbins)
    
    sns.despine()
    plt.savefig(out_png, dpi=300)
    
    return hist


def main(ref_path, pred_path, r_nodata, p_nodata, out_txt, pred_scale=1, ax_limit=1000):

    sns.set_style('white')
    sns.set_context(context='paper', rc={'patch.linewidth': 0})
    
    t0 = time.time()
    r_nodata = int(r_nodata)
    p_nodata = int(p_nodata)

    out_dir, basename = os.path.split(out_txt)
    t1 = time.time()
    
    # Find and read reference raster
    if not os.path.exists(ref_path):
        raise RuntimeError('ref_path does not exist: %s' % ref_path)
    ds_r = gdal.Open(ref_path)
    ar_r = ds_r.ReadAsArray()
    tx_r = ds_r.GetGeoTransform()
    prj_r = ds_r.GetProjection()
    ds_r = None
    
    # Find and read pred raster
    if not os.path.exists(pred_path):
        raise RuntimeError('pred_path does not exist: %s' % pred_path)
    ds_p = gdal.Open(pred_path)
    ar_p = ds_p.ReadAsArray()
    tx_p = ds_p.GetGeoTransform()
    prj_p = ds_p.GetProjection()
    ds_p = None
    
    if not tx_p == tx_r and prj_p == prj_r:
        raise ValueError('Geo transform and/or projection of reference and prediction rasters do not match.')

    mask = (ar_r != r_nodata) & (ar_p != p_nodata)
    #print ar_p.min()
    ar_r = ar_r[mask].astype(np.int32)
    ar_p = ar_p[mask].astype(np.int32)
    if 'ltbiomass' in ref_path:
         ar_r = ar_r * float(pred_scale)
    else:
         ar_p = ar_p * float(pred_scale)
    #import pdb; pdb.set_trace()
    
    # Calc stats
    rmse, rmspe, r2, r, gmfr_a, gmfr_b = calc_stats(ar_r, ar_p)
    stats = {'n_pixels': ar_p.size,
                  'rmse': rmse,
                  'r2': r2,
                  'pearsonr': r,
                  'odr_intercept': gmfr_a,
                  'odr_slope': gmfr_b
                  }
    
    # Make 2D histograms
    xlabel = 'reference'
    ylabel = 'predicted'
    this_bn = '%s_%s_vs_%s.png' % (basename.replace('.txt', ''), os.path.basename(ref_path), os.path.basename(pred_path))
    title = this_bn.replace('_vs_', ' vs ').replace('.png','')
    out_png = os.path.join(out_dir, this_bn)
    ax = histogram_2d(ar_r, ar_p, out_png, hexplot=False, cmap='plasma', xlabel=xlabel, ylabel=ylabel, bins=50)
    ax_limit = max(max(ax.get_xlim(), ax_limit))
    plt.sca(ax)
    
    # Plot GMFR (RMA) regression line
    max_val = max(ar_r.max(), ar_p.max())
    x = np.array([0, max_val + 100])
    y = x * gmfr_b + gmfr_a
    plt.plot(x, y, '-', lw=2, color='k')
    label_text = '$r^2$ = %.3f' %  r2

    plt.suptitle(title)
    plt.title(label_text, fontsize=12)
    
    #set plotting limits. 
    plt.ylim((0,ax_limit))
    plt.xlim((0,ax_limit))
    
    plt.savefig(out_png, dpi=300)
         
    plt.clf()
    
    df_xy = pd.DataFrame({'id': np.arange(ar_r.size),
                          'landtrendr': ar_p.astype(np.int16),
                          'lidar': ar_r.astype(np.int16)
                          })
    df_xy.set_index('id', inplace=True)
    df_xy.to_csv(out_png.replace('.png', '_xy.txt'))
    
    desc = '2D histograms made with the following parameters:\n'
    desc += '\tref_path: %s\n' % ref_path
    desc += '\tpred_path: %s\n' % pred_path
    desc += '\tr_nodata: %s\n' % r_nodata
    desc += '\tp_nodata: %s\n' % p_nodata
    desc += '\tout_txt: %s\n' % out_txt.replace('_stats.txt', '.txt')
    desc += '\tpred_scale: %s\n' % pred_scale
    desc += '\nStats for this comparison:'
    for k in sorted(stats.keys()):
        stat_str = '%s: %s' % (k, stats[k])
        desc += '\n\t' + stat_str
        print stat_str
    #desc += '\n\t'.join(['%s: %s' % (k, stats[k]) for k in sorted(stats.keys())])

    createMetadata(sys.argv, out_txt, description=desc)
    
    print '\nText file written to', out_txt
    print 'Total time: %.1f minutes' % ((time.time() - t0)/60)


if __name__ == '__main__':
    
    sys.exit(main(*sys.argv[1:]))
    
    
    
    