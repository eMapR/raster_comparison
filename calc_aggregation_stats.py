#!/usr/bin/env python
'''
\nGiven search strings for reference and prediction rasters, find all rasters 
that match, and calc stats and make 2D histograms showing agreement.

Usage:
  calc_aggregation_stats.py <ref_search_str>, <pred_search_str>, <r_nodata>, <p_nodata>, <agg_levels>, <out_txt>, [--pred_scale=<s>]
  calc_aggregation_stats.py -h | --help

Options:
  -h --help     	  Show this screen.
  --pred_scale=<s>  	  Scaling factor for the prediction map (float)
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
        mydata = odr.RealData(obs, pred)
        myodr=odr.ODR(mydata, linear, beta0=[ols_slope, ols_inter])
        myoutput=myodr.run()
    except: 
        print("Error in calc_orthog_regression")
        return None
    
    #calculate the RMSE by comparing 
    newpred = obs * myoutput.beta[0 ]+ myoutput.beta[1]
    rmse = np.sqrt(((pred-newpred)**2).mean())
        
    return myoutput, rmse


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
    #r2 = r2_from_coefficients(ar_r, ar_p, odr_slope, odr_inter)
    #import pdb; pdb.set_trace()
    
    # OLS regression
    #slope, intercept, r_value, p_value, std_err = stats.linregress(ar_r, ar_p)
    #R2 = r_value ** 2
    
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
        step = max_val / bins
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


def main(ref_search_str, pred_search_str, r_nodata, p_nodata, agg_levels, out_txt, pred_scale=1, cutoffs=None, ranges=None):

    sns.set_style('white')
    sns.set_context(context='paper', font_scale=1, rc={'patch.linewidth': 0})
    
    t0 = time.time()
    agg_levels = [l.strip() for l in agg_levels.split(',')]
    r_nodata = int(r_nodata)
    p_nodata = int(p_nodata)

    out_dir, basename = os.path.split(out_txt)
    agg_stats = []
    ref_paths = glob(ref_search_str)
    pred_paths = glob(pred_search_str)
    for l in agg_levels:
        t1 = time.time()
        print 'Calculating stats for agg. level:', l
        
        # Find and read reference raster
        #ref_str = ref_search_str.format(l)
        #ref_path = glob(ref_str)
        ref_path = [p for p in ref_paths if '%sx%s' % (l, l) in p]
        if len(ref_path) == 0:
            if l == 1:
                ref_path = [p for p in ref_paths if '%sx%s' % (l, l) not in p]
            if len(ref_path) == 0:
                raise IOError('Could not find files with reference search string %s and prediction level %s' % (ref_search_str, l))
        elif len(ref_path) > 1:
            raise IOError('Multiple files found with reference search string %s and prediction level %s: %s' % (ref_search_str, l, '\n'.join(ref_path)))
        ref_path = ref_path[0]
        ds_r = gdal.Open(ref_path)
        ar_r = ds_r.ReadAsArray()
        tx_r = ds_r.GetGeoTransform()
        prj_r = ds_r.GetProjection()
        ds_r = None
        
        # Find and read pred raster
        #pred_str = pred_search_str.format(l)
        #pred_path = glob(pred_str)
        pred_path = [p for p in pred_paths if '%sx%s' % (l, l) in p]
        if len(pred_path) == 0:
            if l == 1:
                pred_path = [p for p in pred_paths if '%sx%s' % (l, l) not in p]
            raise IOError('Could not find files in with prediction search string %s and prediction level %s' % (pred_search_str, l))
        elif len(pred_path) > 1:
            raise IOError('Multiple files found with prediction search string %s and prediction level %s: %s' % (pred_search_str, l, '\n'.join(pred_path)))
        pred_path = pred_path[0]
        ds_p = gdal.Open(pred_path)
        ar_p = ds_p.ReadAsArray()
        tx_p = ds_p.GetGeoTransform()
        prj_p = ds_p.GetProjection()
        ds_p = None
        
        if not tx_p == tx_r and prj_p == prj_r:
            raise ValueError('Geo transform and/or projection of reference and prediction rasters do not match.')
        try:
            mask = (ar_r != r_nodata) & (ar_p != p_nodata)
        except ValueError:
            raise ValueError('Ref. and pred. rasters different sizes')
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
        these_stats = {'agg_level': l,
                      'n_pixels': ar_p.size,
                      'rmse': rmse,
                      'r2': r2,
                      'pearsonr': r,
                      'odr_intercept': gmfr_a,
                      'odr_slope': gmfr_b
                      }
                
                
        print '%s seconds\n' % int(time.time() - t1)
        
        # Make 2D histograms
        xlabel = os.path.basename(ref_search_str).replace('*.tif','')
        ylabel = os.path.basename(pred_search_str).replace('*.tif','')#'''
        '''if 'ltbiomass' in pred_path:
            xlabel = 'lidar'
            ylabel = 'landtrendr'
        else:
            xlabel = 'landtrendr'
            ylabel = 'lidar'#'''
        this_bn = '%s_%s_vs_%s_agg%s.png' % (basename.replace('.txt', ''), xlabel, ylabel, l)
        title = (this_bn + ' ').replace('_', ' ').replace('.png', '')
        out_png = os.path.join(out_dir, this_bn)
        ax = histogram_2d(ar_r, ar_p, out_png, hexplot=False, cmap='plasma', xlabel=xlabel, ylabel=ylabel)
        plt.sca(ax)
        nbins = 50
        #if normalize.lower() == 'log': normalize = colors.LogNorm()
        #hist2d(ar_p, ar_r, out_png, nbins=nbins, title=title, cmap='plasma', norm=normalize)
        
        # Plot GMFR (RMA) regression line
        #if not cutoffs:
        max_val = max(ar_r.max(), ar_p.max())
        x = np.array([0, max_val + 100])
        y = x * gmfr_b + gmfr_a
        #plt.plot(x, y, '-', lw=3, color='white', alpha=0.5)
        plt.plot(x, y, '-', lw=2, color='k')
        label_text = '$r^2$ = %.3f' %  r2
        '''label_x = max_val * .75 - 25
        label_y = label_x * gmfr_b + gmfr_a + 25
        text_position = np.array([label_x, label_y]).reshape(1,2)
        label_angle = (np.arctan(gmfr_b) / math.pi) * 180
        plot_angle = ax.transData.transform_angles(np.array((label_angle,)), text_position)[0]
        ax.text(label_x,
                 label_y, 
                 label_text, 
                 fontsize=8, 
                 ha='center', 
                 rotation_mode='anchor',
                 rotation=plot_angle)#'''
        plt.title(label_text, fontsize=12)
        #set plotting limits. 
        plt.ylim((0,1000))
        plt.xlim((0,1000))
        
        # Plot minor axes in light gray. Not sure if there's a mor legit way
        #   to plot axes with different colors
        '''plot_min, plot_max = plt.xlim()
        plot_max += plot_max * 0.003
        plt.plot([0, plot_max], [plot_max, plot_max], '-', lw=4, color='0.8')
        plt.plot([plot_max, plot_max], [0, plot_max], '-', lw=4, color='0.8')#'''
        plt.savefig(out_png, dpi=300)
        
        if cutoffs:
            ranges = [[int(l) for l in rge.split(',')] for rge in cutoffs.split(';')]
            this_max = max([ar_r.max(), ar_p.max()])
            this_min = min([ar_r.min(), ar_p.min()])
            #fig = plt.figure()
            #ax.text(this_max/2, this_max + this_max * .05, 'Overall $r^2$: %.3f' % r2, fontsize=10, ha='center')
            #plot_scale = float(nbins)/this_max
            for i, (lower, upper) in enumerate(ranges):
                this_mask = (ar_r >= lower) & (ar_r <= upper) & (ar_p >= lower) & (ar_p <= upper)
                this_r = ar_r[this_mask]
                this_p = ar_p[this_mask]
                
                # Calc additional stats
                rmse, rmspe, r2, r, gmfr_a, gmfr_b = calc_stats(this_r, this_p)
                these_stats['rmse_%s-%s' % (lower, upper)] = rmse
                these_stats['pearsonr_%s-%s' % (lower, upper)] = r
                these_stats['odr_intercept_%s-%s' % (lower, upper)] = gmfr_a
                these_stats['odr_slope_%s-%s' % (lower, upper)] = gmfr_b
                these_stats['r2_%s-%s' % (lower, upper)] = r2

                # Plot this line on the histogram
                x = np.array([lower, upper])
                if i == 1:
                    x = np.array([lower, upper + 100])
                y = gmfr_b * x + gmfr_a
                #x = x * plot_scale
                #y2 = gmfr_b * upper + gmfr_a * float(nbins)/this_max
                #print y1, y2
                #ax = histogram_2d(this_r, this_p, out_png, hexplot=False, cmap='plasma', xlabel=xlabel, ylabel=ylabel)
                plt.plot(x, y, '-', color='k', alpha=.5) # x and y switched because the predicted is on x axis
                # plot the rest of the line dotted and transparent
                if lower <= this_min:
                    x1 = np.arange(upper, this_max + 1, 10, dtype=np.int16)
                else:
                    x1 = np.arange(this_min, lower + 1, 10, dtype=np.int16)
                y1 = (gmfr_b * x1 + gmfr_a)# * plot_scale
                #x1 = x1 * plot_scale
                #plt.plot(x1, y1, '--', color='k', alpha=0.4)
                
                # Plot the title again with the new r2 value appended
                addtl_title = '$r^2$ = %.3f' %  r2
                '''label_x = (x.min() + (x.max() - x.min())) * .75 - 25
                label_y = label_x * gmfr_b + gmfr_a + 25
                text_position = np.array([label_x, label_y]).reshape(1,2)
                label_angle = (np.arctan(gmfr_b) / math.pi) * 180
                plot_angle = ax.transData.transform_angles(np.array((label_angle,)), text_position)[0]
                ax.text(label_x,
                         label_y, 
                         addtl_title, 
                         fontsize=8, 
                         ha='center', 
                         rotation_mode='anchor',
                         rotation=plot_angle)#'''
                label_text += ',  seg%s %s' % (i + 1, addtl_title)
                plt.title(label_text, fontsize=10)
            
            # PLot axes minor axes again since the other ones got written over
            '''plot_min, plot_max = plt.xlim()
            plot_max += plot_max * 0.003
            plt.plot([0, plot_max], [plot_max, plot_max], '-', lw=3, color='0.8')
            plt.plot([plot_max, plot_max], [0, plot_max], '-', lw=3, color='0.8')#'''
            
            #set plotting limits. 
            plt.ylim((0,1000))
            plt.xlim((0,1000))
            
            plt.savefig(out_png, dpi=300)
        agg_stats.append(these_stats)        
        plt.clf()
        
        df_xy = pd.DataFrame({'id': np.arange(ar_r.size),
                              'landtrendr': ar_p.astype(np.int16),
                              'lidar': ar_r.astype(np.int16)
                              })
        df_xy.set_index('id', inplace=True)
        df_xy.to_csv(out_png.replace('.png', '_xy.txt'))
                
    
    out_txt = out_txt.replace('.txt', '_%s_vs_%s_stats.txt' % (xlabel, ylabel))
    df_stats = pd.DataFrame(agg_stats)
    #columns = ['agg_level', 'n', 'rmse', 'nrmse', 'r2', 'pearsonr', 'gmfr_a', 'gmfr_b']
    df_stats.to_csv(out_txt, sep='\t', index=False)
    
    desc = 'Text file of stats and 2D histograms made with the following parameters:\n'
    desc += '\tref_search_str: %s\n' % ref_search_str
    desc += '\tpred_search_str: %s\n' % pred_search_str
    desc += '\tr_nodata: %s\n' % r_nodata
    desc += '\tp_nodata: %s\n' % p_nodata
    desc += '\tagg_levels: %s\n' % ','.join(agg_levels)
    desc += '\tout_txt: %s\n' % out_txt.replace('_stats.txt', '.txt')
    desc += '\tpred_scale: %s\n' % pred_scale
    if ranges:
        desc +='\tcutoffs: %s\n' % cutoffs
    createMetadata(sys.argv, out_txt, description=desc)
    
    print df_stats
    print '\nText file written to', out_txt
    print 'Total time: %.1f minutes' % ((time.time() - t0)/60)


if __name__ == '__main__':

    sys.exit(main(*sys.argv[1:]))
    
    
    
    