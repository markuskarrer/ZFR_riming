'''
this script is meant to plot the observations in a specific time slot in order to compare the McSnow simulations to that. This plots (pol-) moments and (pol-)spectra
'''

from IPython.core.debugger import set_trace

import numpy as np
import xarray as xr
import glob 
import pandas as pd
import matplotlib.pyplot as plt
import plotRoutines as plotRout
import matplotlib as mpl
import sys
import os
mpl.use("Agg")

import argparse
parser =  argparse.ArgumentParser(description='plot melting statistics')
parser.add_argument('-d','--date', nargs=1,help="format yyyymmdd")
parser.add_argument('-st','--starttime', nargs=1,help="format HH:MM",default=["00:00"])
parser.add_argument('-sp','--allSpectra', nargs=1,help="plot all spectra (True:1,False:0)",default=["0"])
parser.add_argument('-SiS','--SimpleSpectra', nargs=1,help="plot simple spectra (Ka-Band and W to illustrate w-estimate) spectra (True:1,False:0)",default=["0"])
parser.add_argument('-et','--endtime', nargs=1,help="format HH:MM",default=["23:59"])
parser.add_argument('-ts','--timeseries', nargs=1,help="plot time-height series",default=["0"])
parser.add_argument('-nmn','--noMieNotch', nargs=1,help="dont plot the mie-notch (takes time)",default=["0"])
args        = parser.parse_args()
date        = args.date[0]
st          = args.starttime[0]
plot_all_spectra = args.allSpectra[0]=="1"
plot_simple_Spectra = args.SimpleSpectra[0]=="1"
plot_timeseries = args.timeseries[0]=="1"
et          = args.endtime[0]
noMieNotch  = args.noMieNotch[0]=="1"

dateStartSpec = pd.to_datetime(date+' ' + st); dateEndSpec = pd.to_datetime(date+' ' + et)
dateStart = pd.to_datetime(date+' ' + st); dateEnd = pd.to_datetime(date+' ' + et)
    
dataLV2Path = '/data/obs/campaigns/tripex-pol/processed/tripex_pol_level_2/'
dataPolPath = '/data/obs/campaigns/tripex-pol/processed/tripex_pol_level_1/'
dataLV0Path     = '/data/obs/campaigns/tripex-pol/processed/tripex_pol_level_0/'
dataPeaks       = '/data/optimice/tripex-pol/peaks/resampledMerged'
dataEdge        = '/data/obs/campaigns/tripex-pol/spectralEdges/'

def getLdr(date):

    date = pd.to_datetime(date)
    dataPath = '/data/data_hatpro/jue/cloudnet/juelich/processed/categorize/{0}'.format(date.strftime('%Y'))
    fileID = '{selDay}{content}{ext}'
    fileName = fileID.format(selDay=date.strftime('%Y%m%d'), content='_juelich_categorize', ext='.nc')
    filePath = ('/').join([dataPath, fileName])
    
    data = xr.open_dataset(filePath)
    alt = data.altitude.copy() 
    ldr = data.ldr.copy()
    
    data.close()
    return ldr

# read in files 
dataLV2 = xr.open_dataset(dataLV2Path+date+'_tripex_pol_3fr_L2_mom.nc') # LV2 moments
if not (dataLV2.Ka_DBZ_H>10).any():
    print("all Ka<10dB on ",date) 
    sys.exit()

dataLV0List = glob.glob(dataLV0Path+dateStartSpec.strftime('%Y')+'/'+dateStartSpec.strftime('%m')+'/'+dateStartSpec.strftime('%d')+'/'+date+'*_tripex_pol_3fr_spec_filtered_regridded.nc') # LV0, so spectra
dataLV0 = xr.open_mfdataset(dataLV0List[int(dateStartSpec.strftime('%H')):int(dateEndSpec.strftime('%H'))+1])
dataLV0.time.attrs['units']='seconds since 1970-01-01 00:00:00 UTC' # time attribute is weird, and python does not recognize it
LDR = getLdr(date)
LDR["height"] = LDR["height"] - 111 #alt is the height of the radar

#peaks in spectra 
peaks_file = ('_').join([dateStart.strftime('%Y%m%d'),'peaks_joyrad35.nc'])
filePath = os.path.join(dataPeaks, peaks_file)
Peaks = xr.open_dataset(filePath)

#peaks in spectra 
edges_file = 'spectMaxMinVelTKa_' + dateStart.strftime('%Y%m%d') + '.nc'
filePath = os.path.join(dataEdge, edges_file)
Edges = xr.open_dataset(filePath)

dataLV0 = xr.decode_cf(dataLV0)

#####################
## slice data
####################
#polMean = dataPol.sel(time=slice(dateStart,dateEnd)).resample(time='30min').mean()
if plot_all_spectra or plot_timeseries or plot_simple_Spectra:
    LV0 = dataLV0.sel(time=slice(dateStartSpec,dateEndSpec)).resample(time='4s').nearest(tolerance='2s') #spectra #plot all times
    LV2 = dataLV2.sel(time=slice(dateStart,dateEnd)).resample(time='4s').nearest(tolerance='2s') #nearest(tolerance='3min') #moments
    Peaks = Peaks.sel(time=slice(dateStart,dateEnd)).resample(time='4s').nearest(tolerance='2s') 
    Edges = Edges.sel(time=slice(dateStart,dateEnd)).resample(time='4s').nearest(tolerance='2s') 
    LDR = LDR.rename({'height': 'range'}) #use same name for height variable
    LDR = LDR.sel(time=slice(dateStart,dateEnd))
    LDR = LDR.interp(range=LV2.range.values)
    LDR = LDR.interp(time=LV2.time.values).resample(time='4s').nearest(tolerance='2s') #.nearest(tolerance='3min')
else:
    LV0 = dataLV0.sel(time=slice(dateStartSpec,dateEndSpec)).resample(time='5min').nearest(tolerance='1min') #spectra
    LV2 = dataLV2.sel(time=slice(dateStart,dateEnd)).resample(time='5min').mean() #nearest(tolerance='3min') #moments
    Peaks = Peaks.sel(time=slice(dateStart,dateEnd))
    Edges = Edges.sel(time=slice(dateStart,dateEnd))
    LDR = LDR.rename({'height': 'range'}) #use same name for height variable
    LDR = LDR.sel(time=slice(dateStart,dateEnd))
    LDR = LDR.interp(range=LV2.range.values)
    LDR = LDR.interp(time=LV2.time.values).resample(time='5min').mean() #.nearest(tolerance='3min')
if plot_timeseries:
    ####################
    # plot moments timeseries
    ####################
    #plotRout.plotMomentsObs(LDR,dataLV2.sel(time=slice(dateStart,dateEnd)),'plots/timeheight_',dateStart.strftime('%Y%m%d_%H%M%S')+'_'+dateEnd.strftime('%H%M%S')+'_moments',date_str=date)
    plotRout.plotMomentsObs4paper(LDR,dataLV2.sel(time=slice(dateStart,dateEnd)),'plots/timeheight_',dateStart.strftime('%Y%m%d_%H%M%S')+'_'+dateEnd.strftime('%H%M%S')+'_moments',date_str=date)
else:
    #############################
    # plot spectra and profiles:
    # plot a spectra every 5 minutes, otherwise it takes too long 
    # plot profiles of the observations, but 5 minute mean to get rid of KDP uncertainty
    ############################
    if plot_all_spectra:
        plotRout.plotProfilesAndSpectraObs(LDR,LV2,LV0,Peaks,Edges,'/net/morget/melting_plots/spectra/',plot_all_times=True,noMieNotch=noMieNotch)
    elif plot_simple_Spectra:
        plotRout.plotSimpleSpectra(LDR,LV2,LV0,Peaks,Edges,'plots/spectra/',plot_all_times=True,noMieNotch=noMieNotch)
    else:
        plotRout.plotProfilesAndSpectraObs(LDR,LV2,LV0,Peaks,Edges,'plots/spectra/')

    
    
