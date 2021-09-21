#!/usr/bin/env python
# coding: utf-8

from functions import obs_functions as fc #jupyter notebook code
import glob as gb
import pandas as pd
import numpy as np
import os
import xarray as xr
import argparse
import re
import pickle
import matplotlib.pyplot as plt
import turbo_colormap_mpl
turbo_colormap_mpl.register_turbo()

from IPython.core.debugger import set_trace
import matplotlib as mpl
mpl.use('Agg')

mpl.rcParams['legend.fancybox'] = True
mpl.rcParams['legend.framealpha'] = 0.7
mpl.rcParams['legend.facecolor']='silver'
mpl.rcParams['legend.frameon']=True

mpl.rcParams['lines.linewidth']=5

def get_save_string(onlydate,filterZeflux=None,RHbt=None,MDVtLT=None,MDVtBT=None):


    if filterZeflux is None:
        Zefluxadd_spec  = ""
    else:
        Zefluxadd_spec  = "_Zeflux" + str(filterZeflux)
    if RHbt is None:
        RHadd_spec  = ""
    else:
        RHadd_spec  = "_RHbt" + str(RHbt)
    if MDVtLT is None:
        MDVtLT_spec  = ""
    else:
        MDVtLT_spec  = "_MDVtLT" + str(int(MDVtLT*10))
    if MDVtBT is None:
        MDVtBT_spec  = ""
    else:
        MDVtBT_spec  = "_MDVtBT" + str(int(MDVtBT*10))

    save_spec   = "obs_" + RHadd_spec +Zefluxadd_spec + MDVtLT_spec + MDVtBT_spec + "_" + onlydate

    return save_spec

def categorize_part_type(var,MDVxT,DWRxkT,var_coll,varKey,i,clustering_char):

    ####classify according to MDV and DWR
    DWR_MDV_PRb1_dict = dict()
    DWR_MDV_PRb1_dict["unrimed-transitional"] = [0.6,7.3] #fit to rime fraction [0,0.2] and 1mm/h<RR<4mm/h # separates unrimed from transitional
    DWR_MDV_PRb1_dict["transitional-rimed"]   = [0.75,2.58] #fit to rime fraction [0,0.2] and 1mm/h<RR<4mm/h # separates transitional from rimed
    for key in ["unrimed","transitional","rimed"]:
        if key=="unrimed":
            key_thres = "unrimed-transitional" 
            var_coll[clustering_char+str(i)+key+varKey] = var.where(DWRxkT>DWR_MDV_PRb1_dict[key_thres][0]*MDVxT**DWR_MDV_PRb1_dict[key_thres][1]).copy()
        elif key=="transitional":
            key_thres = "transitional-rimed" 
            key_thres = "unrimed-transitional" 
            tmp = var.where(DWRxkT<DWR_MDV_PRb1_dict[key_thres][0]*MDVxT**DWR_MDV_PRb1_dict[key_thres][1]).copy()
            key_thres = "transitional-rimed" 
            var_coll[clustering_char + str(i) + key + varKey] = tmp.where(DWRxkT>DWR_MDV_PRb1_dict[key_thres][0]*MDVxT**DWR_MDV_PRb1_dict[key_thres][1]).copy()
        elif key=="rimed":
            key_thres = "transitional-rimed" 
            var_coll[clustering_char + str(i) + key + varKey] = var.where(DWRxkT<DWR_MDV_PRb1_dict[key_thres][0]*MDVxT**DWR_MDV_PRb1_dict[key_thres][1]).copy()
    accum = 0
    for key in ["unrimed","transitional","rimed"]:
        var_coll["Ndata_" + clustering_char + str(i) + key]  = np.sum(~np.isnan(var_coll[clustering_char + str(i) + key + varKey].data)) #fraction of .. particles
        accum += var_coll["Ndata_" + clustering_char + str(i) + key]
    for key in ["unrimed","transitional","rimed"]:
        var_coll[clustering_char + str(i) + key + "_perc"]  = var_coll["Ndata_" + clustering_char + str(i) + key]/accum #fraction of .. particles
    N_data = accum
    N_data_hour = N_data*4/3600 #number of data points in hours
    var_coll[clustering_char + str(i) + "Ndata"] = accum
    var_coll[clustering_char + str(i) + "Hdata"] = N_data_hour

    return var_coll

def get_observed_melting_properties(onlydate="",av_min=0,calc_or_load=1,no_mie_notch=False,profiles=False):

    dataPath = '/data/obs/campaigns/tripex-pol/processed/tripex_pol_level_2/'
    dataEdge        = '/data/obs/campaigns/tripex-pol/spectralEdges/'
    dataPeaks       = '/data/optimice/tripex-pol/peaks/resampledMerged/' 
    dataLV0         = '/data/obs/campaigns/tripex-pol/processed/tripex_pol_level_0/'
    if isinstance(onlydate,str):
        if onlydate!="":
            if int(onlydate)<20160103:
                dataPath = '/data/optimice/tripex/tripex_level_02_X_pol/' 
                #get list of file names
                file_pre = dataPath + '*' + onlydate + '*'
                input("ask Jose how to merge both datasets")
            else:
                #get list of file names
                file_pre        = dataPath + onlydate + '*'
                file_preEdge    = dataEdge + 'spectMaxMinVelTKa_' + onlydate + '.nc' 
                file_prePeaks   = dataPeaks + onlydate + '_peaks_joyrad35.nc' 
        else:
            #get list of file names
            file_pre = dataPath + onlydate + '*'
            file_preEdge = dataEdge + 'spectMaxMinVelTKa_*.nc'
            file_prePeaks   = dataPeaks + '_peaks_joyrad35.nc' 
        fileList = sorted(gb.glob(file_pre+'*nc'))
        #edges_file = 'spectMaxMinVelTKa_' + onlydate + '.nc'
        #fileListEdge = sorted(gb.glob(file_preEdge))
            
    else:
        fileList = []
        fileListEdge = []
        for date in onlydate:
            if date=="20190106":
                continue
            if date=="20190220":
                continue
            file_pre = dataPath + date + '*'
            fileList.extend(gb.glob(file_pre+'*nc'))

    maskR = int('0000000010000000',2) #rain
    maskL = int('0000000001000000',2) #lwp
    maskP = int('1000000000000000',2) #n points
    maskC = int('0100000000000000',2) # correl
    maskV = int('0010000000000000',2) # Variance

    #read data
    results         = xr.Dataset()
    results_at_temp = xr.Dataset()
    for fileName in fileList:
        date_now = re.split("_",re.split("level_2/*",fileName)[1])[0]

        if date_now=="20190220":
            continue
        #if date_now.startswith("2019") or date_now.startswith("201811"):
        #    continue
        #if date_now.startswith("2018112"):
        #    continue
        fileNameEdge = dataEdge + 'spectMaxMinVelTKa_' + date_now + '.nc'
        fileNamePeaks = dataPeaks +  date_now + '_peaks_joyrad35.nc'
        year = date_now[0:4]
        month = date_now[4:6]
        day = date_now[6:8]
        fileNameLV0 = dataLV0 + year + "/" + month + "/" + day + "/" + date_now + "*_tripex_pol_3fr_spec_filtered_regridded.nc"

        if profiles:
            print("load prof-obs ",date_now)
            tmpData = processOneDayProfiles(fileName,fileNameEdge,fileNamePeaks,fileNameLV0, maskR, maskL, 
                                    maskP, maskC ,maskV,calc_or_load,no_mie_notch=no_mie_notch)
        else:
            print("load obs ",date_now)
            tmpData = processOneDay(fileName,fileNameEdge,fileNamePeaks,fileNameLV0, maskR, maskL, 
                                    maskP, maskC ,maskV,calc_or_load,no_mie_notch=no_mie_notch)
        
        results         = xr.merge([results, tmpData])
    if profiles:
        return results

    ###derive more quantities
    ##apply density correction to MDV
    p0 = 1.01325e5
    results["MDVxT_denscorr"] = results["MDVxT"]*(p0/results["PaT"])**0.54 #density correction according to Heymsfield 2007
    results["MDVxB_denscorr"] = results["MDVxB"]*(p0/results["PaB"])**0.54
    ##calculate ZFR
    results["ZeXt"]        = results["ZeXt"].where(results["ZeXt"]>-50) 
    results["MDVvCorrT"] = results["MDVxT"] - results["w_estT"]
    results["MDVvCorrB"] = results["MDVxB"] - results["w_estB"]
    results["MDVvCorrB_denscorr"] = results["MDVvCorrB"]*(p0/results["PaB"])**0.54 #density correction according to Heymsfield 2007
    results["MDVvCorrT_denscorr"] = results["MDVvCorrT"]*(p0/results["PaT"])**0.54 #density correction according to Heymsfield 2007
    #results["ZFR"]       = results["MDVxT"]                       * results["ZeXt"] / (results["MDVxB"]                     * results["ZeXb"]) 
    #results["ZFRvCorrT"] = ((results["MDVxT"] - results["w_estT"]) * results["ZeXt"] / (results["MDVxB"]                     * results["ZeXb"])        ).copy()
    #results["ZFRvCorrB"] = ( results["MDVxT"]                     * results["ZeXt"] / ((results["MDVxB"]-results["w_estB"])  * results["ZeXb"])        ).copy()
        
    #results["ZFRvCorr"]  = ((results["MDVxT"] - results["w_estT"]) * results["ZeXt"] /((results["MDVxB"]-results["w_estB"]) * results["ZeXb"])        ).copy()
 
    ###NEW ZFR definition: 1 for 1to1 melting; higher for collision, lower for break-up
    #results["ZFR"]       = results["MDVxB"].copy()                       * results["ZeXb"].copy() / (results["MDVxT"].copy()                     * results["ZeXt"].copy()) * 0.23        

    results["ZFR"] = (results["MDVxB"] - 0.0) * results["ZeXb"] / (results["MDVxT"]                     * results["ZeXt"])  * 0.23
    results["ZFRvCorrB"] = (results["MDVxB"] - results["w_estB"]) * results["ZeXb"] / (results["MDVxT"]                     * results["ZeXt"])  * 0.23
    results["ZFRvCorrT"] =  results["MDVxB"]                     * results["ZeXb"] / ((results["MDVxT"]-results["w_estT"])  * results["ZeXt"])  * 0.23

    results["ZFRvCorr"]  = (results["MDVxB"] - results["w_estB"]) * results["ZeXb"] /((results["MDVxT"]-results["w_estT"]) * results["ZeXt"])  * 0.23

    results["Delta_ZF_upper"]           =  (results["ZfluxX_thirdML"]) / (-results["MDVxT"] * results["ZeXt"]) #increase of reflectivity flux in upper (half/two thirds/...)
    results["Delta_ZF_lower"]           =  ((-results["MDVxB"]*results["ZeXb"])- results["ZfluxX_thirdML"]) / (-results["MDVxT"] * results["ZeXt"]) #increase of reflectivity flux in upper (half/third/...)

    if av_min=="0":
        pass
    else:
        #results = results.rolling(min_periods=int(av_min*15/2),center=True,time=av_min*15).mean().copy() #moving average with <av_min>minute window
        if "s" in av_min:
            av_sec = int(av_min[:-1])
            #results = results.rolling(min_periods=int(av_sec/4),center=True,time=int(av_sec/4)).mean().copy() #min_periods=av_min/8 -> more than 50% in the averaging box must be non-nan
            results = results.rolling(min_periods=int(av_sec/4),center=True,time=int(av_sec/4)).quantile(0.9).copy() #min_periods=av_min/8 -> more than 50% in the averaging box must be non-nan
        else:
            results = results.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int(av_min)*15)).mean().copy() #min_periods=int(av_min*15/2)-> more than 50% in the averaging box must be non-nan

    return results

#def getParsivelRainRate(date):
#
#    dataPath = '/data/hatpro/jue/data/parsivel/netcdf/{0}'.format(date.strftime('%y%m'))
#    fileID = '{content}{selDay}{ext}'
#    fileName = fileID.format(selDay=date.strftime('%Y%m%d'), content='parsivel_jue_', ext='.nc')
#    filePath = ('/').join([dataPath, fileName])
#    
#    data = xr.open_dataset(filePath)
#    
#    rainRate = xr.DataArray(data.rain_rate.values, dims=('time'), coords={'time':data.time.values})    
#    
#    return rainRate

def getLdr(date): #,height=None,time=None):

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

def processOneDay(fileName,fileNameEdge,fileNamePeaks,fileNameLV0, maskR, maskL, 
                  maskP, maskC ,maskV,calc_or_load=True,no_mie_notch=False):
    
    date = pd.to_datetime(fileName.split('/')[-1].split('_')[0])
    date_str = pd.to_datetime(str(date)).strftime('%Y%m%d')

    if calc_or_load:
        data = xr.open_dataset(fileName)
        if os.path.isfile(fileNameEdge):
            dataEdge = xr.open_dataset(fileNameEdge)
        else:
            return xr.Dataset({})
        if os.path.isfile(fileNamePeaks):
            dataPeaks = xr.open_dataset(fileNamePeaks)
        else:
            return xr.Dataset({})
        dataSpec = xr.open_mfdataset(fileNameLV0)
        dataSpec = xr.decode_cf(dataSpec) #Decode the given Dataset or Datastore according to CF conventions into a new Dataset.
       
        #bring time to same standard and values as LV2 data 
        dataSpec.time.attrs['units']='seconds since 1970-01-01 00:00:00 UTC'
        specW   = 10*np.log10(dataSpec.WSpecH).copy()
        specW["time"] = pd.to_datetime(specW.time.values*1e9)
        specW.resample(time='4s').nearest(tolerance="2s")
        dataSpec.close()

        #rainRate = getParsivelRainRate(date)
        #rainRateH= rainRate.interp(time=data.time.values, method='slinear')
        
        qFlagW = data.quality_flag_offset_w
        qFlagX = data.quality_flag_offset_x

        dbz_x = data.X_DBZ_H.copy()
        dbz_ka = data.Ka_DBZ_H.copy()
        dbz_w = data.W_DBZ_H.copy()
        
        rv_x = data.X_VEL_H.copy()
        rv_w = data.W_VEL_H.copy()
        rv_ka = data.Ka_VEL_H.copy()

        width_ka = data.Ka_WIDTH_H
        
        temp = data['ta']
        pa = data.pa.copy()
        rh = data.hur.copy()
        
        #edges
        max_edge = dataEdge.maxVelXR
        min_edge = dataEdge.minVelXR

        #peaks
        peak1  = dataPeaks["peakVelClass"].sel(peakIndex=1).drop("peakIndex") #peak with smaller DV than main peak
        peak2  = dataPeaks["peakVelClass"].sel(peakIndex=2).drop("peakIndex") #peak with smaller DV than peak1
        peak1pow  = dataPeaks["peakPowClass"].sel(peakIndex=1).drop("peakIndex") #peak with smaller DV than main peak
        peak2pow  = dataPeaks["peakPowClass"].sel(peakIndex=2).drop("peakIndex") #peak with smaller DV than peak1

        #getLDR -------
        ldr_kaTmp = getLdr(date)
        ldr_kaTmp["height"] = ldr_kaTmp["height"]-111 #cloudnet files are above NN (JOYCE is at 11m above NN), but level2 files are height above radar
        ldr_kaTmp = ldr_kaTmp.interp(height=data.range.values)
        ldr_kaTmp = ldr_kaTmp.interp(time=data.time.values)
        #transformation
        ldr_ka = rv_ka.copy()
        ldr_ka.values = ldr_kaTmp.values
        #-------------

        dbz_w = fc.maskData(dbz_w, qFlagW, maskP)
        dbz_x = fc.maskData(dbz_x, qFlagX, maskP)
        dbz_w = fc.maskData(dbz_w, qFlagW, maskC)
        dbz_x = fc.maskData(dbz_x, qFlagX, maskC) 
        dbz_w = fc.maskData(dbz_w, qFlagW, maskV)
        dbz_x = fc.maskData(dbz_x, qFlagX, maskV) 

        ldr_ka = fc.smoothData(ldr_ka, 5)
        ddv_xw= (rv_x - rv_w) * -1
        ddv_xk= (rv_x - rv_ka) * -1
        dwr_xka = (dbz_x - dbz_ka)
        dwr_kaw = (dbz_ka - dbz_w)

        print("start getting ML props")
        heightTopMaxLDR, heightLDRmax, heightBottonMaxLDR,        ldrGrad, ldrGrad2 = fc.getMLProperties(ldr_ka, temp)

        offset_MLtop =     36
        offset_MLbot =    -36

        #get velocity from rain (bottom-melting -36m)
        _, _, _, ddvxwBotLDR, _ =         fc.getProp(ddv_xw, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, ddvxkBotLDR, _ =         fc.getProp(ddv_xk, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)

        #get velocity from ice (top-melting +offset_MLtop m)
        _, _, _, _, dwrXKaTopLDR =         fc.getProp(dwr_xka, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, _, dwrKaWTopLDR =         fc.getProp(dwr_kaw, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)

        #get increase of DWRxk in ML
        DWRxk_inML = dwr_xka.where(dwr_xka.range>=heightBottonMaxLDR + offset_MLbot).copy()
        DWRxk_inML = DWRxk_inML.where(dwr_xka.range<=heightTopMaxLDR + offset_MLtop)
        DWRxk_inML_max = DWRxk_inML.max(dim="range")
        Delta_DWRxk = DWRxk_inML_max - dwrXKaTopLDR

        
        #get increase/decrease of ZFR in lower fifth of ML
        MLhalf = (heightBottonMaxLDR + offset_MLbot + (heightTopMaxLDR + offset_MLtop-heightBottonMaxLDR + offset_MLbot)/2)
        _, _, _, MDV_half, _ =         fc.getProp(rv_x, MLhalf, 0)
        _, _, _, Ze_half, _ =         fc.getProp(dbz_x, MLhalf, 0)
        ZfluxX_halfML = MDV_half * Ze_half 
        #third
        MLthird = (heightBottonMaxLDR + offset_MLbot + (heightTopMaxLDR + offset_MLtop-heightBottonMaxLDR + offset_MLbot)/3)
        _, _, _, MDV_third, _ =         fc.getProp(rv_x, MLthird, 0)
        _, _, _, Ze_third, _ =         fc.getProp(dbz_x, MLthird, 0)
        ZfluxX_thirdML = MDV_third * Ze_third 
        #fourth
        MLfourth = (heightBottonMaxLDR + offset_MLbot + (heightTopMaxLDR + offset_MLtop-heightBottonMaxLDR + offset_MLbot)/4)
        _, _, _, MDV_fourth, _ =         fc.getProp(rv_x, MLfourth, 0)
        _, _, _, Ze_fourth, _ =         fc.getProp(dbz_x, MLfourth, 0)
        ZfluxX_fourthML = MDV_fourth * Ze_fourth 
        #fifth
        MLfifth = (heightBottonMaxLDR + offset_MLbot + (heightTopMaxLDR + offset_MLtop-heightBottonMaxLDR + offset_MLbot)/5)
        _, _, _, MDV_fifth, _ =         fc.getProp(rv_x, MLfifth, 0)
        _, _, _, Ze_fifth, _ =         fc.getProp(dbz_x, MLfifth, 0)
        ZfluxX_fifthML = MDV_fifth * Ze_fifth 

        _, _, _, rvXBottomLDR, rvXTopLDR =         fc.getProp(rv_x, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, rvXBottomLDR, rvXTopLDR =         fc.getProp(rv_x, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, widthKaBottomLDR, widthKaTopLDR =      fc.getProp(width_ka, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, zeXBottomLDR, zeXTopLDR =         fc.getProp(dbz_x, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, paBottomLDR, paTopLDR =         fc.getProp(pa, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, rhBottomLDR, rhTopLDR =         fc.getProp(rh, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)

        if not no_mie_notch:
            ##w-estimate (ML bottom from Mie-notch)
            print("start getting mie-notch w-estimate")
            w_estB = paBottomLDR.copy() #copy from pressure array to get x-array format (e.g. with time coordinate) which we fill with other variables
            w_estB.rename("w_estB")
            w_estB.values = np.nan*np.ones_like(paBottomLDR.values)      #empty values
            for i_t,t in enumerate(specW.time):
                print(t.values)
                if not (-rvXBottomLDR.sel(time=t))>3.5: #skip profiles where large rain is not expected
                    w_estB.sel(time=t).values = np.nan
                    continue
                specW_now = specW.sel(time=t).copy()
                specW_MLbot = specW_now.sel(range=heightBottonMaxLDR.sel(time=t)+offset_MLbot,method="nearest").copy()
                w_estB.values[i_t],__ = fc.get_mie_notch_DV(specW_MLbot,paBottomLDR.sel(time=t).values)
                print("w_estB",w_estB.sel(time=t).values)
                print("N(mie_notches)",sum(~np.isnan(w_estB.values)))
            
        #edges
        _, _, _, maxEdgeMaxBottomLDR, maxEdgeTopLDR =         fc.getProp(max_edge, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, minEdgeBottomLDR, minEdgeTopLDR =         fc.getProp(min_edge, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        #peaks
        _, _, _, peak1MaxBottomLDR, peak1TopLDR =         fc.getProp(peak1, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, peak2MaxBottomLDR, peak2TopLDR =         fc.getProp(peak2, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, peak1powMaxBottomLDR, peak1powTopLDR =         fc.getProp(peak1pow, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)
        _, _, _, peak2powMaxBottomLDR, peak2powTopLDR =         fc.getProp(peak2pow, heightBottonMaxLDR + offset_MLbot, heightTopMaxLDR + offset_MLtop)

        ML_thickness = heightTopMaxLDR-heightBottonMaxLDR

        ##w-estimate (MLtop from cloud droplet peak)
        peak1_cleanedIce        = np.where(peak1powTopLDR.values<-30,peak1TopLDR.values,peak2TopLDR.values) #remove second ice mode (should have more than 30dBz power)
        peak1_cleanedIceNoise   = np.where(peak1powTopLDR.values>-50,peak1_cleanedIce,np.nan) #remove Peaks with very low power (probably noise)
        peak2_cleanedNoise      = np.where(peak2powTopLDR.values>-50,peak2TopLDR.values,np.nan)    #remove Peaks with very low power (probably noise)
        peak1TopLDR.values      = np.nanmin([[-peak1_cleanedIceNoise,-peak2_cleanedNoise]],axis=1)[0]

        if not no_mie_notch:
            tmpData = xr.Dataset({'DDVxwB': ddvxwBotLDR,
                                  'DDVxkB': ddvxkBotLDR,
                                  'DWRxkT': dwrXKaTopLDR,
                                  'Delta_DWRxk': Delta_DWRxk,
                                  'ZfluxX_half': ZfluxX_halfML,
                                  'ZfluxX_thirdML': ZfluxX_thirdML,
                                  'ZfluxX_fourthML': ZfluxX_fourthML,
                                  'ZfluxX_fifthML': ZfluxX_fifthML,
                                  'DWRkwT': dwrKaWTopLDR,
                                  'MDVxT':   -rvXTopLDR,
                                  'WkaT':    widthKaTopLDR,
                                  'MDVxB':   -rvXBottomLDR,
                                  'DVedgeLT':-maxEdgeTopLDR,
                                  'DVedgeHT':-minEdgeTopLDR,
                                  'w_estT':peak1TopLDR,
                                  'w_estB':w_estB,
                                  'peak1powT':peak1powTopLDR,
                                  'peak2powT':peak2powTopLDR,
                                  'DVedgeHT':-minEdgeTopLDR,
                                  'DVedgeHB':-minEdgeBottomLDR,
                                  'ZeXt':zeXTopLDR,
                                  'ZeXb':zeXBottomLDR,
                                  'PaB':paBottomLDR,
                                  'PaT':paTopLDR,
                                  'RHb':rhBottomLDR,
                                  'RHt':rhTopLDR,
                                  'ML_thickness':ML_thickness,
                                  'ML_bottom':heightBottonMaxLDR,
                                  'ML_top':heightTopMaxLDR
                                   })
        else:
            tmpData = xr.Dataset({'DDVxwB': ddvxwBotLDR,
                                  'DDVxkB': ddvxkBotLDR,
                                  'DWRxkT': dwrXKaTopLDR,
                                  'Delta_DWRxk': Delta_DWRxk,
                                  'ZfluxX_halfML': ZfluxX_halfML,
                                  'ZfluxX_thirdML': ZfluxX_thirdML,
                                  'ZfluxX_fourthML': ZfluxX_fourthML,
                                  'ZfluxX_fifthML': ZfluxX_fifthML,
                                  'DWRkwT': dwrKaWTopLDR,
                                  'MDVxT':   -rvXTopLDR,
                                  'WkaT':    widthKaTopLDR,
                                  'MDVxB':   -rvXBottomLDR,
                                  'DVedgeLT':-maxEdgeTopLDR,
                                  'DVedgeHT':-minEdgeTopLDR,
                                  'w_estT':peak1TopLDR,
                                  'peak1powT':peak1powTopLDR,
                                  'peak2powT':peak2powTopLDR,
                                  'DVedgeHT':-minEdgeTopLDR,
                                  'DVedgeHB':-minEdgeBottomLDR,
                                  'ZeXt':zeXTopLDR,
                                  'ZeXb':zeXBottomLDR,
                                  'PaB':paBottomLDR,
                                  'PaT':paTopLDR,
                                  'RHb':rhBottomLDR,
                                  'RHt':rhTopLDR,
                                  'ML_thickness':ML_thickness,
                                  'ML_bottom':heightBottonMaxLDR,
                                  'ML_top':heightTopMaxLDR
                                   })

        data.close()
        dataEdge.close()
        dataPeaks.close()
       
        if os.path.exists('data/MLprop' + date_str  + '.pkl'):
            ###append variables and not run expensive mie-notch again 

            #load old file
            with open('data/MLprop' + date_str  + '.pkl', 'rb') as f:
                tmpData_loaded = pickle.load(f)

            #overwrite all variables with values from currently calculated dictionary
            for key in tmpData.keys():
                tmpData_loaded[key] = tmpData[key]

            #save new file
            with open('data/MLprop'+ date_str + '.pkl', 'wb') as f:
                pickle.dump(tmpData_loaded, f, pickle.HIGHEST_PROTOCOL)
            print("overwrite data where available")
            tmpData = tmpData_loaded
        else:
            with open('data/MLprop'+ date_str + '.pkl', 'wb') as f:
                pickle.dump(tmpData, f, pickle.HIGHEST_PROTOCOL)
            print("create new file")

    else: #load data from previous calculation
        try:
            with open('data/MLprop' + date_str  + '.pkl', 'rb') as f:
                tmpData = pickle.load(f)
            print("load MLprop file")
        except:
            print(date,"not there yet")
            tmpData = {}
    
    return tmpData

def processOneDayProfiles(fileName,fileNameEdge,fileNamePeaks,fileNameLV0, maskR, maskL, 
                             maskP, maskC ,maskV,calc_or_load,no_mie_notch=False):
    
    date = pd.to_datetime(fileName.split('/')[-1].split('_')[0])
    date_str = pd.to_datetime(str(date)).strftime('%Y%m%d')

    if calc_or_load:
        data = xr.open_dataset(fileName)
        if os.path.isfile(fileNameEdge):
            dataEdge = xr.open_dataset(fileNameEdge)
        else:
            return xr.Dataset({})
        if os.path.isfile(fileNamePeaks):
            dataPeaks = xr.open_dataset(fileNamePeaks)
        else:
            return xr.Dataset({})
        dataSpec = xr.open_mfdataset(fileNameLV0)
        dataSpec = xr.decode_cf(dataSpec) #Decode the given Dataset or Datastore according to CF conventions into a new Dataset.
       
        #bring time to same standard and values as LV2 data 
        dataSpec.time.attrs['units']='seconds since 1970-01-01 00:00:00 UTC'
        specW   = 10*np.log10(dataSpec.WSpecH).copy()
        specW["time"] = pd.to_datetime(specW.time.values*1e9)
        specW.resample(time='4s').nearest(tolerance="2s")
        dataSpec.close()

        #rainRate = getParsivelRainRate(date)
        #rainRateH= rainRate.interp(time=data.time.values, method='slinear')
        
        qFlagW = data.quality_flag_offset_w
        qFlagX = data.quality_flag_offset_x

        dbz_x = data.X_DBZ_H.copy()
        dbz_ka = data.Ka_DBZ_H.copy()
        dbz_w = data.W_DBZ_H.copy()
        
        rv_x = data.X_VEL_H.copy()
        rv_w = data.W_VEL_H.copy()
        rv_ka = data.Ka_VEL_H.copy()

        width_ka = data.Ka_WIDTH_H
        
        temp = data['ta']
        pa = data.pa.copy()
        rh = data.hur.copy()
        
        #edges
        max_edge = dataEdge.maxVelXR
        min_edge = dataEdge.minVelXR

        #peaks
        peak1  = dataPeaks["peakVelClass"].sel(peakIndex=1).drop("peakIndex") #peak with smaller DV than main peak
        peak2  = dataPeaks["peakVelClass"].sel(peakIndex=2).drop("peakIndex") #peak with smaller DV than peak1
        peak1pow  = dataPeaks["peakPowClass"].sel(peakIndex=1).drop("peakIndex") #peak with smaller DV than main peak
        peak2pow  = dataPeaks["peakPowClass"].sel(peakIndex=2).drop("peakIndex") #peak with smaller DV than peak1

        #getLDR -------
        ldr_kaTmp = getLdr(date)
        ldr_kaTmp["height"] = ldr_kaTmp["height"]-111 #cloudnet files are above NN (JOYCE is at 11m above NN), but level2 files are height above radar
        ldr_kaTmp = ldr_kaTmp.interp(height=data.range.values)
        ldr_kaTmp = ldr_kaTmp.interp(time=data.time.values)
        #transformation
        ldr_ka = rv_ka.copy()
        ldr_ka.values = ldr_kaTmp.values
        #-------------

        dbz_w = fc.maskData(dbz_w, qFlagW, maskP)
        dbz_x = fc.maskData(dbz_x, qFlagX, maskP)
        dbz_w = fc.maskData(dbz_w, qFlagW, maskC)
        dbz_x = fc.maskData(dbz_x, qFlagX, maskC) 
        dbz_w = fc.maskData(dbz_w, qFlagW, maskV)
        dbz_x = fc.maskData(dbz_x, qFlagX, maskV) 

        ldr_ka = fc.smoothData(ldr_ka, 5)
        ddv_xw= (rv_x - rv_w) * -1
        ddv_xk= (rv_x - rv_ka) * -1
        dwr_xka = (dbz_x - dbz_ka)
        dwr_kaw = (dbz_ka - dbz_w)
        fz_X = dbz_x * (-rv_x)


        print("start getting ML profiles")
        heightTopMaxLDR, heightLDRmax, heightBottonMaxLDR,        ldrGrad, ldrGrad2 = fc.getMLProperties(ldr_ka, temp)

        offset_MLtop =    36
        offset_MLbot =   -36

        h_rel_lims = [-0.2,1.5]
        h_rel_steps = (h_rel_lims[1]-h_rel_lims[0])*50 #50 grid points within the ML
        h_rel_ML = np.linspace(h_rel_lims[0],h_rel_lims[1],h_rel_steps) #height relative to melting layer
        tmpData = xr.Dataset({})
        keys = ["peak1","peak2","peak1pow","peak2pow","LDR" ,"ZeX","MDVx","DWRxk" ,"DWRkw","Fz","DVedgeL","DVedgeH","pa"]
        vars = [peak1,peak2,peak1pow,peak2pow,ldr_ka,dbz_x,-rv_x  ,dwr_xka,dwr_kaw,fz_X, -max_edge ,-min_edge,pa]
        #keys = ["MDVx","DWRxk","Fz","pa"]
        #vars = [-rv_x,dwr_xka,fz_X, pa]
        Delta_ML = (heightTopMaxLDR + offset_MLtop - heightBottonMaxLDR + offset_MLtop) #ML thickness
        h_rel_grid = np.ones([ldr_ka.shape[0],h_rel_ML.shape[0]])*h_rel_ML
        h_rel_grid = xr.DataArray(h_rel_grid,
                             coords={'time':ldr_ka.time,
                                     'range':h_rel_ML},
                             dims=('time','range'))

        for var,key in zip(vars,keys):
            print("get: ",key)
            tmpData[key] = fc.getProfile(var, (heightBottonMaxLDR+ offset_MLbot) + h_rel_grid * Delta_ML)

        if True:
            ##w-estimate (MLtop from cloud droplet peak)
            peak1_cleanedIce        = np.where(tmpData["peak1pow"].values<-30,tmpData["peak1"].values,tmpData["peak2"].values) #remove second ice mode (should have more than 30dBz power)
            peak1_cleanedIceNoise   = np.where(tmpData["peak1pow"].values>-50,peak1_cleanedIce,np.nan) #remove Peaks with very low power (probably noise)
            peak2_cleanedNoise      = np.where(tmpData["peak2pow"].values>-50,tmpData["peak2"].values,np.nan)    #remove Peaks with very low power (probably noise)
            tmpData["w_estT"] = tmpData["peak1"].copy(data=np.ones_like(tmpData["peak1"])*np.nan) #initialize empty dataframe
            tmpData["w_estT"].values = np.nanmin([[-peak1_cleanedIceNoise,-peak2_cleanedNoise]],axis=1)[0]
            tmpData["w_estT"].loc[dict(range=slice(h_rel_lims[0],0.9))] = np.nan*np.ones_like(tmpData["w_estT"].loc[dict(range=slice(h_rel_lims[0],0.9))]) #remove data within ML

        #if True: #you might want to deactivate this slow calculation of the mie-notch
        if not no_mie_notch:
            ##w-estimate (ML bottom from Mie-notch)
            print("start getting mie-notch w-estimate")
            w_estB = tmpData["MDVx"].copy(data=np.nan*np.ones_like(tmpData["MDVx"])) #copy to get x-array format (e.g. with time coordinate) which we fill with other values
            w_estB.rename("w_estB")
            for i_t,t in enumerate(specW.time):
                print(t.values)
                if not any((tmpData["MDVx"].sel(time=t,range=slice(h_rel_lims[0],0.1)))>3.5): #skip profiles where large rain is not expected
            
                    #w_estB.sel(time=t).values = np.nan * np.ones_like(tmpData["MDVx"].sel(time=t,range=slice(h_rel_lims[0],0.1)))
                    continue
                for MDVlower,abs_height,h_rel in zip(tmpData["MDVx"].sel(time=t,range=slice(h_rel_lims[0],0.1)),((heightBottonMaxLDR+ offset_MLbot) + h_rel_grid * Delta_ML).sel(time=t,range=slice(h_rel_lims[0],0.1)),tmpData["MDVx"].sel(time=t,range=slice(h_rel_lims[0],0.1)).range):
                    if MDVlower>3.5: #skip profiles where large rain is not expected
                        specW_now = specW.sel(time=t,range=abs_height,method="nearest").copy()
                        w_estB.loc[dict(time=t,range=h_rel)],__ = fc.get_mie_notch_DV(specW_now,tmpData["pa"].sel(time=t,range=h_rel))
                        print("w_estB",h_rel,w_estB.loc[dict(time=t,range=h_rel)])
                        print("N(mie_notches)",sum(~np.isnan(w_estB.values)))
            tmpData["w_estB"] = w_estB
            
        #derive some quantities
        tmpData["FzNorm"] = tmpData["Fz"]/tmpData["Fz"].sel(range=1.0,method="nearest") #normalize by flux at MLtop
        Fz_smoothed = tmpData["FzNorm"].rolling(range=3, min_periods=2,center=True).mean().copy()
        tmpData["FzGrad"] = -Fz_smoothed.differentiate('range') #normalize by flux at MLtop


        data.close()
        dataEdge.close()
        dataPeaks.close()
       
        if os.path.exists('/net/morget/melting_data/MLprofiles' + date_str  + '.pkl'):
            ###append variables and not run expensive mie-notch again 

            #load old file
            with open('/net/morget/melting_data/MLprofiles' + date_str  + '.pkl', 'rb') as f:
                tmpData_loaded = pickle.load(f)

            #overwrite all variables with values from currently calculated dictionary
            for key in tmpData.keys():
                tmpData_loaded[key] = tmpData[key]

            #save new file
            with open('/net/morget/melting_data/MLprofiles'+ date_str + '.pkl', 'wb') as f:
                pickle.dump(tmpData_loaded, f, pickle.HIGHEST_PROTOCOL)
            print("overwrite /net/morget/melting_data where available")
            tmpData = tmpData_loaded
        else:
            with open('/net/morget/melting_data/MLprofiles'+ date_str + '.pkl', 'wb') as f:
                pickle.dump(tmpData, f, pickle.HIGHEST_PROTOCOL)
            print("create new file")

    else: #load /net/morget/melting_data from previous calculation
        try:
            with open('/net/morget/melting_data/MLprofiles' + date_str  + '.pkl', 'rb') as f:
                tmpData = pickle.load(f)
            print("load MLprofiles file")
            tmpData = tmpData.dropna(dim="time", #to save memory remove timesteps where all relevant data is missing
                subset=["FzNorm","LDR","ZeX","MDVx","DWRxk","DWRkw","DVedgeL","DVedgeH","w_estT","w_estB","FzNorm","FzGrad"],
                how="all")
        except:
            print(date,"not there yet")
            tmpData = {}
    
    return tmpData

def get_vars(vars_out,res):

    '''
    get all variables from the MeltingLayer dictionary
    vars_out: name of variables to load
    res: results dictionary
    '''

    class vars(object):
        def __init__(self, results_name,plot_label,lims):
            #names
            self.results_name   = results_name              #name in "results"-dictionary
            self.plot_label     = plot_label
            #limits
            self.lims           = lims

        def get_data(self,res):
            self.data           = res[self.results_name]     #copy data from "results"-dictionary
            

    #collect data
    #                          results_name    plot_label              lims

    all_dic = dict()
    #all_dic["ZFR"]                 = vars( "ZFR"            ,"ZFR"                              ,[0.23-0.20,0.23+0.20])
    #all_dic["ZFR2"]                 = vars( "ZFR"            ,"ZFR"                              ,[0.23-0.20,0.23+0.20])
    #all_dic["ZFRvCorrT"]           = vars( "ZFRvCorrT"      ,"ZFR$_{v-corr,top}$"               ,[0.23-0.20,0.23+0.20])
    #all_dic["ZFRvCorrB"]           = vars( "ZFRvCorrB"      ,"ZFR$_{v-corr,bottom}$"            ,[0.23-0.20,0.23+0.20])
    #all_dic["ZFRvCorr"]            = vars( "ZFRvCorr"       ,"ZFR$_{v-corr}$"                   ,[0.23-0.20,0.23+0.20])
    #all_dic["ZFRvCorr2"]           = vars( "ZFRvCorr"       ,"ZFR$_{v-corr}$"                   ,[0.23-0.20,0.23+0.20]) #Quick and dirty fix to use variable two time
    #all_dic["ZFRvCorr3"]           = vars( "ZFRvCorr"       ,"ZFR$_{v-corr}$"                   ,[0.23-0.20,0.23+0.20]) #Quick and dirty fix to use variable two time
    all_dic["ZFR"]                 = vars( "ZFR"            ,"ZFR"                              ,[0.3,1.3])
    all_dic["ZFR2"]                 = vars( "ZFR"            ,"ZFR"                              ,[0.3,1.3])
    all_dic["ZFRvCorrT"]           = vars( "ZFRvCorrT"      ,"ZFR$_{v-corr,top}$"               ,[0.3,1.3])
    all_dic["ZFRvCorrB"]           = vars( "ZFRvCorrB"      ,"ZFR$_{v-corr,bottom}$"            ,[0.3,1.3])
    all_dic["ZFRvCorr"]            = vars( "ZFRvCorr"       ,"ZFR$_{v-corr}$"                   ,[0.3,1.3])
    all_dic["ZFRvCorr2"]           = vars( "ZFRvCorr"       ,"ZFR$_{v-corr}$"                   ,[0.3,1.3]) #Quick and dirty fix to use variable two time
    all_dic["ZFRvCorr3"]           = vars( "ZFRvCorr"       ,"ZFR$_{v-corr}$"                   ,[0.3,1.3]) #Quick and dirty fix to use variable two time
    all_dic["Delta_ZF_upper"]      = vars( "Delta_ZF_upper" ,"$\Delta$ ZF$_{upper}$"           ,[1.0,8.0])
    all_dic["Delta_ZF_lower"]      = vars( "Delta_ZF_lower" ,"$\Delta$ ZF$_{lower}$"           ,[-1.0,1.0])
    all_dic["DWRxkT"]              = vars( "DWRxkT"         ,"DWR$_{X-Ka,top}$ [dB]"            ,[-1.5,15]            )
    all_dic["Delta_DWRxk"]         = vars( "Delta_DWRxk"    ,"$\Delta$ DWR$_{X-Ka}$ [dB]"       ,[0,10]               )
    all_dic["DWRkwT"]              = vars( "DWRkwT"         ,"DWR$_{Ka-W,top}$ [dB]"            ,[-1.5,10]            )
    all_dic["ZeX"]                 = vars( "ZeXt"           ,"Ze$_{X,top}$ [dB]"                ,[0,35]               )
    all_dic["ZeXb"]                = vars( "ZeXb"           ,"Ze$_{X,bottom}$ [dB]"             ,[0,35]               )
    all_dic["MDVxT"]               = vars( "MDVxT_denscorr" ,"MDV$_{X,top}$ [m/s]"              ,[0.6,2.5]            )
    all_dic["MDVxT2"]              = vars( "MDVxT_denscorr" ,"MDV$_{X,top}$ [m/s]"              ,[0.6,2.5]            )#Quick and dirty fix to use variable two time
    all_dic["MDVxB"]               = vars( "MDVxB_denscorr" ,"MDV$_{X,bottom}$ [m/s]"           ,[3.0,8.0]            )
    all_dic["MDVvCorrT"]           = vars( "MDVvCorrT"      ,"MDV$_{X,v-corr,top}$ [m/s]"       ,[0.6,2.5]            )
    all_dic["MDVvCorrT_denscorr"]  = vars( "MDVvCorrT_denscorr","MDV$_{X,v-corr,top}$ [m/s]"       ,[0.6,2.5]            )
    all_dic["MDVvCorrT2"]          = vars( "MDVvCorrT"      ,"MDV$_{X,v-corr,top}$ [m/s]"       ,[0.6,2.5]            )#Quick and dirty fix to use variable two time
    all_dic["MDVvCorrT_denscorr2"] = vars( "MDVvCorrT_denscorr","MDV$_{X,v-corr,top}$ [m/s]"       ,[0.6,2.5]            )
    all_dic["MDVvCorrT_denscorr3"] = vars( "MDVvCorrT_denscorr","MDV$_{X,v-corr,top}$ [m/s]"       ,[0.6,2.5]            )
    all_dic["MDVvCorrB"]           = vars( "MDVvCorrB"      ,"MDV$_{X,v-corr,bottom}$ [m/s]"    ,[2.0,8.0]            )
    all_dic["MDVvCorrB_denscorr"]  = vars( "MDVvCorrB_denscorr"      ,"MDV$_{X,v-corr,bottom}$ [m/s]"    ,[2.0,9.0]            )
    all_dic["DV_edge_lowT"]        = vars( "DVedgeLT"       ,"DV$_{edge,low,top}$ [m/s]"        ,[-1.5,1.5]           )
    all_dic["DV_edge_highT"]       = vars( "DVedgeHT"       ,"DV$_{edge,high,top}$ [m/s]"       ,[1.0,6.0]            )
    all_dic["DV_edge_highB"]       = vars( "DVedgeHB"       ,"DV$_{edge,high,bottom}$ [m/s]"    ,[4.0,8.0]            )
    all_dic["w_estT"]              = vars( "w_estT"         ,"w$_{est.,top}$ [m/s]"             ,[-1.0,0.5]           ) #estimate of the wind velocity
    all_dic["w_estB"]              = vars( "w_estB"         ,"w$_{est.,bottom}$ [m/s]"          ,[-1.0,1.0]           ) #estimate of the wind velocity
    all_dic["peak1powT"]           = vars( "peak1powT"      ,"Ze$_{peak1,top}$ [dBz]"           ,[-60,0]              )
    all_dic["peak2powT"]           = vars( "peak2powT"      ,"Ze$_{peak2,top}$ [dBz]"           ,[-60,0]              )
    all_dic["widthKaT"]            = vars( "WkaT"           ,"SW$_{Ka}$ [m/s]"                  ,[0.1,1.0]            )
    all_dic["delz"]                = vars( "ML_thickness"   ,"$\Delta z_{melt}$ [m]"            ,[100.,600.]          )
    all_dic["rhT"]                 = vars( "RHt"            ,"RH$_{top}$ [%]"                   , [50,105]            )
    all_dic["rhB"]                 = vars( "RHb"            ,"RH$_{bottom}$ [%]"                , [70,105]            )

    dic = dict()
    for key in vars_out:
        dic[key] = vars(all_dic[key].results_name,all_dic[key].plot_label,all_dic[key].lims)
        dic[key].get_data(res)

    return dic

def get_vars_profiles(vars_out,res):

    '''
    get all variables from the MeltingLayer profiles dictionary
    vars_out: name of variables to load
    res: results dictionary
    '''

    class vars(object):
        def __init__(self, results_name,plot_label,lims):
            #names
            self.results_name   = results_name              #name in "results"-dictionary
            self.plot_label     = plot_label
            #limits
            self.lims           = lims

        def get_data(self,res):
            self.data           = res[self.results_name]     #copy data from "results"-dictionary
            

    #collect data
    #                          results_name    plot_label              lims

    all_dic = dict()
    all_dic["LDR"]                 = vars( "LDR"            ,'LDR$_{Ka}$ [dB]'                  ,[-35,-5])
    all_dic["ZeX"]                 = vars( "ZeX"            ,"Ze$_{X}$ [dB]"                ,[10,35]               )
    all_dic["MDVx"]                = vars( "MDVx"           ,"MDV$_{X}$ [m/s]"              ,[1.0,7]            )
    all_dic["Fz"]                  = vars( "Fz"             ,"F$_{Z,X}$ [dB m/s]"              ,[20,150]            )
    all_dic["DWRxk"]               = vars( "DWRxk"          ,"DWR$_{X,Ka}$ [dB]"             ,[-1.5,15]            )
    all_dic["DWRkw"]               = vars( "DWRkw"          ,"DWR$_{Ka,W}$ [dB]"             ,[-1.5,15]            )
    all_dic["FzNorm"]              = vars( "FzNorm"         ,"F$_{Z,X}$/F$_{Z,X,top}$"              ,[0.5,6]            )
    all_dic["FzGrad"]              = vars( "FzGrad"         ,"dF$_{Z,X,norm}$/d(-h)"              ,[-2,15]            )
    all_dic["DV_edge_low"]         = vars( "DVedgeL"        ,"SEV [m/s]"                       ,[-0.5,9]           )
    all_dic["DV_edge_high"]        = vars( "DVedgeH"        ,"DV$_{edges}$ [m/s]"       ,[-0.5,9]            )
    all_dic["w_estT"]              = vars( "w_estT"         ,"w$_{est.}$ [m/s]"             ,[-1.0,1.0]           ) #estimate of the wind velocity
    all_dic["w_estB"]              = vars( "w_estB"         ,"w$_{est.}$ [m/s]"          ,[-1.0,1.0]           ) #estimate of the wind velocity

    dic = dict()
    for key in vars_out:
        dic[key] = vars(all_dic[key].results_name,all_dic[key].plot_label,all_dic[key].lims)
        dic[key].get_data(res)

    return dic

def profiles(results,save_spec,av_min="0",col=1,onlydate="",no_mie_notch=False,correct_w_before_categorization=False):
    '''
    plot profiles of several quantities vs. the normalized height within the melting layer
    '''

    mpl.style.use('seaborn')
    mpl.rcParams['font.size'] = 40
    mpl.rcParams['legend.fontsize'] = 40
    mpl.rcParams['figure.titlesize'] = 40

    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.framealpha'] = 0.7
    mpl.rcParams['legend.facecolor']='silver'
    mpl.rcParams['legend.frameon']=True

    mpl.rcParams['ytick.labelsize']= 40
    mpl.rcParams['xtick.labelsize']= 40
    mpl.rcParams['axes.titlesize']=40
    mpl.rcParams['axes.labelsize']=40

    fig,axes = plt.subplots(ncols=2,nrows=4,figsize=(20,40),facecolor="w",sharey=True)

    ZFR_coll   = dict() #collection of all ZFRs

    #get variables for categorization
    vars_dict    = get_vars(["DWRxkT","MDVxT"],results)
    MDVxT = vars_dict["MDVxT"]
    DWRxkT = vars_dict["DWRxkT"]

    #get variables
    results = get_observed_melting_properties(onlydate=onlydate,calc_or_load=col,profiles=True,no_mie_notch=no_mie_notch) #calc_or_load: 1: calculate each day 0: load each day
    #keys = ["LDR","ZeX","MDVx","DWRxk","Fz","FzNorm","FzGrad","DWRkw"]
    keys = ["LDR","ZeX","MDVx","DV_edge_high","DV_edge_low","Fz","FzNorm","FzGrad","w_estT","w_estB"]
    #keys = ["Fz","w_estB"]
    #keys = ["w_estT"]
    axflat = axes.flatten()
    axflat = [axflat[0],axflat[1],axflat[2],axflat[3],axflat[3],axflat[4],axflat[5],axflat[6],axflat[7],axflat[7]] #put DV edges together in same plot
    letters=["a)","b)","c)","d)","d)","e)","f)","g)","h)","h)"] 
    for i_var,(key,ax) in enumerate(zip(keys,axflat)):

        if key!="FzNorm":
            vars_dict    = get_vars_profiles([key],results)
            var = vars_dict[key]
        else:
            vars_dict    = get_vars_profiles([key],results)
            var = vars_dict[key]
            var.data = vars_dict[key].data/vars_dict[key].data.sel(range=1.0,method="nearest") #normalize by flux at MLtop
        if correct_w_before_categorization:
            w_estT = get_vars_profiles(["w_estT"],results); varColl = categorize_part_type(var.data,MDVxT.data+w_estT["w_estT"].data.sel(range=1.0,method="nearest"),DWRxkT.data,dict(),key,0,"F") #returns dictionary with categorized variables e.g. varXcoll["F0rimedZFR"] #correct for w_estT before categoriztaion
        else:
            varColl = categorize_part_type(var.data,MDVxT.data,DWRxkT.data,dict(),key,0,"F") #returns dictionary with categorized variables e.g. varXcoll["F0rimedZFR"]

        z_dic = dict() #save "histogram" in dictionary
        for i_ptype,ptype in enumerate(["unrimed","transitional","rimed"]):

            ax.axhline(1.0,c="k",ls="--",lw=5)
            ax.axhline(0.0,c="k",ls="--",lw=5)
            if key=="FzNorm":
                ax.axvline(1/0.23,c="k",ls="--",lw=5)
            if key=="w_estB":
                varColl["F0"+ ptype + key] = varColl["F0"+ ptype + key].where(varColl["F0"+ ptype + key].range<0.0).copy() #crop mie-notch within ML
            low_quant = varColl["F0"+ ptype + key].quantile(0.25,dim="time")
            upp_quant = varColl["F0"+ ptype + key].quantile(0.75,dim="time")
            med_quant = varColl["F0"+ ptype + key].mean(dim="time")
            if key=="FzGrad":
                low_quant = low_quant.rolling(min_periods=1,range=3).mean()
                upp_quant = upp_quant.rolling(min_periods=1,range=3).mean()
                med_quant = med_quant.rolling(min_periods=1,range=3).mean()
            ax.plot(low_quant,varColl["F0" + ptype +key].range,label="__None",color=["blue","green","red"][i_ptype],lw=1,ls="-",alpha=0.5)
            ax.plot(upp_quant,varColl["F0" + ptype +key].range,label="__None",color=["blue","green","red"][i_ptype],lw=1,ls="-",alpha=0.5)
            ax.plot(med_quant,varColl["F0" + ptype +key].range,label=ptype,c=["blue","green","red"][i_ptype],lw=5)
            ax.fill_betweenx(varColl["F0" + ptype +key].range,varColl["F0"+ ptype + key].quantile(0.25,dim="time"),varColl["F0"+ ptype + key].quantile(0.75,dim="time"),label="__None",color=["blue","green","red"][i_ptype],lw=1,ls="-",alpha=0.1)
        

        #legend limits etc.
        if i_var==0:
            ax.legend()
        if i_var in [0,2,5,7]:
            ax.set_ylim([varColl["F0" + ptype +key].range[0],varColl["F0" + ptype +key].range[-1]])
            ax.set_ylabel("h$_{rel}$")
        ax.set_xlim(var.lims)
        ax.set_xlabel(var.plot_label)

        #add letters
        ax.text(0.0,1.0, letters[i_var],fontsize=52,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes, bbox=dict(facecolor='white', alpha=0.9))
        if key=="w_estB":
            ax.arrow(0.05, 0.08, 0.25, 0.0,lw=8)
            ax.arrow(-0.05, 0.08, -0.25, 0.0,lw=8)
            ax.text(0.05,0.15, "downwind",fontsize=40,horizontalalignment='left',verticalalignment='center')
            ax.text(-0.05,0.15, "upwind",fontsize=40,horizontalalignment='right',verticalalignment='center')
        elif key=="MDVx":
            #add melting fraction according to Mitra
            from scipy.optimize import brentq
            for i_ptype,ptype in enumerate(["unrimed","transitional","rimed"]):
                MDVsnow = varColl["F0"+ ptype + key].mean(dim="time").sel(range=1.0,method="nearest").copy()
                MDVrain = varColl["F0"+ ptype + key].mean(dim="time").sel(range=0.0,method="nearest").copy()
                MDVmean = varColl["F0"+ ptype + key].mean(dim="time").copy()
                phi = (MDVmean-MDVsnow)/(MDVrain-MDVsnow)

                a_fit = 0.246 #from Frick et al "A bulk parametrization of melting snowflakes with explicit liquid water fraction for the COSMO model" GMD Discussion 
                def func(x,y):
                    return a_fit*x + (1-a_fit) * x**7 - y
                    
                fmelt = []
                for i,phi_now in enumerate(phi.values):
                    if MDVmean.range.values[i]<1.0 and MDVmean.range.values[i]>0.0:
                        res = brentq(func,-0.5,1.5,args=phi.values[i])
                    else:
                        res = np.nan
                    fmelt.append(res)
                fmelt = np.array(fmelt)
                ax2 = ax.twiny()
                ax2.plot(fmelt,varColl["F0" + ptype +key].range,label="__",c=["blue","green","red"][i_ptype],lw=2,ls="--")
                if i_ptype==0:
                    ax2.plot(np.nan,np.nan,ls="--",c="k",label="$f_{melt}$")
                    ax2.legend()
                
            
            #c=["blue","green","red"][i_ptype]
    ax2.set_xlabel("$f_{melt}$")
    #fig.delaxes(axes[1][3])
    #plt.tight_layout() #rect=(0,0.00,1,1))
    plt.subplots_adjust(left=0.1,
                    bottom=0.03, 
                    right=0.95, 
                    top=0.98, 
                    wspace=0.1, 
                    hspace=0.3)
    #save figure
    savestr = 'plots/profiles_ptype' + save_spec 
    if correct_w_before_categorization:
        savestr += "_correct_w_before_categorization"
    plt.savefig(savestr + '.pdf')
    print("pdf is at: ",savestr + '.pdf')
    plt.clf()

def boxplot(results,av_min="0",showfliers=False,ONLYbci=False):
    '''
    plot boxplots after different filtering (Zeflux,RH) and corrections (vertical wind estimate)
    showfliers: show outliers of boxplot
    ONLYbci: show only panels b) c) and i) (for presentation)
    '''
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator
    mpl.style.use('seaborn')

    mpl.rcParams['font.size'] = 20
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['figure.titlesize'] = 20

    mpl.rcParams['ytick.labelsize']= 20
    mpl.rcParams['xtick.labelsize']= 20
    mpl.rcParams['axes.titlesize']=20
    mpl.rcParams['axes.labelsize']=20

    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.framealpha'] = 0.7
    mpl.rcParams['legend.facecolor']='silver'
    mpl.rcParams['legend.frameon']=True

    mpl.rcParams['lines.linewidth']=3

    def add_letters(ax,row=0):
        if row==0:
            letters=["a)","b)","c)"] 
        elif row==1:
            letters=["d)","e)","f)"] 
        elif row==2:
            letters=["g)","h)","i)"] 

        #make labels
        ax.text(0.0,0.9, letters[0],fontsize=42,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
        ax.text(0.33,0.9, letters[1],fontsize=42,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
        ax.text(0.67,0.9, letters[2],fontsize=42,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)

    if not ONLYbci:
        fig,axes = plt.subplots(ncols=1,nrows=3,figsize=(12,20))
    else: ###part for presentation (simplified plot)
        fig,axes = plt.subplots(ncols=1,nrows=1,figsize=(12,7))
        ZFR_coll   = dict() #collection of all ZFRs
        filterNumbers = [0,1,2]
        filterDescr   = ["F1: F$_{Z,X}$>20dBz m/s","F2: F1 & RH>95%","F1 + w-corr"]
        for i_filter in filterNumbers:
            if i_filter in [0,2]:
                filterZeflux  = 20
                #filter small fluxes
                results_now = results.where((results["MDVxT"] * results["ZeXt"])  >filterZeflux)
                results_now = results_now.where((results_now["MDVxB"] * results_now["ZeXb"]/0.23)  >filterZeflux)
                #results_now = results_now.where((results_now["ZeXt"]>0)) #filter also cases where Ze<0 and MDV<0
            elif i_filter==1:
                filterZeflux  = 20
                #filter small fluxes
                results_now = results.where((results["MDVxT"] * results["ZeXt"])  >filterZeflux)
                results_now = results_now.where((results_now["MDVxB"] * results_now["ZeXb"]/0.23)  >filterZeflux)
                #results_now = results_now.where((results_now["ZeXt"]>0)) #filter also cases where Ze<0 and MDV<0
                results_now = results_now.where((results_now["RHb"]>95.)) #filter also cases with low RH

            #2D loop to generate plot_matrix
            if i_filter in [0,1]:
                vars_dict    = get_vars(["ZFR"  ,"MDVxT","DWRxkT"],results_now)
                MDVkey = "MDVxT"
                ZFRkey = "ZFR"
            else:
                vars_dict    = get_vars(["ZFR"  ,"ZFRvCorrT","ZFRvCorrB","ZFRvCorr","MDVxT","MDVvCorrT_denscorr","DWRxkT"],results_now)
                MDVkey = "MDVvCorrT_denscorr"
                ZFRkey = "ZFRvCorr"

            MDVxT = vars_dict[MDVkey]
            DWRxkT = vars_dict["DWRxkT"]
            ZFR = vars_dict[ZFRkey]

            ZFR_coll = categorize_part_type(ZFR.data,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_filter,"F")
            
            #uncomment next lines get other variables for categories
            key = "delz"
            var = get_vars([key],results_now)[key]
            var_coll = categorize_part_type(var.data,MDVxT.data,DWRxkT.data,dict(),key,i_filter,"F")

        # DATAFRAMES WITH TRIAL COLUMN ASSIGNED
        for i_filter in filterNumbers:
            if i_filter in [0,1]:
                ZFRkey = "ZFR"
            else:
                ZFRkey = "ZFRvCorr"
            ZFRs_filter = np.transpose(np.stack([ZFR_coll["F" + str(i_filter) + "unrimed" + ZFRkey],ZFR_coll["F" + str(i_filter) + "transitional" + ZFRkey],ZFR_coll["F" + str(i_filter) + "rimed" + ZFRkey]]))
            df_now = pd.DataFrame(ZFRs_filter, columns=list(["unrimed","transitional","rimed"])).assign(Filter=filterDescr[i_filter] +  "\nN=" + "{:.1f}".format(ZFR_coll["F" + str(i_filter) + "Hdata"]) + "h" + "\n(" + "{:.1f}".format(ZFR_coll["F" + str(i_filter) + "unrimed" + "_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["F" + str(i_filter) + "transitional" + "_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["F" + str(i_filter) + "rimed" + "_perc"]*100)+ "%)")
            if i_filter==0:
                df = df_now.copy()
            else:
                df = pd.concat([df,df_now]) #, df2, df3])                                # CONCATENATE
        mdf = pd.melt(df, id_vars=['Filter'], var_name=['Particle Type'])      # MELT
        mdf.rename(columns={'value':'ZFR'}, inplace=True)

        ####plot
        ax = sns.boxplot(x="Filter", y="ZFR", hue="Particle Type", data=mdf,
            #showmeans=True,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"},
            showfliers = showfliers, flierprops={"marker":'o', "markersize":2,"alpha":0.01},
            ax=axes) 
        ax.axhline(1.0,c="magenta",lw=2)
        ax.set_ylim([0.0,2.1])
        plt.tight_layout() #rect=(0,0.00,1,1))
        #save figure
        savestr = 'plots/Boxplot4pres'
        #if av_min!="0":
        #    savestr+= "_av" + str(av_min) + "min"
        plt.savefig(savestr + '.pdf')
        print("pdf is at: ",savestr + '.pdf')
        return
        ###END: part for presentation (simplified plot)

    ZFR_coll   = dict() #collection of all ZFRs
    filterNumbers = [0,1,2]
    filterDescr   = ["none","F1: F$_{Z,X}$>20dBz m/s","F2: F1 & RH>95%"]
    for i_filter in filterNumbers:
        if i_filter==0:
            results_now = results #no filter
        elif i_filter==1:
            filterZeflux  = 20
            #filter small fluxes
            results_now = results.where((results["MDVxT"] * results["ZeXt"])  >filterZeflux)
            results_now = results_now.where((results_now["MDVxB"] * results_now["ZeXb"]/0.23)  >filterZeflux)
            #results_now = results_now.where((results_now["ZeXt"]>0)) #filter also cases where Ze<0 and MDV<0
        elif i_filter==2:
            filterZeflux  = 20
            #filter small fluxes
            results_now = results.where((results["MDVxT"] * results["ZeXt"])  >filterZeflux)
            results_now = results_now.where((results_now["MDVxB"] * results_now["ZeXb"]/0.23)  >filterZeflux)
            #results_now = results_now.where((results_now["ZeXt"]>0)) #filter also cases where Ze<0 and MDV<0
            results_now = results_now.where((results_now["RHb"]>95.)) #filter also cases with low RH

        #2D loop to generate plot_matrix
        vars_dict    = get_vars(["ZFR"  ,"MDVxT","DWRxkT"],results_now)
        ZFRkey = "ZFR"
        MDVxT = vars_dict["MDVxT"]
        DWRxkT = vars_dict["DWRxkT"]
        ZFR = vars_dict[ZFRkey]

        ZFR_coll = categorize_part_type(ZFR.data,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_filter,"F")
        
        #uncomment next lines get other variables for categories
        key = "delz"
        var = get_vars([key],results_now)[key]
        var_coll = categorize_part_type(var.data,MDVxT.data,DWRxkT.data,dict(),key,i_filter,"F")

    # DATAFRAMES WITH TRIAL COLUMN ASSIGNED
    for i_filter in filterNumbers:
        ZFRs_filter = np.transpose(np.stack([ZFR_coll["F" + str(i_filter) + "unrimedZFR"],ZFR_coll["F" + str(i_filter) + "transitionalZFR"],ZFR_coll["F" + str(i_filter) + "rimedZFR"]]))
        df_now = pd.DataFrame(ZFRs_filter, columns=list(["unrimed","transitional","rimed"])).assign(Filter=filterDescr[i_filter] +  "\nN=" + "{:.1f}".format(ZFR_coll["F" + str(i_filter) + "Hdata"]) + "h" + "\n(" + "{:.1f}".format(ZFR_coll["F" + str(i_filter) + "unrimed" + "_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["F" + str(i_filter) + "transitional" + "_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["F" + str(i_filter) + "rimed" + "_perc"]*100)+ "%)")
        if i_filter==0:
            df = df_now.copy()
        else:
            df = pd.concat([df,df_now]) #, df2, df3])                                # CONCATENATE
    mdf = pd.melt(df, id_vars=['Filter'], var_name=['Particle Type'])      # MELT
    mdf.rename(columns={'value':'ZFR'}, inplace=True)

    ####plot
    ax = sns.boxplot(x="Filter", y="ZFR", hue="Particle Type", data=mdf,
        #showmeans=True,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"},
        showfliers = showfliers, flierprops={"marker":'o', "markersize":2,"alpha":0.01},
        ax=axes[0]) 

    if not ONLYbci:
        add_letters(ax,row=0)

    #add 1to1 melting line
    ax.axhline(1.0,c="magenta",lw=2)
    #limits and labels
    #ax.set_ylim([-0.5,1.25])
    ax.set_ylim([-1.0,3.0])


    ####################################
    #second plot: temporal averages#####
    ####################################


    averageNumbers = [0,1,2]
    averageDescr   = ["F1 + 2min av.","F1 + 5min av.","F1 + 10min av."]
    for i_average in averageNumbers: #loop over different vertical wind averages

        #select corrected MDV and ZFRs
        if i_average==0:
            av_min="2"
        elif i_average==1:
            av_min="5"
        elif i_average==2:
            av_min="10"
        results_av = results.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int(av_min)*15)).mean().copy() #min_periods=int(av_min*15/2)-> more than 50% in the averaging box must be non-nan
        #results_av = results.rolling(min_periods=2,center=True,time=10).mean().copy() #min_periods=int(av_min*15/2)-> more than 50% in the averaging box must be non-nan
        MDVkey = "MDVxT"
        ZFRkey = "ZFR"
        #load variables
        #filter small fluxes
        filterZeflux  = 20
        results_now = results_av.where((results_av["MDVxT"] * results_av["ZeXt"])  >filterZeflux).copy()
        #results_now = results_now.where((results_now["MDVxB"] * results_now["ZeXb"]/0.23)  >filterZeflux).copy()
        #results_now = results_now.where((results_now["ZeXt"]>0)).copy() #correction also cases where Ze<0 and MDV<0
        #filter low RH
        #results_now = results_now.where((results["RHb"]>95.)) #filter also cases with low RH
        vars_dict    = get_vars(["ZFR"  ,"MDVxT","DWRxkT"],results_now).copy()

        ZFRkey = "ZFR"
        MDVkey = "MDVxT"
        MDVxT = vars_dict[MDVkey]
        DWRxkT = vars_dict["DWRxkT"]
        ZFR = vars_dict[ZFRkey]

        ZFR_coll = categorize_part_type(ZFR.data,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_average,"A")

    # DATAFRAMES WITH TRIAL COLUMN ASSIGNED
    for i_average in averageNumbers:
        ZFRs_average = np.transpose(np.stack([ZFR_coll["A" + str(i_average) + "unrimed" + ZFRkey],ZFR_coll["A" + str(i_average) + "transitional" + ZFRkey],ZFR_coll["A" + str(i_average) + "rimed" + ZFRkey]]))
        df_now = pd.DataFrame(ZFRs_average, columns=list(["unrimed","transitional","rimed"])).assign(Average=averageDescr[i_average] +  "\nN=" + "{:.1f}".format(ZFR_coll["A" + str(i_average) + "Hdata"]) + "h" + "\n(" + "{:.1f}".format(ZFR_coll["A" + str(i_average) + "unrimed" + "_perc"]*100) + "%" + ",{:.1f}".format(ZFR_coll["A" + str(i_average) + "transitional_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["A" + str(i_average) + "rimed_perc"]*100)+ "%)")
        if i_average==0:
            df = df_now.copy()
        else:
            df = pd.concat([df,df_now]) #, df2, df3])                                # CONCATENATE
    mdf = pd.melt(df, id_vars=['Average'], var_name=['Particle Type'])      # MELT
    mdf.rename(columns={'value':'ZFR'}, inplace=True)

    ####plot
    ax = sns.boxplot(x="Average", y="ZFR", hue="Particle Type", data=mdf,
        #showmeans=True,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"},
        showfliers = showfliers, flierprops={"marker":'o', "markersize":2,"alpha":0.01},
        ax=axes[1])  # RUN PLOT   
    ax.legend([],[], frameon=False) #remove legend because it is already in first subplot
    ax.axhline(1.0,c="magenta",lw=2)
    #limits and labels
    #ax.set_ylim([0.0,0.6])
    ax.set_ylim([0.0,2.1])
    if not ONLYbci:
        add_letters(ax,row=1)

    ####################################
    #third plot: different corrections#
    ####################################
    #filter small fluxes
    filterZeflux  = 20
    results_now = results.where((results["MDVxT"] * results["ZeXt"])  >filterZeflux)
    results_now = results_now.where((results["MDVxB"] * results["ZeXt"]/0.23)  >filterZeflux)
    results_now = results_now.where((results["ZeXt"]>0)) #correction also cases where Ze<0 and MDV<0
    #filter low RH
    #results_now = results_now.where((results["RHb"]>95.)) #filter also cases with low RH

    #load variables
    vars_dict    = get_vars(["ZFR"  ,"ZFRvCorrT","ZFRvCorrB","ZFRvCorr","MDVxT","MDVvCorrT_denscorr","DWRxkT"],results_now)

    correctionNumbers = [0,1,2]
    correctionDescr   = ["F1 + ML Top","F1 + ML Bottom","F1 + ML Top & Bottom"]
    for i_correction in correctionNumbers: #loop over different vertical wind corrections

        #select corrected MDV and ZFRs
        if i_correction==0:
            MDVkey = "MDVvCorrT_denscorr"
            ZFRkey = "ZFRvCorrT"
        elif i_correction==1:
            MDVkey = "MDVxT"
            ZFRkey = "ZFRvCorrB"
        elif i_correction==2:
            MDVkey = "MDVvCorrT_denscorr"
            ZFRkey = "ZFRvCorr"
        DWRxkT = vars_dict["DWRxkT"]
        MDVxT = vars_dict[MDVkey]
        ZFR = vars_dict[ZFRkey]

        ZFR_coll = categorize_part_type(ZFR.data,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_correction,"C")

    # DATAFRAMES WITH TRIAL COLUMN ASSIGNED
    for i_correction in correctionNumbers:
        if i_correction==0:
            ZFRkey = "ZFRvCorrT"
        elif i_correction==1:
            ZFRkey = "ZFRvCorrB"
        elif i_correction==2:
            ZFRkey = "ZFRvCorr"
        ZFRs_correction = np.transpose(np.stack([ZFR_coll["C" + str(i_correction) + "unrimed" + ZFRkey],ZFR_coll["C" + str(i_correction) + "transitional" + ZFRkey],ZFR_coll["C" + str(i_correction) + "rimed" + ZFRkey]]))
        df_now = pd.DataFrame(ZFRs_correction, columns=list(["unrimed","transitional","rimed"])).assign(Corrections=correctionDescr[i_correction] +  "\nN=" + "{:.1f}".format(ZFR_coll["C" + str(i_correction) + "Hdata"]) + "h" + "\n(" + "{:.1f}".format(ZFR_coll["C" + str(i_correction) + "unrimed" + "_perc"]*100) + "%" + ",{:.1f}".format(ZFR_coll["C" + str(i_correction) + "transitional_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["C" + str(i_correction) + "rimed_perc"]*100)+ "%)")
        if i_correction==0:
            df = df_now.copy()
        else:
            df = pd.concat([df,df_now]) #, df2, df3])                                # CONCATENATE
    mdf = pd.melt(df, id_vars=['Corrections'], var_name=['Particle Type'])      # MELT
    mdf.rename(columns={'value':'ZFR'}, inplace=True)

    ####plot
    ax = sns.boxplot(x="Corrections", y="ZFR", hue="Particle Type", data=mdf,
        #showmeans=True,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"},
        showfliers = showfliers, flierprops={"marker":'o', "markersize":5,"alpha":0.01},
        ax=axes[2])  # RUN PLOT   
    ax.legend([],[], frameon=False) #remove legend because it is already in first subplot
    ax.axhline(1.0,c="magenta",lw=2)
    if not ONLYbci:
        add_letters(ax,row=2)

    #limits and labels
    #ax.set_yticks()

    ax.set_ylim([0.0,2.1])
    #ax.set_ylim([-0.5,1.25])

    for ax in axes.flatten():
        ml = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(ml)

        ax.grid(b=True,which="minor")
    plt.tight_layout() #rect=(0,0.00,1,1))
    #save figure
    savestr = 'plots/Boxplot'
    #if av_min!="0":
    #    savestr+= "_av" + str(av_min) + "min"
    plt.savefig(savestr + '.pdf')
    print("pdf is at: ",savestr + '.pdf')
    plt.clf()

if __name__ == '__main__':

    parser =  argparse.ArgumentParser(description='plot melting statistics')
    parser.add_argument('-d','--date', nargs=1, default=[""], help='gimme date e.g. 20190210 (if not given all data are used)')
    parser.add_argument('-av_min','--averageMin', nargs=1, default=["0"], help='how many minutes of rolling average?')
    parser.add_argument('-col','--calc_or_load', nargs=1, default=[0], help='1: calculate each day 0: load each day from previous calculation')
    parser.add_argument('-colprofiles','--calc_or_load_profiles', nargs=1, default=[0], help='1: calculate each day 0: load each day from previous calculation')
    parser.add_argument('-nmn','--no_mie_notch', nargs=1, default=[0], help='1: dont calculate mie-notch')

    args        = parser.parse_args()
    onlydate    = args.date[0]
    av_min      = args.averageMin[0]
    calc_or_load= int(args.calc_or_load[0])
    calc_or_load_profiles= int(args.calc_or_load_profiles[0])
    no_mie_notch= int(args.no_mie_notch[0])

    #get data
    results = get_observed_melting_properties(onlydate=onlydate,av_min=av_min,calc_or_load=calc_or_load,no_mie_notch=no_mie_notch) #calc_or_load: 1: calculate each day 0: load each day
    filterZeflux  = 999
    save_spec = get_save_string(onlydate,filterZeflux=filterZeflux)

    if av_min=="0" and onlydate!="":
        #plot_timeseries(results,save_spec) #illustrate averaging by plotting timeseries of one day
        pass 

    boxplot(results,av_min=av_min)

    #shuffle order to make scatter plot more objective (otherways last days are most visible)
    if onlydate=="":
        rand_index = np.random.permutation(results.ZFR.shape[0]).copy()
        time_shuffled = results.ZFR.time.isel(time=rand_index)
        results = results.reindex(time=time_shuffled).copy()

    #1. do some quality filters (rimed/unrimed, small fluxes)
    filterZeflux  = 20

    save_spec = get_save_string(onlydate,filterZeflux=filterZeflux)
    #filter small fluxes
    results = results.where((results["MDVxT"] * results["ZeXt"])  >filterZeflux)
    results = results.where((results["MDVxB"] * results["ZeXt"]/0.23)  >filterZeflux)
    results = results.where((results["ZeXt"]>0)) #filter also cases where Ze<0 and MDV<0
    #plot with different filters
    profiles(results,save_spec,av_min=av_min,col=calc_or_load_profiles,onlydate=onlydate,no_mie_notch=no_mie_notch)
    #profiles(results,save_spec,av_min=av_min,col=calc_or_load_profiles,onlydate=onlydate,no_mie_notch=no_mie_notch,correct_w_before_categorization=True)

    #filter RH
    RHbt = 95 #RH bigger than
    results = results.where((results["RHb"]>RHbt)) #filter also cases where Ze<0 and MDV<0
    save_spec = get_save_string(onlydate,filterZeflux=filterZeflux,RHbt=RHbt)

    #plots with RH filtered
    profiles(results,save_spec,av_min=av_min,col=calc_or_load_profiles,onlydate=onlydate,no_mie_notch=no_mie_notch)

    if onlydate!="":
        #plot_refl_flux_vs_allvars_scatter(results,save_spec,av_min,obs_pam="obs",plot_time=True)
        pass
