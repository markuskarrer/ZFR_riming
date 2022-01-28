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
import datetime
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

def dB(x): #conversion: linear [mm**6/m**3] to logarithmic [dB]
    return 10.0*np.log10(x)

def Bd(x): #conversion: logarithmic [dB] to linear [mm**6/m**3]
    return 10.0**(0.1*x)

def Z_R_MarshallPalmer(R): #reflectivity-rain relationship from Marshall and Palmer (1948)
    return 200*R**1.6 #Z in dB, R in mm**6/m**4

def R_Z_MarshallPalmer(Z): #rain-reflectivity relationship from Marshall and Palmer (1948)
    return (Z/200.)**(1./1.6) #Z in mm**6/m**3, R in mm/h

def attR_Matrosov(R): #relationship between attenuation and rain rate at X-Band
    return 0.048*R**1.05 #att in dB rain in mm/h

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
        #if int(date_now)>20190125: #TMP!
        #    continue
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

    results["RRfromZR"] = R_Z_MarshallPalmer(Bd(results["ZeXb"])) #get rain rate from Z-R relation
    results["attMelt"] = attR_Matrosov(results["RRfromZR"])
    results["FZtop"] = results["MDVxT"] * (Bd(results["ZeXt"]+results["attMelt"]))
    results["FZbot"] = results["MDVxB"] *  Bd(results["ZeXb"])

    K_factor = 0.23 
    results["ZFR"] = np.log10((results["MDVxB"] - 0.0) * Bd(results["ZeXb"]) / (results["MDVxT"]                     * (Bd(results["ZeXt"]+results["attMelt"])))  * K_factor)
    results["ZFRvCorrB"] = np.log10( (results["MDVxB"] - results["w_estB"]) * Bd(results["ZeXb"]) / (results["MDVxT"]                     * (Bd(results["ZeXt"]+results["attMelt"])))  * K_factor)
    results["ZFRvCorrT"] = np.log10(  results["MDVxB"]                     * Bd(results["ZeXb"]) / ((results["MDVxT"]-results["w_estT"])  * (Bd(results["ZeXt"]+results["attMelt"])))  * K_factor)

    results["ZFRvCorr"]  = np.log10(( results["MDVxB"] - results["w_estB"]) * Bd(results["ZeXb"]) /((results["MDVxT"]-results["w_estT"]) * (Bd(results["ZeXt"]+results["attMelt"])))  * K_factor)

    results["Delta_ZF_upper"]           =  (results["ZfluxX_thirdML"]) / (-results["MDVxT"] * results["ZeXt"]) #increase of reflectivity flux in upper (half/two thirds/...)
    results["Delta_ZF_lower"]           =  ((-results["MDVxB"]*Bd(results["ZeXb"]))- results["ZfluxX_thirdML"]) / (-results["MDVxT"] * Bd(results["ZeXt"])) #increase of reflectivity flux in upper (half/third/...)

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
        ZfluxX_halfML = MDV_half * Bd(Ze_half)
        #third
        MLthird = (heightBottonMaxLDR + offset_MLbot + (heightTopMaxLDR + offset_MLtop-heightBottonMaxLDR + offset_MLbot)/3)
        _, _, _, MDV_third, _ =         fc.getProp(rv_x, MLthird, 0)
        _, _, _, Ze_third, _ =         fc.getProp(dbz_x, MLthird, 0)
        ZfluxX_thirdML = MDV_third * Bd(Ze_third) 
        #fourth
        MLfourth = (heightBottonMaxLDR + offset_MLbot + (heightTopMaxLDR + offset_MLtop-heightBottonMaxLDR + offset_MLbot)/4)
        _, _, _, MDV_fourth, _ =         fc.getProp(rv_x, MLfourth, 0)
        _, _, _, Ze_fourth, _ =         fc.getProp(dbz_x, MLfourth, 0)
        ZfluxX_fourthML = MDV_fourth * Bd(Ze_fourth)
        #fifth
        MLfifth = (heightBottonMaxLDR + offset_MLbot + (heightTopMaxLDR + offset_MLtop-heightBottonMaxLDR + offset_MLbot)/5)
        _, _, _, MDV_fifth, _ =         fc.getProp(rv_x, MLfifth, 0)
        _, _, _, Ze_fifth, _ =         fc.getProp(dbz_x, MLfifth, 0)
        ZfluxX_fifthML = MDV_fifth * Bd(Ze_fifth) 

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
        #apply filters
        for key in tmpData:
            tmpData[key] = tmpData[key].where(tmpData["ML_thickness"]<700,drop=True).copy()

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
        fz_X = Bd(dbz_x) * (-rv_x)


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
        
        if not no_mie_notch: #possibility to deactivate this slow calculation of the mie-notch
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
            try: #sometimes w_estT and w_estB might be deactivated because this needs a lot of time
                tmpData = tmpData.dropna(dim="time", #to save memory remove timesteps where all relevant data is missing
                    subset=["FzNorm","LDR","ZeX","MDVx","DWRxk","DWRkw","DVedgeL","DVedgeH","w_estT","w_estB","FzNorm","FzGrad"],
                    how="all")
            except:
                tmpData = tmpData.dropna(dim="time", #to save memory remove timesteps where all relevant data is missing
                    subset=["FzNorm","LDR","ZeX","MDVx","DWRxk","DWRkw","DVedgeL","DVedgeH","FzNorm","FzGrad"],
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
    all_dic["MDVxT"]               = vars( "MDVxT_denscorr" ,"MDV$_{X,top}$ [m s$^{-1}$]"              ,[0.6,2.5]            )
    all_dic["MDVxT2"]              = vars( "MDVxT_denscorr" ,"MDV$_{X,top}$ [m s$^{-1}$]"              ,[0.6,2.5]            )#Quick and dirty fix to use variable two time
    all_dic["MDVxB"]               = vars( "MDVxB_denscorr" ,"MDV$_{X,bottom}$ [m s$^{-1}$]"           ,[3.0,8.0]            )
    all_dic["MDVvCorrT"]           = vars( "MDVvCorrT"      ,"MDV$_{X,v-corr,top}$ [m s$^{-1}$]"       ,[0.6,2.5]            )
    all_dic["MDVvCorrT_denscorr"]  = vars( "MDVvCorrT_denscorr","MDV$_{X,v-corr,top}$ [m s$^{-1}$]"       ,[0.6,2.5]            )
    all_dic["MDVvCorrT2"]          = vars( "MDVvCorrT"      ,"MDV$_{X,v-corr,top}$ [m s$^{-1}$]"       ,[0.6,2.5]            )#Quick and dirty fix to use variable two time
    all_dic["MDVvCorrT_denscorr2"] = vars( "MDVvCorrT_denscorr","MDV$_{X,v-corr,top}$ [m s$^{-1}$]"       ,[0.6,2.5]            )
    all_dic["MDVvCorrT_denscorr3"] = vars( "MDVvCorrT_denscorr","MDV$_{X,v-corr,top}$ [m s$^{-1}$]"       ,[0.6,2.5]            )
    all_dic["MDVvCorrB"]           = vars( "MDVvCorrB"      ,"MDV$_{X,v-corr,bottom}$ [m s$^{-1}$]"    ,[2.0,8.0]            )
    all_dic["MDVvCorrB_denscorr"]  = vars( "MDVvCorrB_denscorr"      ,"MDV$_{X,v-corr,bottom}$ [m s$^{-1}$]"    ,[2.0,9.0]            )
    all_dic["DV_edge_lowT"]        = vars( "DVedgeLT"       ,"DV$_{edge,low,top}$ [m s$^{-1}$]"        ,[-1.5,1.5]           )
    all_dic["DV_edge_highT"]       = vars( "DVedgeHT"       ,"DV$_{edge,high,top}$ [m s$^{-1}$]"       ,[1.0,6.0]            )
    all_dic["DV_edge_highB"]       = vars( "DVedgeHB"       ,"DV$_{edge,high,bottom}$ [m s$^{-1}$]"    ,[4.0,8.0]            )
    all_dic["w_estT"]              = vars( "w_estT"         ,"w$_{est.,top}$ [m s$^{-1}$]"             ,[-1.0,0.5]           ) #estimate of the wind velocity
    all_dic["w_estB"]              = vars( "w_estB"         ,"w$_{est.,bottom}$ [m s$^{-1}$]"          ,[-1.0,1.0]           ) #estimate of the wind velocity
    all_dic["peak1powT"]           = vars( "peak1powT"      ,"Ze$_{peak1,top}$ [dBz]"           ,[-60,0]              )
    all_dic["peak2powT"]           = vars( "peak2powT"      ,"Ze$_{peak2,top}$ [dBz]"           ,[-60,0]              )
    all_dic["widthKaT"]            = vars( "WkaT"           ,"SW$_{Ka}$ [m s$^{-1}$]"                  ,[0.1,1.0]            )
    all_dic["delz"]                = vars( "ML_thickness"   ,"$\Delta z_{melt}$ [m]"            ,[100.,600.]          )
    all_dic["rhT"]                 = vars( "RHt"            ,"RH$_{top}$ [%]"                   , [50,105]            )
    all_dic["rhB"]                 = vars( "RHb"            ,"RH$_{bottom}$ [%]"                , [70,105]            )
    all_dic["FZtop"]                 = vars( "FZtop"            ,"F$_{Z,top}$ [mm$^6$ m$^{-3}$ m s$^{-1}$]"                , [1e1,1e4]            )
    all_dic["FZbot"]                 = vars( "FZbot"            ,"F$_{Z,bottom}$ [mm$^6$ m$^{-3}$ m s$^{-1}]$"             , [1e1,1e4]            )

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
    all_dic["MDVx"]                = vars( "MDVx"           ,"MDV$_{X}$ [m s$^{-1}$]"              ,[1.0,7]            )
    all_dic["Fz"]                  = vars( "Fz"             ,"F$_{Z,X}$ [mm$^6$/m$^3$ m s$^{-1}$]"              ,[0,1e15]            )
    all_dic["DWRxk"]               = vars( "DWRxk"          ,"DWR$_{X,Ka}$ [dB]"             ,[-1.5,15]            )
    all_dic["DWRkw"]               = vars( "DWRkw"          ,"DWR$_{Ka,W}$ [dB]"             ,[-1.5,15]            )
    all_dic["FzNorm"]              = vars( "FzNorm"         ,"F$_{Z,X}$/F$_{Z,X,top}$"              ,[0.5,6]            )
    all_dic["FzGrad"]              = vars( "FzGrad"         ,"dF$_{Z,X,norm}$/d(-h)"              ,[-2,15]            )
    all_dic["DV_edge_low"]         = vars( "DVedgeL"        ,"SEV [m s$^{-1}$]"                       ,[-0.5,9]           )
    all_dic["DV_edge_high"]        = vars( "DVedgeH"        ,"DV$_{edges}$ [m s$^{-1}$]"       ,[-0.5,9]            )
    all_dic["w_estT"]              = vars( "w_estT"         ,"w$_{est.}$ [m s$^{-1}$]"             ,[-1.0,1.0]           ) #estimate of the wind velocity
    all_dic["w_estB"]              = vars( "w_estB"         ,"w$_{est.}$ [m s$^{-1}$]"          ,[-1.0,1.0]           ) #estimate of the wind velocity

    dic = dict()
    for key in vars_out:
        dic[key] = vars(all_dic[key].results_name,all_dic[key].plot_label,all_dic[key].lims)
        dic[key].get_data(res)

    return dic

def Profiles(results,save_spec,av_min="0",col=1,onlydate="",no_mie_notch=False,correct_w_before_categorization=False):
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
    #keys = ["DWRxk","DWRkw","MDVx","DV_edge_high","DV_edge_low","Fz","FzNorm","FzGrad","w_estT","w_estB"]
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
        #for i_ptype,ptype in enumerate(["unrimed","transitional","rimed"]):
        colors = ["blue","red"]
        for i_ptype,ptype in enumerate(["unrimed","rimed"]):
            ax.axhline(1.0,c="k",ls="--",lw=5)
            ax.axhline(0.0,c="k",ls="--",lw=5)
            if key=="FzNorm":
                ax.axvline(1/0.23,c="k",ls="--",lw=5)
            if key=="w_estB":
                varColl["F0"+ ptype + key] = varColl["F0"+ ptype + key].where(varColl["F0"+ ptype + key].range<0.0).copy() #crop mie-notch within ML
            low_quant = varColl["F0"+ ptype + key].quantile(0.25,dim="time")
            upp_quant = varColl["F0"+ ptype + key].quantile(0.75,dim="time")
            med_quant = varColl["F0"+ ptype + key].median(dim="time")
            if key=="FzGrad":
                low_quant = low_quant.rolling(min_periods=1,range=3).mean()
                upp_quant = upp_quant.rolling(min_periods=1,range=3).mean()
                med_quant = med_quant.rolling(min_periods=1,range=3).mean()
            ax.plot(low_quant,varColl["F0" + ptype +key].range,label="__None",color=colors[i_ptype],lw=1,ls="-",alpha=0.5)
            ax.plot(upp_quant,varColl["F0" + ptype +key].range,label="__None",color=colors[i_ptype],lw=1,ls="-",alpha=0.5)
            ax.plot(med_quant,varColl["F0" + ptype +key].range,label=ptype,c=colors[i_ptype],lw=5)
            ax.fill_betweenx(varColl["F0" + ptype +key].range,varColl["F0"+ ptype + key].quantile(0.25,dim="time"),varColl["F0"+ ptype + key].quantile(0.75,dim="time"),label="__None",color=colors[i_ptype],lw=1,ls="-",alpha=0.1)
            if key=="Fz":
                ax.set_xscale("log")
            #med_quant.isel(range=med_quant.argmax()) #get maximum
            #med_quant.sel(range=..,method="nearest") #get value at hrel=...
        
        #legend limits etc.
        if i_var==0:
            ax.legend()
        if i_var in [0,2,5,7]:
            ax.set_ylim([varColl["F0" + ptype +key].range[0],varColl["F0" + ptype +key].range[-1]])
            ax.set_ylabel("h$_{rel}$")

        ax.set_xlabel(var.plot_label)

        #add letters
        ax.text(0.0,1.0, letters[i_var],fontsize=52,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes, bbox=dict(facecolor='white', alpha=0.9))
        if key=="w_estB":
            ax.arrow(0.05, 0.08, 0.25, 0.0,lw=8)
            ax.arrow(-0.05, 0.08, -0.25, 0.0,lw=8)
            ax.text(0.05,0.15, "downdraft",fontsize=40,horizontalalignment='left',verticalalignment='center')
            ax.text(-0.05,0.15, "updraft" ,fontsize=40,horizontalalignment='right',verticalalignment='center')
 
            ax.set_xlim([-0.8,0.8])
        elif key=="MDVx":
            #add melting fraction according to Mitra
            from scipy.optimize import brentq
            for i_ptype,ptype in enumerate(["unrimed","rimed"]):
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
                ax2.plot(fmelt,varColl["F0" + ptype +key].range,label="__",c=colors[i_ptype],lw=2,ls="--")
                if i_ptype==0:
                    ax2.plot(np.nan,np.nan,ls="--",c="k",label="$f_{melt}$")
                    ax2.legend()
                #set_trace()
                #med_quant.range.values[np.nanargmin(np.abs(fmelt-0.5))] #get hrel[fmelt=0.5]
                
            
            #c=["blue","green","red"][i_ptype]
    ax2.set_xlabel("$f_{melt}$")
    #fig.delaxes(axes[1][3])
    #plt.tight_layout() #rect=(0,0.00,1,1))
    plt.subplots_adjust(left=0.1,
                    bottom=0.04, 
                    right=0.95, 
                    top=0.98, 
                    wspace=0.1, 
                    hspace=0.3)
    #save figure
    if "20" in onlydate:
        savestr = 'plots/days/profiles_ptype' + save_spec 
    else:
        savestr = 'plots/profiles_ptype' + save_spec 

    plt.savefig(savestr + '.pdf')
    plt.savefig(savestr + '.png')
    if correct_w_before_categorization:
        savestr += "_correct_w_before_categorization"
    plt.savefig(savestr + '.pdf')
    print("pdf is at: ",savestr + '.pdf')
    plt.clf()

def ProfilesLowHigFluxes(resultsAll,save_spec,av_min="0",col=1,onlydate="",no_mie_notch=False,filterZefluxLow=100,filterZefluxLow2=0,filterZefluxHig2=100,correct_w_before_categorization=False):
    '''
    plot profiles of several quantities vs. the normalized height within the melting layer
    filter first low fluxes then a range of fluxes
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


    #get variables
    keys = ["LDR","ZeX","MDVx","DV_edge_high","DV_edge_low","Fz","FzNorm","FzGrad","w_estT","w_estB"]
    axflat = axes.flatten()
    axflat = [axflat[0],axflat[1],axflat[2],axflat[3],axflat[3],axflat[4],axflat[5],axflat[6],axflat[7],axflat[7]] #put DV edges together in same plot
    letters=["a)","b)","c)","d)","d)","e)","f)","g)","h)","h)"] 
    #for i_fluxrange in [0]: 
    for i_fluxrange in [0,1]:

        if i_fluxrange==0: #filter low fluxes
            results = resultsAll
            results = results.where((results["MDVxT"] * Bd(results["ZeXt"]))  >filterZefluxLow)
            results = results.where((results["MDVxB"] * Bd(results["ZeXb"])*0.23)  >filterZefluxLow)
            #get variables for categorization
            vars_dict    = get_vars(["DWRxkT","MDVxT"],results)
            MDVxT = vars_dict["MDVxT"]
            DWRxkT = vars_dict["DWRxkT"]
            results = get_observed_melting_properties(onlydate=onlydate,calc_or_load=col,profiles=True,no_mie_notch=no_mie_notch) #calc_or_load: 1: calculate each day 0: load each day
            linestyle="-"
        elif i_fluxrange==1: #filter in range of fluxes
            results = resultsAll
            results = results.where((results["MDVxT"] * Bd(results["ZeXt"])     )   > filterZefluxLow2)
            results = results.where((results["MDVxB"] * Bd(results["ZeXb"])*0.23)   > filterZefluxLow2)
            results = results.where((results["MDVxT"] * Bd(results["ZeXt"])     )   < filterZefluxHigh2)
            results = results.where((results["MDVxB"] * Bd(results["ZeXb"])*0.23)   < filterZefluxHigh2)
            #get variables for categorization
            vars_dict    = get_vars(["DWRxkT","MDVxT"],results)
            MDVxT = vars_dict["MDVxT"]
            DWRxkT = vars_dict["DWRxkT"]
            results = get_observed_melting_properties(onlydate=onlydate,calc_or_load=col,profiles=True,no_mie_notch=no_mie_notch) #calc_or_load: 1: calculate each day 0: load each day
            linestyle="--"
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
            colors = ["blue","red"]
            for i_ptype,ptype in enumerate(["unrimed","rimed"]):
                if i_fluxrange==0:
                    ax.axhline(1.0,c="k",ls="--",lw=5)
                    ax.axhline(0.0,c="k",ls="--",lw=5)
                if key=="FzNorm":
                    ax.axvline(1/0.23,c="k",ls="--",lw=5)
                if key=="w_estB":
                    varColl["F0"+ ptype + key] = varColl["F0"+ ptype + key].where(varColl["F0"+ ptype + key].range<0.0).copy() #crop mie-notch within ML
                low_quant = varColl["F0"+ ptype + key].quantile(0.25,dim="time")
                upp_quant = varColl["F0"+ ptype + key].quantile(0.75,dim="time")
                med_quant = varColl["F0"+ ptype + key].median(dim="time")
                if key=="FzGrad":
                    low_quant = low_quant.rolling(min_periods=1,range=3).mean()
                    upp_quant = upp_quant.rolling(min_periods=1,range=3).mean()
                    med_quant = med_quant.rolling(min_periods=1,range=3).mean()

                if key in ["Fz"]:
                    ax.set_xscale('log')
                ax.plot(low_quant,varColl["F0" + ptype +key].range,label="__None",color=colors[i_ptype],lw=1,ls=linestyle,alpha=0.5)
                ax.plot(upp_quant,varColl["F0" + ptype +key].range,label="__None",color=colors[i_ptype],lw=1,ls=linestyle,alpha=0.5)
                if i_fluxrange==0:
                    label=ptype
                else:
                    label="__None"
                ax.plot(med_quant,varColl["F0" + ptype +key].range,label=label,c=colors[i_ptype],lw=5,ls=linestyle)
                ax.fill_betweenx(varColl["F0" + ptype +key].range,varColl["F0"+ ptype + key].quantile(0.25,dim="time"),varColl["F0"+ ptype + key].quantile(0.75,dim="time"),label="__None",color=colors[i_ptype],lw=1,ls=linestyle,alpha=0.1)
            

            #legend limits etc.
            if i_fluxrange==1:  #modify labels etc. and save only in second round
                if i_var==0:
                    ax.legend()
                if i_var in [0,2,5,7]:
                    ax.set_ylim([varColl["F0" + ptype +key].range[0],varColl["F0" + ptype +key].range[-1]])
                    ax.set_ylabel("h$_{rel}$")
                #ax.set_xlim(var.lims)
                ax.set_xlabel(var.plot_label)

                #add letters
                ax.text(0.0,1.0, letters[i_var],fontsize=52,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes, bbox=dict(facecolor='white', alpha=0.9))
            if key=="w_estB":
                ax.arrow(0.05, 0.08, 0.25, 0.0,lw=8)
                ax.arrow(-0.05, 0.08, -0.25, 0.0,lw=8)
                ax.text(0.05,0.15, "downdraft",fontsize=40,horizontalalignment='left',verticalalignment='center')
                ax.text(-0.05,0.15, "updraft",fontsize=40,horizontalalignment='right',verticalalignment='center')
            elif key=="MDVx":
                #add melting fraction according to Mitra
                from scipy.optimize import brentq
                for i_ptype,ptype in enumerate(["unrimed","rimed"]):
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
                    if i_fluxrange==0:
                        ax2 = ax.twiny()
                    ax2.plot(fmelt,varColl["F0" + ptype +key].range,label="__",c=colors[i_ptype],lw=2,ls=linestyle)
                    if i_ptype==0:
                        ax2.plot(np.nan,np.nan,ls="-",c="k",label="$f_{melt}$")
                        ax2.legend()
                   
        axes[0][0].plot(np.nan,np.nan,label="F$_Z$>" + str(filterZefluxLow),ls="-",c="k",lw=5)
        axes[0][0].plot(np.nan,np.nan,label=str(filterZefluxLow2) +"<$\,$F$_Z$$\,$<" + str(filterZefluxHigh2),ls="--",c="k",lw=5)
        if i_fluxrange==1:  #modify labels etc. and save only in second round
            ax2.set_xlabel("$f_{melt}$")

            plt.subplots_adjust(left=0.1,
                            bottom=0.03, 
                            right=0.95, 
                            top=0.98, 
                            wspace=0.1, 
                            hspace=0.3)
            #save figure
            if "20" in onlydate:
                savestr = 'plots/days/profiles_ptype' + save_spec 
            else:
                savestr = 'plots/profiles_ptype' + save_spec 

            plt.savefig(savestr + '.pdf')
            plt.savefig(savestr + '.png')
            if correct_w_before_categorization:
                savestr += "_correct_w_before_categorization"
            plt.savefig(savestr + '.pdf')
            print("pdf is at: ",savestr + '.pdf')
            plt.clf()

def Boxplot(results,av_min="0",showfliers=False,ONLYbci=False,day=None,filterZeflux=30,addWsubsetWithoutCorr=False):
    '''
    plot boxplots after different filtering (Zeflux,RH) and corrections (vertical wind estimate)
    showfliers: show outliers of boxplot
    ONLYbci: show only panels b) c) and i) (for presentation)
    day: None: all days; otherways date in YYYYMMDD
    filterZeflux: filter of reflectivity flux applied to several boxplot panels
    addWsubsetWithoutCorr: add one row of plots with subset of data where w-correction is possible but not applied. 
        This is done to test whether the subset is specific or whether the correction is making the difference.
    '''
    import seaborn as sns
    from matplotlib.ticker import MultipleLocator
    from statsmodels.stats.weightstats import ztest
    from scipy.stats import ttest_1samp,skew,kurtosis,wilcoxon
    from sklearn import preprocessing

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
        elif row==3:
            letters=["j)","k)","l)"] 

        #make labels
        ax.text(0.0,0.9, letters[0],fontsize=42,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
        ax.text(0.33,0.9, letters[1],fontsize=42,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)
        ax.text(0.67,0.9, letters[2],fontsize=42,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes)

    if not ONLYbci:
        if addWsubsetWithoutCorr:
            fig,axes = plt.subplots(ncols=1,nrows=4,figsize=(12,26))
        else:
            fig,axes = plt.subplots(ncols=1,nrows=3,figsize=(12,20))
    else: ###part for presentation (simplified plot)
        pass #commented not to be confused every time
        #fig,axes = plt.subplots(ncols=1,nrows=1,figsize=(12,7))
        #ZFR_coll   = dict() #collection of all ZFRs
        #filterNumbers = [0,1,2]
        #filterDescr   = ["F1: F$_{Z,X}$>" + str(filterZeflux) + "mm$^6$ m$^{-3}$ m s$^{-1}$","F2: F1 & RH>95%","F1 + w-corr"]
        #for i_filter in filterNumbers:
        #    if i_filter in [0,2]:
        #        #filter small fluxes
        #        results_now = results.where((results["MDVxT"] * Bd(results["ZeXt"]))  >filterZeflux)
        #        results_now = results_now.where((results_now["MDVxB"] * Bd(results_now["ZeXb"])*0.23)  >filterZeflux)
        #    elif i_filter==1:
        #        #filter small fluxes
        #        results_now = results.where((results["MDVxT"] * Bd(results["ZeXt"]))  >filterZeflux)
        #        results_now = results_now.where((results_now["MDVxB"] * Bd(results_now["ZeXb"])*0.23)  >filterZeflux)
        #        results_now = results_now.where((results_now["RHb"]>95.)) #filter also cases with low RH

        #    #2D loop to generate plot_matrix
        #    if i_filter in [0,1]:
        #        vars_dict    = get_vars(["ZFR"  ,"MDVxT","DWRxkT"],results_now)
        #        MDVkey = "MDVxT"
        #        ZFRkey = "ZFR"
        #    else:
        #        vars_dict    = get_vars(["ZFR"  ,"ZFRvCorrT","ZFRvCorrB","ZFRvCorr","MDVxT","MDVvCorrT_denscorr","DWRxkT"],results_now)
        #        MDVkey = "MDVvCorrT_denscorr"
        #        ZFRkey = "ZFRvCorr"

        #    MDVxT = vars_dict[MDVkey]
        #    DWRxkT = vars_dict["DWRxkT"]
        #    ZFR = vars_dict[ZFRkey] 

        #    ZFR_coll = categorize_part_type(ZFR.data,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_filter,"F")
        #    
        #    #uncomment next lines get other variables for categories
        #    key = "delz"
        #    var = get_vars([key],results_now)[key]
        #    var_coll = categorize_part_type(var.data,MDVxT.data,DWRxkT.data,dict(),key,i_filter,"F")

        ## DATAFRAMES WITH TRIAL COLUMN ASSIGNED
        #for i_filter in filterNumbers:
        #    if i_filter in [0,1]:
        #        ZFRkey = "ZFR"
        #    else:
        #        ZFRkey = "ZFRvCorr"
        #    ZFRs_filter = np.transpose(np.stack([ZFR_coll["F" + str(i_filter) + "unrimed" + ZFRkey],ZFR_coll["F" + str(i_filter) + "transitional" + ZFRkey],ZFR_coll["F" + str(i_filter) + "rimed" + ZFRkey]]))
        #    df_now = pd.DataFrame(ZFRs_filter, columns=list(["unrimed","transitional","rimed"])).assign(Filter=filterDescr[i_filter] +  "\nN=" + "{:.1f}".format(ZFR_coll["F" + str(i_filter) + "Hdata"]) + "h" + "\n(" + "{:.1f}".format(ZFR_coll["F" + str(i_filter) + "unrimed" + "_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["F" + str(i_filter) + "transitional" + "_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["F" + str(i_filter) + "rimed" + "_perc"]*100)+ "%)")
        #    if i_filter==0:
        #        df = df_now.copy()
        #    else:
        #        df = pd.concat([df,df_now]) #, df2, df3])                                # CONCATENATE
        #mdf = pd.melt(df, id_vars=['Filter'], var_name=['Particle Type'])      # MELT
        #mdf.rename(columns={'value':'ZFR'}, inplace=True)


        #####plot
        #ax = sns.boxplot(x="Filter", y="ZFR", hue="Particle Type", data=mdf,
        #    #showmeans=False,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"},
        #    showfliers = showfliers, flierprops={"marker":'o', "markersize":2,"alpha":0.01},
        #    ax=axes) 
        #ax.axhline(0.0,c="magenta",lw=2)

        #plt.tight_layout() #rect=(0,0.00,1,1))
        ##save figure
        #savestr = 'plots/Boxplot4pres'
        ##if av_min!="0":
        ##    savestr+= "_av" + str(av_min) + "min"
        #plt.savefig(savestr + '.pdf')
        #print("pdf is at: ",savestr + '.pdf')
        #return
        ####END: part for presentation (simplified plot)

    ZFR_coll   = dict() #collection of all ZFRs
    filterNumbers = [0,1,2]
    filterDescr   = ["none","F1: F$_{Z,X}$>" + str(filterZeflux) + "$mm^6 m^{-3} m s^{-1}$","F2: F1 & RH>95%"]
    for i_filter in filterNumbers:
        if i_filter==0:
            results_now = results #no filter
        elif i_filter==1:
            #filter small fluxes
            results_now = results.where((results["MDVxT"] * Bd(results["ZeXt"]))  >filterZeflux)
            results_now = results_now.where((results_now["MDVxB"] * Bd(results_now["ZeXb"])*0.23)  >filterZeflux)
        elif i_filter==2:
            #filter small fluxes
            results_now = results.where((results["MDVxT"] * Bd(results["ZeXt"]))  >filterZeflux)
            results_now = results_now.where((results_now["MDVxB"] * Bd(results_now["ZeXb"])*0.23)  >filterZeflux)
            results_now = results_now.where((results_now["RHb"]>95.)) #filter also cases with low RH

        #2D loop to generate plot_matrix
        vars_dict    = get_vars(["ZFR"  ,"MDVxT","DWRxkT"],results_now)
        ZFRkey = "ZFR"
        MDVxT = vars_dict["MDVxT"]
        DWRxkT = vars_dict["DWRxkT"]
        ZFR = vars_dict[ZFRkey]


        ZFR_coll = categorize_part_type(ZFR.data,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_filter,"F")
        DWR_coll = categorize_part_type(DWRxkT.data,MDVxT.data,DWRxkT.data,ZFR_coll,"DWRxkT",i_filter,"F")

        #uncomment next lines get other variables for categories
        #key = "delz"
        #var = get_vars([key],results_now)[key]
        #var_coll = categorize_part_type(var.data,MDVxT.data,DWRxkT.data,dict(),key,i_filter,"F")

        
    # DATAFRAMES WITH TRIAL COLUMN ASSIGNED
    for i_filter in filterNumbers:
        #write out some the mean properties of some of the subpanels
        print("unrimed F" + str(i_filter) + "; median(DWRxkT)",np.nanmedian(DWR_coll["F" + str(i_filter) + "unrimedDWRxkT"]),"90perc(DWRxkT)",np.nanpercentile(DWR_coll["F" + str(i_filter) + "unrimedDWRxkT"],90)) #,"N(DWRxkT)",sum(~np.isnan(DWR_coll["F" + str(i_filter) + "unrimedDWRxkT"])).values)

        ZFRs_filter = np.transpose(np.stack([ZFR_coll["F" + str(i_filter) + "unrimedZFR"],ZFR_coll["F" + str(i_filter) + "transitionalZFR"],ZFR_coll["F" + str(i_filter) + "rimedZFR"]]))
        df_now = pd.DataFrame(ZFRs_filter, columns=list(["unrimed","transitional","rimed"])).assign(Filter=filterDescr[i_filter] +  "\nN=" + "{:.1f}".format(ZFR_coll["F" + str(i_filter) + "Hdata"]) + "h" + "\n(" + "{:.1f}".format(ZFR_coll["F" + str(i_filter) + "unrimed" + "_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["F" + str(i_filter) + "transitional" + "_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["F" + str(i_filter) + "rimed" + "_perc"]*100)+ "%)")
        if i_filter==0:
            df = df_now.copy()
        else:
            df = pd.concat([df,df_now]) #, df2, df3])                                # CONCATENATE
        for key in ["unrimed","transitional","rimed"]:
            true_value = 0.0 #for the z-test we assume ZFR=0.0 (melting-only) as the null-hypothesis
  
            normed_data = preprocessing.normalize([df[key].dropna()])[0]
            if i_filter==1 and key=="unrimed":
                pass
                #set_trace()
            dfNow = df[key].dropna()
            #print("Filter:",i_filter,key,"z-test, p-value",ztest(df[key].dropna(), value=true_value))
            #print("Filter:",i_filter,key,"t-test, p-value",ttest_1samp(dfNow, popmean=true_value),"skew",skew(dfNow),"kurto",kurtosis(dfNow,fisher=False))
            #print("Filter:",i_filter,key,"W-test, p-value",wilcoxon(dfNow))

    mdf = pd.melt(df, id_vars=['Filter'], var_name=['Particle Type'])      # MELT
    mdf.rename(columns={'value':'ZFR'}, inplace=True)

    ####plot
    ax = sns.boxplot(x="Filter", y="ZFR", hue="Particle Type", data=mdf,
        #showmeans=False,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"},
        showfliers = showfliers, flierprops={"marker":'o', "markersize":2,"alpha":0.01},
        ax=axes[0],showmeans=False) 

    if not ONLYbci:
        add_letters(ax,row=0)

    #add 1to1 melting line
    ax.axhline(0.0,c="magenta",lw=2)

    #limits and labels
    ax.set_ylim([-1.5,1.5])


    ####################################
    #second plot: temporal averages#####
    ####################################


    averageNumbers = [0,1,2]
    averageDescr   = ["F1 + 1min av.","F1 + 2min av.","F1 + 5min av."]
    for i_average in averageNumbers: #loop over different vertical wind averages

        #select corrected MDV and ZFRs
        if i_average==0:
            av_min="1"
        elif i_average==1:
            av_min="2"
        elif i_average==2:
            av_min="5"

        #filter small fluxes
        results_now = results.where(     (results["MDVxT"]    * Bd(results["ZeXt"])     )     > filterZeflux).copy()
        results_now = results_now.where(     (results_now["MDVxB"]    * Bd(results_now["ZeXb"])*0.23)     > filterZeflux).copy()

        results_av = results_now.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int(av_min)*15)).mean().copy() #min_periods=int(av_min*15/2)-> more than 50% in the averaging box must be non-nan
        MDVkey = "MDVxT"
        ZFRkey = "ZFR"
        #load variables
        vars_dict    = get_vars(["ZFR"  ,"MDVxT","DWRxkT"],results_av).copy()

        ZFRkey = "ZFR"
        MDVkey = "MDVxT"
        MDVxT = vars_dict[MDVkey]
        DWRxkT = vars_dict["DWRxkT"]
        ZFR = vars_dict[ZFRkey]

        ZFR_coll = categorize_part_type(ZFR.data,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_average,"A")

        if False: #TMP! debug ZFR values
            ax.text(0.5,1.5,"rim low {:.2f} ".format(np.nanpercentile(ZFR_coll["F1rimedZFR"],25))+"med{:.2f} ".format(np.nanmedian(ZFR_coll["F1rimedZFR"]))+"hig{:.2f}".format(np.nanpercentile(ZFR_coll["F1rimedZFR"],75)))
            ax.text(0.5,1.0,"trans low {:.2f} ".format(np.nanpercentile(ZFR_coll["F1transitionalZFR"],25))+"med{:.2f} ".format(np.nanmedian(ZFR_coll["F1transitionalZFR"]))+"hig{:.2f}".format(np.nanpercentile(ZFR_coll["F1transitionalZFR"],75)))
            ax.text(0.5,0.5,"unr low {:.2f} ".format(np.nanpercentile(ZFR_coll["F1unrimedZFR"],25))+"med{:.2f} ".format(np.nanmedian(ZFR_coll["F1unrimedZFR"]))+"hig{:.2f}".format(np.nanpercentile(ZFR_coll["F1unrimedZFR"],75)))

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
        #showmeans=False,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"},
        showfliers = showfliers, flierprops={"marker":'o', "markersize":2,"alpha":0.01},
        ax=axes[1],showmeans=False)  # RUN PLOT   
    ax.legend([],[], frameon=False) #remove legend because it is already in first subplot
    ax.axhline(0.0,c="magenta",lw=2)
    #limits and labels
    ax.set_ylim([-1.0,1.0])

    if not ONLYbci:
        add_letters(ax,row=1)

    ####################################
    #third plot: different wind corrections#
    ####################################
    #filter small fluxes
    #filterZeflux  = 100
    results_now = results.where((results["MDVxT"] * Bd(results["ZeXt"]))  >filterZeflux)
    results_now = results_now.where((results["MDVxB"] * Bd(results["ZeXb"])*0.23)  >filterZeflux)
    #filter low RH
    #results_now = results_now.where((results["RHb"]>95.)) #filter also cases with low RH

    #load variables
    vars_dict    = get_vars(["ZFR"  ,"ZFRvCorrT","ZFRvCorrB","ZFRvCorr","MDVxT","MDVvCorrT_denscorr","MDVvCorrB_denscorr","DWRxkT"],results_now)

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
        #MDVkey = "MDVxT" #vCorrT_denscorr"
        DWRxkT = vars_dict["DWRxkT"]
        MDVxT = vars_dict[MDVkey]
        ZFR = vars_dict[ZFRkey]

        ZFR_coll = categorize_part_type(ZFR.data,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_correction,"C")
        DWRxkTnow = DWRxkT
        DWRxkTnow.data = DWRxkT.data.where(~np.isnan(ZFR.data)).copy()
        DWR_collC = categorize_part_type(DWRxkTnow.data,MDVxT.data,DWRxkT.data,ZFR_coll,"DWRxkT",i_correction,"C")


        #print("i_correction",i_correction,ZFRkey,"N(ZFR)",sum(~np.isnan(ZFR.data.values)),"N(ZFRkey)",sum(~np.isnan(vars_dict[ZFRkey].data.values)))
    # DATAFRAMES WITH TRIAL COLUMN ASSIGNED
    for i_correction in correctionNumbers:
        #write out some the mean properties of some of the subpanels
        print("unrimed C" + str(i_correction) + "; median(DWRxkT)",np.nanmedian(DWR_collC["C" + str(i_correction) + "unrimedDWRxkT"]),"90perc(DWRxkT)",np.nanpercentile(DWR_collC["C" + str(i_correction) + "unrimedDWRxkT"],90)) #,"N(DWRxkT)",sum(~np.isnan(DWR_collC["C" + str(i_correction) + "unrimedDWRxkT"])).values)
        #write out some the mean properties of some of the subpanels #comment out to make script faster?
        #ZFR_tmp = categorize_part_type(DWRxkT.data,MDVxT.data,DWRxkT.data,ZFR_coll,"DWRxkT",i_correction,"C")
        #print("unrimed C" +str(i_correction) + "ML Top; median(DWRxkT)",np.nanmedian(ZFR_tmp["C0unrimedDWRxkT"]),"90perc(DWRxkT)",np.nanpercentile(ZFR_tmp["C0unrimedDWRxkT"],90),"95perc(DWRxkT)",np.nanpercentile(ZFR_tmp["C0unrimedDWRxkT"],95))
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
        #showmeans=False,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"},
        showfliers = showfliers, flierprops={"marker":'o', "markersize":5,"alpha":0.01},
        ax=axes[2],showmeans=False)  # RUN PLOT   
    ax.legend([],[], frameon=False) #remove legend because it is already in first subplot
    ax.axhline(0.0,c="magenta",lw=2)
    if not ONLYbci:
        add_letters(ax,row=2)

    ax.set_ylim([-1.0,1.0])

    if addWsubsetWithoutCorr:

        ####################################
        #fourth plot: subset of data where wind-correction would be possible but is not applied
        ####################################
        #filter small fluxes
        results_now = results.where(    (results["MDVxT"] * Bd(results["ZeXt"]))       >filterZeflux)
        results_now = results_now.where((results["MDVxB"] * Bd(results["ZeXb"])*0.23)  >filterZeflux)

        #load variables
        vars_dict    = get_vars(["ZFR"  ,"ZFRvCorrT","ZFRvCorrB","ZFRvCorr","MDVxT","MDVvCorrT_denscorr","MDVvCorrB_denscorr","DWRxkT"],results_now)

        correctionNumbers = [0,1,2]
        correctionDescr   = ["F1 + ML Top","F1 + ML Bottom","F1 + ML Top & Bottom"]

        ZFR_coll = categorize_part_type(ZFR.data,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_correction,"C")
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
            DWRxkT  = vars_dict["DWRxkT"]
            MDVxT   = vars_dict[MDVkey]
            ZFR     = vars_dict["ZFR"].data.copy()

            #remove all datapoints where <ZFRkey> is not defined, which means that it cant be corrected for w
            ZFR.values = np.where(~np.isnan(vars_dict[ZFRkey].data.values),vars_dict["ZFR"].data.values,np.nan)
            ZFR_coll = categorize_part_type(ZFR,MDVxT.data,DWRxkT.data,ZFR_coll,ZFRkey,i_correction,"W")

            # DATAFRAMES WITH TRIAL COLUMN ASSIGNED
            ZFRs_correction = np.transpose(np.stack([ZFR_coll["W" + str(i_correction) + "unrimed" + ZFRkey],ZFR_coll["W" + str(i_correction) + "transitional" + ZFRkey],ZFR_coll["W" + str(i_correction) + "rimed" + ZFRkey]]))
            df_now = pd.DataFrame(ZFRs_correction, columns=list(["unrimed","transitional","rimed"])).assign(Subset=correctionDescr[i_correction] +  "\nN=" + "{:.1f}".format(ZFR_coll["W" + str(i_correction) + "Hdata"]) + "h" + "\n(" + "{:.1f}".format(ZFR_coll["W" + str(i_correction) + "unrimed" + "_perc"]*100) + "%" + ",{:.1f}".format(ZFR_coll["W" + str(i_correction) + "transitional_perc"]*100)+ "%" + ",{:.1f}".format(ZFR_coll["W" + str(i_correction) + "rimed_perc"]*100)+ "%)")
            if i_correction==0:
                df = df_now.copy()
            else:
                df = pd.concat([df,df_now]) #, df2, df3])                                # CONCATENATE
        mdf = pd.melt(df, id_vars=['Subset'], var_name=['Particle Type'])      # MELT
        mdf.rename(columns={'value':'ZFR'}, inplace=True)

        ####plot
        ax = sns.boxplot(x="Subset", y="ZFR", hue="Particle Type", data=mdf,
            #showmeans=False,meanprops={"marker":"o","markerfacecolor":"white", "markeredgecolor":"black","markersize":"10"},
            showfliers = showfliers, flierprops={"marker":'o', "markersize":5,"alpha":0.01},
            ax=axes[3],showmeans=False)  # RUN PLOT   
        ax.legend([],[], frameon=False) #remove legend because it is already in first subplot
        ax.axhline(0.0,c="magenta",lw=3)
        if not ONLYbci:
            add_letters(ax,row=3)

        ax.set_ylim([-1.0,1.0])

    for ax in axes.flatten():
        ml = MultipleLocator(0.1)
        ax.yaxis.set_minor_locator(ml)

        ax.grid(b=True,which="minor")
    plt.tight_layout() #rect=(0,0.00,1,1))
    #save figure
    #if av_min!="0":
    #    savestr+= "_av" + str(av_min) + "min"
    if "20" in day:
        savestr = 'plots/days/Boxplot' + day + "Fz" + str(filterZeflux)
    else:
        savestr = 'plots/Boxplot' + "Fz" +  str(filterZeflux)

    plt.savefig(savestr + '.pdf')
    plt.savefig(savestr + '.png')
    print("pdf is at: ",savestr + '.pdf')
    plt.clf()

def crosscorr(datax, datay, lag=0):
    """ Lag-N cross correlation. 
    Parameters
    ----------
    lag : int, default 0
    datax, datay : pandas.Series objects of equal length

    Returns
    ----------
    crosscorr : float
    """
    return datax.corr(datay.shift(lag))

def crosscorrAllDay(FZtop,FZbot,Delta_t):
    '''
    Make a cross-correlation between the refl. flux ...
        ... at the ML top (FZtop) ...
        .... and ML bottom (FZbot)
        Delta_t: time step in data [probably 4s]
    '''

    #simple correlation
    corr_non_shifted = FZtop.corr(FZbot)

    best_corr = [0,corr_non_shifted] #[position,corr-coeff]
    for lag in range(-75,75,1): #this goes 5 minutes back-/forward
        lag_s = Delta_t*lag
        corr_shifted = crosscorr(FZtop,FZbot,lag=lag)
        if corr_shifted>best_corr[1]:
            best_corr = [lag_s,corr_shifted]

    return best_corr

def plot_timeseries(results,save_spec):
    '''
    plot timeseries
    '''
    import matplotlib.dates as mdates

    vars_dict = get_vars(["FZtop","FZbot"],results)

    vars_dict["FZtop"].data = vars_dict["FZtop"].data.sel(time=slice("2019-01-13T05:30:00","2019-01-13T09:30:00"))
    vars_dict["FZbot"].data = vars_dict["FZbot"].data.sel(time=slice("2019-01-13T05:30:00","2019-01-13T09:30:00"))

    #set-up figure
    nrows   = 7
    fig,axes = plt.subplots(nrows=nrows,figsize=(12,nrows*2),sharex=True)

    #plot not-averaged timeseries
    ax = axes[0]
    for i_var,key in enumerate(vars_dict.keys()):
        var = vars_dict[key]
        #plot timeseries
        ax.semilogy(var.data.time.values,var.data.values,lw=1,color=["r","b"][i_var],label=["top","bottom"][i_var])
        ax.set_ylabel("F$_{Z}$ [mm$^{6}$ m$^{-3}$ m s$^{-1}$]")
        #apply limits
        ax.set_ylim([1e-1,1e4])
        #loop over average windows
        for i_av,av_min in enumerate([]): #"1","5","10"]):
            #for i_var,(key,ax) in enumerate(zip(vars_dict.keys(),axes.flatten())):
            if not var.results_name.startswith("peak"):
                if "s" in av_min:
                    av_sec = int(av_min[:-1])
                    var_av = var.data.rolling(min_periods=int(av_sec/8),center=True,time=int(av_sec/4)).mean().copy() #min_periods=av_min/8 -> more than 50% in the averaging box must be non-nan
                else:
                    var_av = var.data.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int(av_min)*15)).mean().copy() #min_periods=int(av_min*15/2)-> more than 50% in the averaging box must be non-nan
                ax.semilogy(var_av.time.values,var_av,lw=1,color=["r","b"][i_var],label=["top","bottom"][i_var])
    ax.legend() 

    #plot ZFR
    ZFR_orig = np.log10(vars_dict["FZbot"].data.values/vars_dict["FZtop"].data.values)
    axes[1].plot(vars_dict["FZbot"].data.time.values,ZFR_orig,lw=1,label="no av.",c="k")
    axes[1].set_ylabel("ZFR (not shifted)")
    axes[1].set_ylim([-1.5,1.5])

    #####################
    ####Auto-correlations
    #####################
    if True:

        #first bring data in right format
        FZtop = vars_dict["FZtop"].data.to_series()
        FZbot = vars_dict["FZbot"].data.to_series()
        Delta_t = np.array((vars_dict["FZtop"].data.time[1]-vars_dict["FZtop"].data.time[0]).astype(int)/1e9)

        ############### 
        #full day - find best correlation of original and shifted data
        ###############
        av_periods = 30 #smooth timeseries before calculating the cross-correlation
        FZtopAv = FZtop.rolling(av_periods, min_periods=1).mean()
        FZbotAv = FZbot.rolling(av_periods, min_periods=1).mean()
        best_corr = crosscorrAllDay(FZtopAv,FZbotAv,Delta_t)
        ind_shift = int(best_corr[0]/Delta_t)
        print("best correlation full day with lag ",best_corr[0], "s :", best_corr[1])

        ax = axes[2]
        for i_var,key in enumerate(["FZtop","FZbotShifted"]):
            #shift FZbottom in time
            if key=="FZbotShifted":
                vars_dict["FZbotShifted"] = vars_dict["FZbot"]
                vars_dict["FZbotShifted"].data["time"]= vars_dict["FZbotShifted"].data.time + pd.Timedelta(seconds=best_corr[0]*Delta_t)
                FZbot = vars_dict["FZbot"].data.to_series()
            var = vars_dict[key]

            #plot timeseries
            ax.semilogy(var.data.time.values,var.data.values,lw=1,color=["r","b"][i_var],label=["top","bottom ($\Delta$t="+ str(best_corr[0]) +"s)"][i_var])
            ax.set_ylabel("F$_{Z}$ [mm$^{6}$ m$^{-3}$ m s$^{-1}$]")
            #apply limits
            ax.set_ylim([1e-1,1e4])
        ax.legend()

        #plot shifted ZFR
        if ind_shift==0:
            ZFR_shifted = np.log10(vars_dict["FZbotShifted"].data.values/vars_dict["FZtop"].data.values)
            axes[3].plot(vars_dict["FZtop"].data.time.values,ZFR_shifted,lw=1,c="k")
        elif ind_shift<0:
            ZFR_shifted = np.log10(vars_dict["FZbotShifted"].data.values[-ind_shift:]/vars_dict["FZtop"].data.values[:ind_shift])
            axes[3].plot(vars_dict["FZtop"].data.time.values[:ind_shift],ZFR_shifted,lw=1,c="k")
        elif ind_shift>0:
            ZFR_shifted = np.log10(vars_dict["FZbotShifted"].data.values[ind_shift:]/vars_dict["FZtop"].data.values[:-ind_shift])
            axes[3].plot(vars_dict["FZtop"].data.time.values[:-ind_shift],ZFR_shifted,lw=1,c="k")
        axes[3].set_ylabel("ZFR (shifted)")
        axes[3].set_ylim([-1.5,1.5])

        #t_intArray = [60]
        t_int = 60 #moving window width [min]
        save_spec+= "t_int" + str(t_int)
        best_corrAll = np.nan*np.zeros([1,FZtop.shape[0],2])

        counter=0
        #for i_t_int,t_int in enumerate(t_intArray):   #moving window [min] for which correlation is checked
        t_ind_range = int(t_int*60/Delta_t) #number of indices which correspond to t_int #*60 from min->sec #/Delta_t to consider timestep

        moving=True
        if moving:
            ax=axes[5]
            blue1 = "b"
            blue2 = "skyblue"
            for i_tranges,(t_start,t_end) in enumerate(zip(np.arange(t_ind_range,FZtop.shape[0],t_ind_range),np.arange(2*t_ind_range,FZtop.shape[0],t_ind_range))):
                if i_tranges%2==0:  
                    blue=blue1
                else:
                    blue=blue2
                if all(np.isnan(FZtop[t_start:t_end])): #skip calculations if all is Nan anyway
                    continue
                t_startHHMM = pd.to_datetime(vars_dict["FZtop"].data.time[t_start].values).strftime('%H:%M:%S')
                t_endHHMM = pd.to_datetime(vars_dict["FZtop"].data.time[t_end].values).strftime('%H:%M:%S')

                #first calculate correlation of non-shifted timeseries "as a reference"
                corr_non_shifted = FZtop[t_start:t_end].corr(FZbot[t_start:t_end])
                best_corrNow = [0,corr_non_shifted] #[position,corr-coeff]

                for lag in range(-50,50,1): #this goes ... minutes back-/forward
    
                    t_startLagHHMM = pd.to_datetime(vars_dict["FZtop"].data.time[t_start].values) + pd.Timedelta(seconds=Delta_t*lag)
                    t_endLagHHMM   = pd.to_datetime(vars_dict["FZtop"].data.time[t_end].values) + pd.Timedelta(seconds=Delta_t*lag)
                    
                    FZtopAv = FZtop[t_start:t_end].rolling(av_periods, min_periods=1).mean()
                    FZbotAv = FZbot[t_start:t_end].rolling(av_periods, min_periods=1).mean()
                    best_corr = crosscorrAllDay(FZtopAv,FZbotAv,Delta_t)
                    best_corrAll[0,int((t_start+t_end)/2)] = crosscorrAllDay(FZtopAv,FZbotAv,Delta_t)
                    #best_corrAll[0,int((t_start+t_end)/2)] = crosscorrAllDay(FZtop[t_start:t_end],FZbot[t_start:t_end],Delta_t)

                    #corr_shifted = crosscorr(FZtop[t_start:t_end],FZbot[t_start:t_end],lag=lag)
                    ##print("lag [s]",lag*Delta_t,"corr",corr_shifted)
                    #if corr_shifted>best_corrNow[1]:
                    #    best_corrAll[0,int((t_start+t_end)/2)] = [lag,corr_shifted]
                    #    counter+=1
                print("best correlation between", t_startHHMM,t_endHHMM , best_corrAll[0,int((t_start+t_end)/2),0]*Delta_t, "s :", best_corrAll[0,int((t_start+t_end)/2),1])


                for i_var,key in enumerate(["FZtop","FZbotShifted"]):
                    #shift FZbottom in time
                    if key=="FZbotShifted":
                        vars_dict["FZbotShifted"] = vars_dict["FZbot"]
                        vars_dict["FZbotShifted"].data["time"]= vars_dict["FZbotShifted"].data.time + pd.Timedelta(seconds=-best_corr[0]*Delta_t)
                    var = vars_dict[key]
                    var.data = var.data.rolling(time=av_periods, min_periods=1).mean()

                    #plot timeseries
                    ax.semilogy(var.data.time.values[t_start:t_end],var.data.values[t_start:t_end],lw=1,color=["r",blue][i_var])
                #ax.semilogy(var.data.time.values,var.data.values,lw=1,color=["r",blue][i_var])
                ax.set_ylabel("F$_{Z}$ [mm$^{6}$ m$^{-3}$ m s$^{-1}$]")
                #apply limits
                ax.set_ylim([1e-1,1e4])
            ax.legend()

            #plot best shift and it's correlation coefficient
            ZFR_orig = np.log10(vars_dict["FZbot"].data.values/vars_dict["FZtop"].data.values)
            #axes[4].plot(vars_dict["FZbot"].data.time.values,best_corrAll[0,:,0]*Delta_t,lw=1,label="no av.",c="k")
            axes[4].scatter(vars_dict["FZtop"].data.time.values,best_corrAll[0,:,0]*Delta_t/60.,marker="x",c="k")
            axes[4].set_ylabel("Best Lag [min]")
            print("time",vars_dict["FZtop"].data.time.values[~np.isnan(best_corrAll[0,:,0])],"lag",best_corrAll[0,:,0][~np.isnan(best_corrAll[0,:,0])]*Delta_t/60.)

            ##plot corr-coeff for lag with highest coeff
            #ZFR_orig = np.log10(vars_dict["FZbot"].data.values/vars_dict["FZtop"].data.values)
            ##axes[5].plot(vars_dict["FZbot"].data.time.values,best_corrAll[0,:,1],lw=1,label="no av.",c="k")
            #axes[5].plot(vars_dict["FZbot"].data.time.values,best_corrAll[0,:,1],marker="x",c="k")
            #axes[5].set_ylabel("Best corr. coeff.")

            ##ZFR when considering shifted F_Z
            ##lag_ind = np.where(~np.isnan(best_corrAll[0,:,0]),best_corrAll[0,:,0],0).astype(int).tolist() #convert the best_corrAll to a list of indices
            #lag_ind = best_corrAll[0,:,0].astype(int).tolist() #convert the best_corrAll to a list of indices
            #lag_ind = lag_ind + np.arange(0,len(best_corrAll[0,:,0])) #sum of lag and original index gives the new index
            ##vars_dict["FZbot"].data.values[best_corrAll[0,:,1]]
            #ZFR_new = np.log10(vars_dict["FZbot"].data.values[lag_ind]/vars_dict["FZtop"].data.values)
            #axes[6].plot(vars_dict["FZbot"].data.time.values,ZFR_new,lw=1,label="no av.",c="k")
            axes[6].set_ylabel("ZFR (shifted)")
            axes[6].set_ylim([-1.5,1.5])

    for ax in axes:
        # Major ticks every 6 months.
        fmt = mdates.HourLocator(interval=1)
        ax.xaxis.set_major_locator(fmt)

        # Minor ticks every month.
        fmt = mdates.MinuteLocator([10,20,30,40,50])
        #fmt = mdates.MinuteLocator(interval=1)
        ax.xaxis.set_minor_locator(fmt)

        ax.grid(b=True,which="major")
        ax.grid(b=True,which="minor",linewidth=0.2,linestyle="--")

        #activate axis labels for all plots
        myFmt = mdates.DateFormatter('%H')
        ax.xaxis.set_major_formatter(myFmt)
        ax.xaxis.set_tick_params(which='major', labelbottom=True) 

        #myFmt_minor = mdates.DateFormatter('%M')
        #ax.xaxis.set_minor_formatter(myFmt_minor)
        #ax.xaxis.set_tick_params(which='minor', labelbottom=True,labelsize=4) #activate axis labels for all plots

    ax.set_xlabel("time [h]")
    #plt.legend()

    plt.tight_layout()
    #save figure
    savestr = 'plots/days/timeseries_' + save_spec 

    plt.savefig(savestr + '.png')
    plt.savefig(savestr + '.pdf')
    #print("png is at: ",savestr + '.png')
    print("pdf is at: ",savestr + '.pdf')
    plt.clf()

def Histograms(results):
    '''
    plot histograms and cumulative distribution functions (CDF) of several variables
    '''


    from matplotlib.ticker import MultipleLocator
    import seaborn as sns
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

    FZthresh = 30

    keys    = ["FZtop","ZeXt","RRfromZR"]
    xlabels = ["F$_{Z,top}$ [mm$^{-6}$ m$^{-3}$ m s$^{-1}$]","Ze_${X,top}$ [dBz]","RR [mm h$^{-1}$]"]
    for key,xlabel in zip(keys,xlabels):
        if key=="FZtop":
            xlogscale=False #logscale somehow doesnt work
            ylogscale=False
            xlims=[0,500]; Nbins = 36
            results_now = results
            results_now = results_now.where((results_now["MDVxT"] * Bd(results_now["ZeXt"]))  > FZthresh)
            results_now = results_now.where((results_now["MDVxB"] * Bd(results_now["ZeXb"])*0.23)  > FZthresh)
        elif key=="ZeXt":
            xlogscale=False
            ylogscale=False
            xlims=[-40,45]; Nbins = 17
            results_now = results
        elif key=="RRfromZR":
            xlogscale=False
            ylogscale=False
            xlims=[0,5]; Nbins = 36
            results_now = results
            results_now = results_now.where((results_now["MDVxT"] * Bd(results_now["ZeXt"])) > FZthresh)
            results_now = results_now.where((results_now["MDVxB"] * Bd(results_now["ZeXb"])*0.23)  > FZthresh)

        fig,ax = plt.subplots()
        ax = sns.histplot(results_now[key],log_scale=xlogscale,stat="probability",bins=np.linspace(xlims[0],xlims[1],Nbins))
        axtwinx = ax.twinx()
        axtwinx = sns.ecdfplot(results_now[key],log_scale=[xlogscale,False],color="k")
        print("median",np.nanmedian(results_now[key]),"90thpercentile",np.nanpercentile(results_now[key],90))

        if ylogscale:
            ax.set_yscale("log")
        axtwinx.set_yscale("linear")

        ax.set_xlim(xlims)

        ax.set_xlabel(xlabel)

        plt.tight_layout()
        #save figure
        savestr = 'plots/hists/hist' + key
        plt.savefig(savestr + '.png')
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
    resultsAll = get_observed_melting_properties(onlydate=onlydate,av_min=av_min,calc_or_load=calc_or_load,no_mie_notch=no_mie_notch) #calc_or_load: 1: calculate each day 0: load each day

    #1. do some quality filters (rimed/unrimed, small fluxes)
    filterZeflux  = 30 #the first box in the boxplot will be still unfiltered!
    save_spec = get_save_string(onlydate)

    
    #for filterZeflux in [10,30,100]:
    for filterZeflux in [30]:
        #Boxplot(resultsAll,av_min=av_min,day=onlydate,filterZeflux=filterZeflux)
        Boxplot(resultsAll,av_min=av_min,day=onlydate,filterZeflux=filterZeflux,addWsubsetWithoutCorr=True)

    #shuffle order to make scatter plot more objective (otherways last days are most visible)
    if onlydate!="":
        pass #plot_timeseries(results,save_spec) #illustrate averaging by plotting timeseries of one day

    #####################################
    #filter small fluxes
    results = resultsAll
    results = results.where((results["MDVxT"] * Bd(results["ZeXt"]))  >filterZeflux)
    results = results.where((results["MDVxB"] * Bd(results["ZeXb"])*0.23)  >filterZeflux)
    save_spec = get_save_string(onlydate,filterZeflux=filterZeflux)
    #plot with different filters
    #Profiles(results,save_spec,av_min=av_min,col=calc_or_load_profiles,onlydate=onlydate,no_mie_notch=no_mie_notch)
    #Profiles(results,save_spec,av_min=av_min,col=calc_or_load_profiles,onlydate=onlydate,no_mie_notch=no_mie_notch,correct_w_before_categorization=True)
    ######################################

    #######################################
    ##filter small fluxes and range of fluxes
    #for filterZefluxLow2,filterZefluxHigh2 in zip([0,1,2,5,10,20,100,100,100],[100,100,100,100,100,100,200,500,10000]):
    ##for filterZefluxLow2,filterZefluxHigh2 in zip([100],[10000]):
    #    save_spec = get_save_string(onlydate)
    #    #plot with different filters
    #    ProfilesLowHigFluxes(resultsAll,save_spec+ "filterflux" + str(filterZeflux) +  "lowFluxes" + str(filterZefluxLow2) + "_" + str(filterZefluxHigh2),av_min=av_min,col=calc_or_load_profiles,onlydate=onlydate,no_mie_notch=no_mie_notch,filterZefluxLow=filterZeflux,filterZefluxLow2=filterZefluxLow2,filterZefluxHig2=filterZefluxHigh2)
    ########################################

    #Histograms(resultsAll)
