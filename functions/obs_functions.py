
# coding: utf-8

# Filtering functions

# In[1]:

import IPython.display as dsp
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from IPython.core.debugger import set_trace
from scipy.signal import argrelextrema

import os
import sys

###resampling###    
def calcRadarDeltaGrid(refGrid, radarGrid):
    
    radGrid2d = np.ones((len(refGrid),
                         len(radarGrid)))*radarGrid
    
    deltaGrid = radGrid2d - np.reshape(refGrid,(len(refGrid),1))
    
    return deltaGrid
    
    
def getNearestIndexM2(deltaGrid, tolerance):

    gridIndex = np.nanargmin(abs(deltaGrid), axis=1)
    deltaGridMin = np.nanmin(abs(deltaGrid), axis=1)
    gridIndex = np.array(gridIndex, np.float)
    gridIndex[deltaGridMin>tolerance] = np.nan
    
    return gridIndex
    
    

def getResampledVar(var, xrDataset, timeIndexArray, rangeIndexArray):

    resampledArr = np.ones((timeIndexArray.shape[0],rangeIndexArray.shape[0]))*np.nan
    resampledTimeArr = np.ones((timeIndexArray.shape[0], xrDataset.range.values.shape[0]))*np.nan

    for t, timeIndex in enumerate(timeIndexArray):

        try:
            resampledTimeArr[t]=  xrDataset[var].values[int(timeIndex)]

        except:
            pass


    resampledTimeArrTra = resampledTimeArr.T
    resampledArr = resampledArr.T

    for r, rangeIndex in enumerate(rangeIndexArray):

            try:
                resampledArr[r] = resampledTimeArrTra[int(rangeIndex)]
            except:
                pass

    return resampledArr.T
    


def getResampledVarTime(var, xrDataset, timeIndexArray):

    resampledTimeArr = np.ones_like(timeIndexArray)*np.nan

    for t, timeIndex in enumerate(timeIndexArray):

        try:
            resampledTimeArr[t]=  xrDataset[var].values[int(timeIndex)]

        except:
            pass
        
    return resampledTimeArr    
###resmapling###


def smoothData(variable, minutes):

    aveWind = minutes
    variable = variable.rolling(time=15*aveWind, 
                                min_periods=34, 
                                center=True).mean()
    
    return variable


def smoothDataRange(variable):

    variable = variable.rolling(range=3, 
                                min_periods=2, 
                                center=True).mean()
    
    return variable

# In[3]:

def maskData(variable, flag, mask):
    
    variable = variable.where((flag.values & mask) != mask)
    
    return variable


# In[4]:

def varTimeData(variable):
    
    aveWind = 15*2
    variance =  variable.rolling(time=aveWind, 
                                 min_periods=15,
                                 center=True).var()
    
    return variance


# In[5]:

def getRsquare(dwr_xka_flat, dwr_kaw_flat, funcEq, *popt):

    res = dwr_xka_flat - funcEq(dwr_kaw_flat, *popt)
    sumRes = np.sum(res**2)
    sumTot = np.sum((dwr_kaw_flat - np.mean(dwr_kaw_flat))**2)
    rSquare = 1 - (sumRes/sumTot)
    rSquare = np.round(rSquare, 2)
    
    return rSquare


# In[5]:

def getMLProperties(ldrDtAr, tempDtAr):

#     ldrDtAr.time.values = tempDtAr.time.values
#     ldrDtAr.range.values = tempDtAr.range.values
    
    range2DArr = np.ones_like(ldrDtAr.values)
    range2DArr = range2DArr*ldrDtAr.range.values
    
    rangeData = xr.DataArray(range2DArr,
                         coords={'time':ldrDtAr.time,
                                 'range':ldrDtAr.range},
                         dims=('time','range'))

    ldrGrad = ldrDtAr.rolling(time=4, min_periods=3,center=True).mean()
    #ldrGrad = ldrGrad.rolling(range=5, min_periods=2, 
    #                          center=True).mean()
    
    ldrGrad = ldrGrad.differentiate('range')
    
    ldrGrad = ldrGrad.rolling(range=3, min_periods=2,center=True).mean()
    ldrGrad2 = ldrGrad.differentiate('range')
    
    
    ldrMax = ldrDtAr.copy()
    ldrMax = ldrMax.where(ldrDtAr>-28)
    ldrMax = ldrMax.max(dim='range')
    
    rangeLdrMax = rangeData.copy()
    rangeLdrMax = rangeLdrMax.where(ldrDtAr==ldrMax)

    #ldrPeakHeight
    heightLDRmax = rangeLdrMax.copy()
    heightLDRmax = heightLDRmax.where(tempDtAr>=-1)
    heightLDRmax = heightLDRmax.where(ldrDtAr>-28)
    heightLDRmax = heightLDRmax.mean(dim='range')
    
    #ML top height
    upMaxLDR = ldrGrad2.copy()
    upMaxLDR = upMaxLDR.where(rangeData > heightLDRmax)
    upMaxLDR = upMaxLDR.where(ldrDtAr>-28)
    upMaxLDR = upMaxLDR.where(tempDtAr >= -1)
    upMaxLDR = upMaxLDR.max(dim='range')
    
    heightTopMaxLDR = rangeData.copy()
    heightTopMaxLDR = heightTopMaxLDR.where(ldrGrad2 == upMaxLDR)
    heightTopMaxLDR = heightTopMaxLDR.where(tempDtAr >= -1)
    heightTopMaxLDR = heightTopMaxLDR.mean(dim='range') #+ 300 #(height offset test trying to avoid as high as 40 dBZ)
    
    #ML botton height
    bottonMaxLDR = ldrGrad2.copy()
    bottonMaxLDR = bottonMaxLDR.where(rangeData < heightLDRmax)
    bottonMaxLDR = bottonMaxLDR.where(ldrDtAr>-28)
    bottonMaxLDR = bottonMaxLDR.where(tempDtAr >= -1)
    bottonMaxLDR = bottonMaxLDR.max(dim='range')
    
    heightBottonMaxLDR = rangeData.copy()
    heightBottonMaxLDR = heightBottonMaxLDR.where(ldrGrad2 == bottonMaxLDR)
    heightBottonMaxLDR = heightBottonMaxLDR.where(tempDtAr >= -1)
    heightBottonMaxLDR = heightBottonMaxLDR.mean(dim='range')
    
    return heightTopMaxLDR, heightLDRmax, heightBottonMaxLDR, ldrGrad, ldrGrad2

def getMLProperties_singleprofile(ldrDtAr, tempDtAr):

#     ldrDtAr.time.values = tempDtAr.time.values
#     ldrDtAr.range.values = tempDtAr.range.values
    
    #range2DArr = np.ones_like(ldrDtAr.values)
    #range2DArr = range2DArr*ldrDtAr.range.values
    #
    #rangeData = xr.DataArray(range2DArr,
    #                     coords={'time':ldrDtAr.time,
    #                             'range':ldrDtAr.range},
    #                     dims=('time','range'))

    #ldrGrad = ldrDtAr.rolling(time=4, min_periods=3,center=True).mean()
    #ldrGrad = ldrGrad.rolling(range=5, min_periods=2, 
    #                          center=True).mean()
    rangeData = ldrDtAr.range    
    ldrGrad = ldrDtAr.differentiate('range')
    
    ldrGrad = ldrGrad.rolling(range=3, min_periods=2,center=True).mean()
    ldrGrad2 = ldrGrad.differentiate('range')
    
    
    ldrMax = ldrDtAr.copy()
    ldrMax = ldrMax.where(ldrDtAr>-28)
    ldrMax = ldrMax.max(dim='range')
    
    rangeLdrMax = rangeData.copy()
    rangeLdrMax = rangeLdrMax.where(ldrDtAr==ldrMax)

    #ldrPeakHeight
    heightLDRmax = rangeLdrMax.copy()
    heightLDRmax = heightLDRmax.where(tempDtAr>=-1)
    heightLDRmax = heightLDRmax.where(ldrDtAr>-28)
    heightLDRmax = heightLDRmax.mean(dim='range')
    
    #ML top height
    upMaxLDR = ldrGrad2.copy()
    upMaxLDR = upMaxLDR.where(rangeData > heightLDRmax)
    upMaxLDR = upMaxLDR.where(ldrDtAr>-28)
    upMaxLDR = upMaxLDR.where(tempDtAr >= -1)
    upMaxLDR = upMaxLDR.max(dim='range')
    
    heightTopMaxLDR = rangeData.copy()
    heightTopMaxLDR = heightTopMaxLDR.where(ldrGrad2 == upMaxLDR)
    heightTopMaxLDR = heightTopMaxLDR.where(tempDtAr >= -1)
    heightTopMaxLDR = heightTopMaxLDR.mean(dim='range') #+ 300 #(height offset test trying to avoid as high as 40 dBZ)
    
    #ML botton height
    bottonMaxLDR = ldrGrad2.copy()
    bottonMaxLDR = bottonMaxLDR.where(rangeData < heightLDRmax)
    bottonMaxLDR = bottonMaxLDR.where(ldrDtAr>-28)
    bottonMaxLDR = bottonMaxLDR.where(tempDtAr >= -1)
    bottonMaxLDR = bottonMaxLDR.max(dim='range')
    
    heightBottonMaxLDR = rangeData.copy()
    heightBottonMaxLDR = heightBottonMaxLDR.where(ldrGrad2 == bottonMaxLDR)
    heightBottonMaxLDR = heightBottonMaxLDR.where(tempDtAr >= -1)
    heightBottonMaxLDR = heightBottonMaxLDR.mean(dim='range')
    
    return heightTopMaxLDR, heightLDRmax, heightBottonMaxLDR, ldrGrad, ldrGrad2


# In[2]:

def getMaxDWR(dwrDtAr, tempDtAr):

    
    range2DArr = np.ones_like(dwrDtAr.values)
    range2DArr = range2DArr*dwrDtAr.range.values
    
    rangeData = xr.DataArray(range2DArr,
                         coords={'time':dwrDtAr.time,
                                 'range':dwrDtAr.range},
                         dims=('time','range'))
    
    dwrGrad = dwrDtAr.rolling(time=4, min_periods=2, 
                              center=True).mean()
    dwrGrad = dwrGrad.differentiate('range')
    
    dwrGrad = dwrGrad.rolling(range=3, min_periods=2,
                              center=True).mean()
    dwrGrad2 = dwrGrad.differentiate('range')

    #dwrPeakHeight
    #dwrDtAr = dwrDtAr.where(tempDtAr > -3)#.where(rangeData < 4000)#.where(tempDtAr > -3).where(dwrDtAr> 5)
    dwrDtAr = dwrDtAr.where(dwrDtAr > 0)#low limit to dwr xka
    dwrDtAr = dwrDtAr.where(tempDtAr > -5)
    dwrMax = dwrDtAr.max(dim='range')
    heightDWRmax = rangeData.where(dwrDtAr==dwrMax)
    heightDWRmax = heightDWRmax.mean(dim='range')
    
    #DWR max slope height
    dwrMaxSlope = dwrGrad.copy()
    dwrMaxSlope = dwrMaxSlope.where(dwrDtAr > 0)#low limit to dwr xka
    dwrMaxSlope = dwrMaxSlope.where(tempDtAr > -5)
    dwrMaxSlope = dwrMaxSlope.min(dim='range')
    
    #temp dwrMax
    dwrMaxTemp = tempDtAr.copy()
    dwrMaxTemp = dwrMaxTemp.where(rangeData == heightDWRmax)
    dwrMaxTemp = dwrMaxTemp.mean(dim='range')
    
    
    return heightDWRmax, dwrMax, dwrGrad, dwrGrad2, dwrMaxTemp 

def get_mie_notch_DV_theor(pres):
    #get expected Mie-notch Doppler velocity with w_vertical=0

    ##apply density correction to velocity
    p0 = 1.01325e5
    return 5.9*(p0/pres)**0.54 #calculate notch terminal velocity based on air density

def get_mie_notch_DV(W_spec,pres,timestr=""):
    '''
    get the Doppler velocity of the mie-notch from the W-Band spectra of liquid clouds to correct for the vertical wind speed
    '''
    #limit range where mie-notch is expected
    W_spec = W_spec.sel(dopplerW=slice(-7.3,-4.9)) 
    W_spec = W_spec.dropna("dopplerW")
    if W_spec.shape[0]<5:
        return np.nan,np.nan
    W_spec_av = W_spec.rolling(dopplerW=6,center=True).mean()
    W_spec_av  = W_spec_av.dropna("dopplerW")
    if W_spec_av.shape[0]<5:
        return np.nan,np.nan

    #mie_notch_vel = W_spec_av.idxmin().values #search for global minima

    i_min = argrelextrema(W_spec_av.values,np.less) #search for local minima 
    if  i_min[0].shape[0]!=1: #there are several local minima detected
        return np.nan,np.nan

    mie_notch_vel = W_spec_av.dopplerW.values[i_min][0]
    
    #plt.plot(-W_spec.dopplerW,W_spec,linestyle="-",lw=1,label="orig.")
    #plt.plot(-W_spec_av.dopplerW,W_spec_av,linestyle="--",lw=1,label="av")
    #plt.text(-mie_notch_vel,W_spec_av.sel(dopplerW=mie_notch_vel).values,"x")
    #plt.legend()
    #plt.savefig("plots/mie-notch-example" + timestr +  ".png")
    #plt.savefig("plots/mie-notch-example.pdf")
    #plt.cla()

    theo_notch_terminal_vel = get_mie_notch_DV_theor(pres)
    if W_spec_av.dopplerW.shape[0]<10: #too small array
        return np.nan,np.nan
    if mie_notch_vel<W_spec_av.dopplerW[2] or mie_notch_vel>W_spec_av.dopplerW[-2] or np.isnan(mie_notch_vel) or W_spec_av.sel(dopplerW=mie_notch_vel).values<-40: #minimum is at largest/smallest velocities or no data in velocity interval -> no Mie-notch detectable
        return np.nan,np.nan
    else:
        w_est = -mie_notch_vel - theo_notch_terminal_vel #5.9m/s comes from Tridon and Battaglia (2015)
    #sys.exit()
    return w_est,-mie_notch_vel
    

def getProp(velDtAr, hghtBotLDR, hghtTopLDR):
    
    range2DArr = np.ones_like(velDtAr.values)
    range2DArr = range2DArr*velDtAr.range.values

    rangeData = xr.DataArray(range2DArr,
                         coords={'time':velDtAr.time,
                                 'range':velDtAr.range},
                         dims=('time','range'))
    
    velGrad = velDtAr.copy()
    velGrad = velGrad.rolling(time=4, min_periods=2, 
                              center=True).mean()
    velGrad = velGrad.differentiate('range')
    
    velGrad = velGrad.rolling(range=3, min_periods=2,
                              center=True).mean()
    velGrad2 = velGrad.differentiate('range')
    velGrad3 = velGrad2.differentiate('range')
    
    velBotLDR = velDtAr.copy()
    velBotLDR = velBotLDR.where(rangeData == hghtBotLDR)
    velBotLDR = velBotLDR.mean(dim='range')
    
    velTopLDR = velDtAr.copy()
    velTopLDR = velTopLDR.where(rangeData == hghtTopLDR)
    velTopLDR = velTopLDR.mean(dim='range')

    
    return velGrad, velGrad2, velGrad3, velBotLDR, velTopLDR

def getProfile(var, hght):
    
    varOut = hght.copy(data=np.ones_like(hght)*np.nan) #initialize empty dataframe

    for i_h,h_rel in enumerate(hght.range):
        varOut[0:var.shape[0],i_h] = var.sel(range=hght[0:var.shape[0],i_h],time=hght.time[0:var.shape[0]],method="nearest").values.copy()
    
    return varOut 
    
def getVarAtTemp(var, temp_field, temp):
    '''
    get values of "var" at "temp" [in degC]
    '''
 
    range2DArr = np.ones_like(var.values)
    range2DArr = range2DArr*var.range.values

    rangeData = xr.DataArray(range2DArr,
                         coords={'time':var.time,
                                 'range':var.range},
                         dims=('time','range'))
    
    #get value within 2 degree interval
    var_at_temp = var.copy()
    var_at_temp = var_at_temp.where(temp_field>=temp-1)
    var_at_temp = var_at_temp.where(temp_field<=temp+1)
    var_at_temp = var_at_temp.mean(dim='range')
    
    return var_at_temp

def getVarAtTempModel(var, temp_field, temp):
    '''
    get values of "var" at "temp" [in degC]
    '''

    range2DArr = np.ones_like(temp_field)
    range2DArr = np.tile(temp_field.height_2,(range2DArr.shape[0],1))

    rangeData = xr.DataArray(range2DArr,
                         coords={'time':temp_field.time,
                                 'height_2':temp_field[:,::-1].height_2},
                         dims=('time','height_2'))

    var_at_temp = xr.DataArray(var,
                         coords={'time':temp_field.time,
                                 'height_2':temp_field[:,::-1].height_2},
                         dims=('time','height_2'))
    
    #get value within 2 degree interval
    #var_at_temp = var.copy()
    var_at_temp = var_at_temp.where(temp_field[::-1]>=temp-1)
    var_at_temp = var_at_temp.where(temp_field[::-1]<=temp+1)
    var_at_temp = var_at_temp.mean(dim='height_2')
    
    return var_at_temp

def plot2hist(dataToPlot):
    
    
    aData = dataToPlot['a']['data']
    bData = dataToPlot['b']['data']
    
    aData[np.isnan(bData)] = np.nan
    bData[np.isnan(aData)] = np.nan

    aData = aData[~ np.isnan(aData)]
    bData = bData[~ np.isnan(bData)]
    
    plt.figure(figsize=(10,8))
    plt.hist2d(aData, bData, bins=30,
               cmin=1, norm=LogNorm(),
               cmap='jet')
    
    plt.xlabel(dataToPlot['a']['label'])
    plt.ylabel(dataToPlot['b']['label'])
    plt.grid()
    plt.colorbar()
    plt.show()


# In[ ]:

#plt.figure(figsize=(16,10))

def getCloudTopTemp(zeDtAr, temp):
    
    range2DArr = np.ones_like(zeDtAr.values)
    range2DArr = range2DArr*zeDtAr.range.values
    
    rangeData = xr.DataArray(range2DArr,
                         coords={'time':zeDtAr.time,
                                 'range':zeDtAr.range},
                         dims=('time','range'))

    rangeData = rangeData.where(~np.isnan(zeDtAr))

    rangeDataMax = rangeData.max(dim='range')
    tempCloudTop = temp.where(rangeData==rangeDataMax)
    tempCloudTop = tempCloudTop.mean(dim='range')
    
    return tempCloudTop


def getStudyReg(var, temp, 
                tempThreshold, 
                heightDWRMax):

    range2DArr = np.ones_like(var.values)
    range2DArr = range2DArr*var.range.values
    
    rangeData = xr.DataArray(range2DArr,
                         coords={'time':var.time,
                                 'range':var.range},
                         dims=('time','range'))

    var = var.where(temp > tempThreshold)
    var = var.where(rangeData > heightDWRMax)
    
    return var


def calcRHI(data):
    
    T=data['ta'].copy()+273.15
    rh = data['hur']
    
    # saturation vapor pressure over ice
    A=-2663.5
    B=12.537
    P=A/T + B
    es_i = 10**(P)
    
    # saturation vapor pressure over water
    e1=1013.250
    TK=273.15
    es = 100*e1*10**(10.79586*(1-TK/(T))-5.02808*np.log10((T)/TK)+
                1.50474*1e-4*(1-10**(-8.29692*((T)/TK-1)))+
                0.42873*1e-3*(10**(4.76955*(1-TK/(T)))-1)-2.2195983)    

    e = rh*es/100.

    # relative humidity with respect to ice
    rhiOri = (100*e / es_i)
    #rhi.T.plot()
    #plt.show()

    return rhiOri


def calcRHSat_ice(data):
    
    T=data['ta'].copy()+273.15
    rh = data['hur']
    
    # saturation vapor pressure over ice
    A=-2663.5
    B=12.537
    P=A/T + B
    es_i = 10**(P)
    
    # saturation vapor pressure over water
    e1=1013.250
    TK=273.15
    es = 100*e1*10**(10.79586*(1-TK/(T))-5.02808*np.log10((T)/TK)+
                1.50474*1e-4*(1-10**(-8.29692*((T)/TK-1)))+
                0.42873*1e-3*(10**(4.76955*(1-TK/(T)))-1)-2.2195983)    
    
    #     
    rhSat_ice = (100 * es_i / es)  

    return rhSat_ice


#Old version
def calcRHIOld(data): 
    
    # saturation vapor pressure with respect to ice
    est = 611.20 #Pa
    Tt = 273.16 #K
    Rv = 461.5 #JKg-1 K-1
    Ls = 2.834*10**6 #JKg-1
    T = data['Temp_Cl'].copy()+273.16
    es_i = est*np.exp((Ls/Rv)*(1/Tt-1/T))

    # vapor pressure 
    temp = data['Temp_Cl'].copy()+273.16
    press = data['Pres_Cl']
    rh = data['RelHum_Cl']

    es = 611*np.exp(17.27*(temp - 273.16)/(237.3 + (temp - 273.16)))
    ws = 0.622*(es/(press-es))

    e = rh*es/100.

    # relative humidity with respect to ice
    rhiOri = (100*e / es_i)
    #rhi.T.plot()
    #plt.show()

    return rhiOri


#Old version
def calcRHSat_iceOld(data):
    
    # saturation vapor pressure over ice
    est = 611.20 #Pa
    Tt = 273.16 #K
    Rv = 461.5 #JKg-1 K-1
    Ls = 2.834*10**6 #JKg-1
    T = data['Temp_Cl'].copy()+273.16
    es_i = est*np.exp((Ls/Rv)*(1/Tt-1/T))

    # saturation vapor pressure over water
    temp = data['Temp_Cl'].copy()+273.16
    es = 611*np.exp(17.27*(temp - 273.16)/(237.3 + (temp - 273.16)))
    
    #     
    rhSat_ice = (100 * es_i / es)
    

    return rhSat_ice
