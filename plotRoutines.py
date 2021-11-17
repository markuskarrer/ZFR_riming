'''
This is the beginning of plotting routines for McSnow output
'''
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import numpy as np
import pickle
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import matplotlib.colors as colors
from functions import obs_functions as fc #jupyter notebook code
from IPython.core.debugger import set_trace
import sys
import turbo_colormap_mpl
turbo_colormap_mpl.register_turbo()

def Bd(x): #conversion: logarithmic [dB] to linear [mm*6/m**3]
    return 10.0**(0.1*x)

    
def plotMomentsObs4paper(LDRall,dataLV2,outPath,outName,average_min="2",date_str=""):

    ###special colormap for MDV
    # sample the colormaps that you want to use. Use 128 from each so we get 256
    # colors in total
    import matplotlib.colors as mcolors
    import matplotlib.dates as mdates
    import matplotlib.collections as collections
    colors1 = plt.cm.get_cmap("turbo")(np.linspace(0., 1, 128*10)) 
    colors2 = plt.cm.gray_r(np.linspace(0.02, 1, 128*10))

    # combine them and build a new colormap
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('combined_cmap', colors)

    import matplotlib.pyplot as mpl
    mpl.style.use('seaborn')
    mpl.rcParams['font.size'] = 35
    mpl.rcParams['legend.fontsize'] = 20
    mpl.rcParams['figure.titlesize'] = 20

    mpl.rcParams['legend.fancybox'] = True
    mpl.rcParams['legend.framealpha'] = 0.7
    mpl.rcParams['legend.facecolor']='silver'
    mpl.rcParams['legend.frameon']=True

    mpl.rcParams['ytick.labelsize']= 20
    mpl.rcParams['xtick.labelsize']= 20
    mpl.rcParams['axes.titlesize']=20
    mpl.rcParams['axes.labelsize']=20

    mpl.rcParams['lines.linewidth']=5

    fig,axes = plt.subplots(nrows=6,figsize=(15,20),sharex=True)
    if date_str!="":
        with open('data/MLprop' + date_str  + '.pkl', 'rb') as f:
            MLprop = pickle.load(f)
    ZeX = dataLV2.X_DBZ_H
    ZeK = dataLV2.Ka_DBZ_H
    ZeW = dataLV2.W_DBZ_H
    DWRxk = ZeX-ZeK
    DWRkw = ZeK-ZeW
    MDVx = -dataLV2.X_VEL_H
    #MDVk = -dataLV2.Ka_VEL_H
    #MDVw = -dataLV2.W_VEL_H
    ZfluxX = Bd(ZeX)*MDVx #derive reflectivity flux
    ZfluxX_dh = ZfluxX.differentiate("range").copy() #built vertical gradient
    ZfluxX_dh = ZfluxX_dh.rolling(time=15*int(average_min)).mean().copy() #built vertical gradient

    #calculate ML top and bottom
    MLtop, heightLDRmax, MLbottom,        ldrGrad, ldrGrad2 = fc.getMLProperties(LDRall, dataLV2.ta)

    MLtop+=36
    MLbottom-=36

    #get properties at ML top and bottom
    ZFtop = ZfluxX.sel(range=MLtop,method="nearest").copy()
    ZFbottom = ZfluxX.sel(range=MLbottom,method="nearest").copy()
    ZFR = ((ZFbottom)/(ZFtop)).values*0.23

    MDVxTop     = MDVx.sel(range=MLtop,method="nearest")
    MDVxBot     = MDVx.sel(range=MLbottom,method="nearest")
    DWRxkTop    = DWRxk.sel(range=MLtop,method="nearest")
    
    MDV_max = 8.0
    letters = ["a)","b)","c)","d)","e)","f)"]
    for i_var,(height,var,label,lims,ax) in enumerate(zip(
                                                          [LDRall.range,ZeX.range,MDVx.range,DWRxk.range], #yvar
                                                          [LDRall,ZeX,MDVx,DWRxk], 
                                                          ['LDR$_{Ka}$ [dB]','Ze$_{X}$ [dB]','MDV$_X$ [m/s]','DWR$_{X,Ka}$ [dB]'], #label
                                                          [[-50,-5],[0,35],[0,MDV_max],[-2,15]], #xlims
                                                          axes.flatten())):
        plt.text(-0.15, 0.95,letters[i_var],horizontalalignment='left',verticalalignment='bottom',transform = ax.transAxes,fontsize=30,color="k")
        if label=='MDV$_X$ [m/s]':
            cmap = mymap
            bounds = np.hstack((np.linspace(0,2.5,128), np.linspace(2.5,MDV_max,128)))
            norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=bounds.shape[-1])
            ticks = [0.0,1.0,2.5,5,8]
        else:
            cmap = getNewNipySpectral()
            norm = None
            ticks = None
        plot = ax.pcolormesh(var.time.values,height.values,np.transpose(var.values),vmin=lims[0],vmax=lims[1],cmap=cmap,norm=norm,zorder=-100,rasterized=True) 

        ##cb = plt.colorbar(plot,ax=ax) 
        cb = plt.colorbar(plot, ax=ax,ticks=ticks)
        cb.set_label(label,fontsize=18)
        cb.ax.tick_params(labelsize=16)
        #add ML top and bottom lines
        MLlinesT = ax.plot(MLtop.time.values,MLtop,color='k',linewidth=3,linestyle='--')
        MLlinesB = ax.plot(MLbottom.time.values,MLbottom,color='k',linewidth=3,linestyle='--')
        #crop x-axis because smothing applied in ML detection removes outermost ML height/bottom
        ax.tick_params(axis='y',labelsize=20)
        ax.set_ylabel('range [m]',fontsize=20) 
        ax.set_ylim([MLbottom.dropna("time")[0].values-500,MLtop.dropna("time")[0].values+500])
        #ax.grid(True,which="both",color='k', linestyle='-')

 

    #plot MDV (top and bottom) and DWRxkTop
    ax=axes[-2]
    ax2 = ax.twinx()

    plt.text(-0.15, 0.95,letters[-2],horizontalalignment='left',verticalalignment='bottom',transform = ax.transAxes,fontsize=30,color="k")
    for i_av,av_min in enumerate(["",average_min]):
        for var,label in zip([MDVxTop,DWRxkTop],['MDV$_{X,top','DWR$_{X,Ka']):

            if av_min!="":
                label_av = "," + av_min + "min-av.}$"
                var = var.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int(av_min)*15)).mean().copy()
                #var = var.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int("20")*15)).mean().copy()
            else:   
                label_av = "}$"

            lw = [1,3][i_av]
            ls = ["--","-"][i_av]
            ax.tick_params(axis='x',labelsize=20)
            ax.tick_params(axis='y',labelsize=20)
            if label=='MDV$_{X,top':
                ax.plot(var.time,var,label= label + label_av,c="orange",ls=ls,lw=lw)
                ax.set_ylim([0.75,2.75])
            elif label=='MDV$_{X,bottom':
                ax.plot(var.time,var/4,label="0.25*" + label + label_av,c="green",ls=ls,lw=lw)
                ax.set_ylim([0.75,2.75])
            elif label=='DWR$_{X,Ka':
                ax2.plot(var.time,var,label=label + label_av,c="k",ls=ls,lw=lw)
                ax2.set_ylim([-2.5,17.5])
    
            ax2.yaxis.grid(b=True,which="major",linestyle="--",c="k")
            ax2.yaxis.grid(b=True,which="minor",linestyle="",c="k")

    #categorize ptype to mark region of unrimed, transitional, rimed
    ptype_flag = categorize_part_type_timeseries(MDVxTop,DWRxkTop)
    #add boxes to differentiate time ranges 
    for i in np.arange(len(ptype_flag.values)-1):
        if ~np.isnan(ptype_flag.values[i]):
            ax.plot([var.time.values[i],var.time.values[i+1]],[ax.get_ylim()[0],ax.get_ylim()[0]],c=["blue","green","red"][int(ptype_flag.values[i])],lw=15)


    ax.legend(loc="upper left",ncol=2,bbox_to_anchor=(0.00, 1.15),fontsize=20)
    ax.set_ylabel("MDV [m/s]",fontsize=20)
    ax2.legend(loc="upper right",ncol=1,bbox_to_anchor=(1.05, 1.15),fontsize=20)
    ax2.set_ylabel("DWR [dB]",fontsize=20)
    ax2.tick_params(axis='y',labelsize=20)
    #plot and remove colorbar to have aligned time with plots above
    cb = plt.colorbar(plot, ax=ax)
    cb.remove()
    plt.draw() 

    ax = axes[-1]
    plt.text(-0.15, 0.95,letters[-1],horizontalalignment='left',verticalalignment='bottom',transform = ax.transAxes,fontsize=30,color="k")
    ax2 = ax.twinx()
    for i_av,av_min in enumerate(["",average_min]):
        lw = [1,3][i_av]
        ls = ["--","-"][i_av]
        if av_min!="":
            label_av = av_min + "min-av.}$"
            var = var.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int(av_min)*15)).mean().copy()
        else:   
            label_av = "}$"
        if av_min!="":
            #ZFR = ZFtop/ZFbottom
            #ZFR_dirmean = ZFR.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int(av_min)*15)).mean().copy()
            ZFtop = ZFtop.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int(av_min)*15)).mean().copy()
            ZFbottom = ZFbottom.rolling(min_periods=int(int(av_min)*15/2),center=True,time=int(int(av_min)*15)).mean().copy()
            ZFR = ZFbottom/ZFtop*0.23
            ax.plot(ZeX.time,np.log10(ZFR),c="k",label="ZFR$_{" + label_av,ls=ls,lw=lw)
        else:
            ax.plot(ZeX.time,np.log10(ZFR),c="k",label="ZFR$_{" + label_av,ls=ls,lw=lw)
        ax2.semilogy(ZeX.time,ZFtop,c="orange",label="ZF$_{top," + label_av,ls=ls,lw=lw)
        ax2.semilogy(ZeX.time,ZFbottom*0.23,c="g",label="ZF$_{bottom," + label_av +  "/0.23",ls=ls,lw=lw)
    ax.set_ylim([-1.0,1.0]) #[0.23-0.23,0.23+0.23])
    #mark regions with too low fluxes
    for i in np.arange(len(ptype_flag.values)-1):
        if ZFtop.values[i]<1e2 or ZFbottom.values[i]<(1e2/0.23):
            ax.plot([var.time.values[i],var.time.values[i+1]],[ax.get_ylim()[0],ax.get_ylim()[0]],c="k",lw=15)
    ax.axhline(0.0,c="magenta")
    ax.legend(loc="upper left",ncol=1,bbox_to_anchor=(0.00, 1.27),fontsize=20)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("time",fontsize=20)
    ax.set_ylabel("log(ZFR)",fontsize=20)
    ax.tick_params(axis='x',labelsize=20)
    ax.tick_params(axis='y',labelsize=20)
    ax2.legend(loc="upper right",ncol=2,bbox_to_anchor=(1.05, 1.27),fontsize=20)
    ax2.set_ylabel("ZF [mm$^6$ m/s]",fontsize=20)
    ax2.tick_params(axis='y',labelsize=20)
    ax2.yaxis.grid(b=True,which="major",linestyle="--",c="k")
    #ax2.yaxis.grid(b=True,which="minor",linestyle="",c="k")
    #plot and remove colorbar to have aligned time with plots above
    cb = plt.colorbar(plot, ax=ax)
    cb.remove()
    plt.draw() 

    for i_ax,ax in enumerate(axes.flatten()):
        ax.set_xlim([ZeX.time.values[150],ZeX.time.values[-150]])
        # Major ticks every ... 
        fmt = mdates.MinuteLocator([10,20,30,40,50,0])
        ax.xaxis.set_major_locator(fmt)
        # Minor ticks every ...
        #fmt = mdates.MinuteLocator(interval=1)
        #ax.xaxis.set_minor_locator(fmt)
        ax.xaxis.grid(b=True,which="major",c="k")
        #ax.xaxis.grid(b=True,which="minor",linewidth=0.2,linestyle="--",c="k")
        if i_ax!=4:
            ax.yaxis.grid(b=True,which="major",linestyle="-",c="k")
        else:
            ax.yaxis.grid(b=True,which="major",linestyle="",c="k")
        ax.yaxis.grid(b=True,which="minor",linestyle="",c="k")

    #plt.tight_layout()
    plt.savefig(outPath+outName+'.png',dpi=200,bbox_inches='tight')
    plt.savefig(outPath+outName+'.pdf',dpi=200,bbox_inches='tight')
    print("plot is at: ", outPath+outName+'.png')
    plt.close()
def getNewNipySpectral():

    from matplotlib import cm
    from matplotlib.colors import ListedColormap

    numEnt = 15

    viridis = cm.get_cmap('nipy_spectral', 256)
    newcolors = viridis(np.linspace(0, 1, 256))

    colorSpace = np.linspace(198, 144, numEnt)/256
    colorTest=np.zeros((numEnt,4))
    colorTest[:,3] = 1
    colorTest[:,0]=colorSpace

    newcolors[- numEnt:, :] = colorTest
    newcmp = ListedColormap(newcolors)

    return newcmp

def categorize_part_type_timeseries(MDVxT,DWRxkT):

    '''
    flag each timestep with 0: unrimed; 1: transitional; 2: rimed
    '''

    ptype_flag = xr.DataArray(data=np.nan*np.ones_like(MDVxT.values),dims=["time"],coords=[MDVxT.time])    

    ####classify according to MDV and DWR
    DWR_MDV_PRb1_dict = dict()
    DWR_MDV_PRb1_dict["unrimed-transitional"] = [0.6,7.3] #fit to rime fraction [0,0.2] and 1mm/h<RR<4mm/h # separates unrimed from transitional
    DWR_MDV_PRb1_dict["transitional-rimed"]   = [0.75,2.58] #fit to rime fraction [0,0.2] and 1mm/h<RR<4mm/h # separates transitional from rimed
    ptype_flag.values = np.where(DWRxkT>DWR_MDV_PRb1_dict["unrimed-transitional"][0]*MDVxT**DWR_MDV_PRb1_dict["unrimed-transitional"][1],0,ptype_flag).copy()
    ptype_flag.values = np.where(DWRxkT<DWR_MDV_PRb1_dict["transitional-rimed"][0]*MDVxT**DWR_MDV_PRb1_dict["transitional-rimed"][1],2,ptype_flag).copy()
    ptype_flag.values = np.where(np.logical_and(
        DWRxkT<DWR_MDV_PRb1_dict["unrimed-transitional"][0]*MDVxT**DWR_MDV_PRb1_dict["unrimed-transitional"][1],
        DWRxkT>DWR_MDV_PRb1_dict["transitional-rimed"][0]*MDVxT**DWR_MDV_PRb1_dict["transitional-rimed"][1]),
        1,ptype_flag).copy()

    return ptype_flag
    
def plotProfilesAndSpectraObs(LDRall,dataLV2,dataLV0,Peaks,Edges,outPath,plot_all_times=False):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import postProcessSpectra as post
    
    #    return
    av_timedelta = np.timedelta64(10, 'm')  #average window
    #LDRall = fc.smoothData(LDRall, 5)
    #heightBottomMaxLDR-=36
    #heightTopMaxLDR+=3*36 #144
    for t in dataLV0.time:
        ti = pd.to_datetime(str(t.values)).strftime('%Y%m%d_%H%M%S') #get string to name plot
        t_spectra = t #take spectra from the middle of the average period
        ti_spectra = pd.to_datetime(str(t_spectra.values)).strftime('%Y%m%d_%H%M%S') #get string to name plot
        t_title_spectra = pd.to_datetime(str(t_spectra.values)).strftime('%Y%m%d %H:%M:%S UTC') 

        print("plot time: ", t_title_spectra)
        # select data according to time to plot
        dataLV2selt = dataLV2.sel(time=t_spectra)
        LDR  = LDRall.sel(time=t_spectra)
        #get ML top/bottom
        MLtop_t, heightLDRmax, MLbottom_t,        ldrGrad, ldrGrad2 = fc.getMLProperties_singleprofile(LDR, dataLV2selt.ta)
        MLtop_t+=36 #144
        MLbottom_t-=36
        height_lims = [MLbottom_t-500,MLtop_t+500]
        if np.isnan(MLtop_t) or np.isnan(MLbottom_t):
            continue
        Peaksselt   = Peaks.sel(time=t_spectra)
        Edgesselt   = Edges.sel(time=t_spectra) #slice(t,t+av_timedelta))
        #integral variables
        ZeX = dataLV2selt.X_DBZ_H
        ZeK = dataLV2selt.Ka_DBZ_H
        ZeW = dataLV2selt.W_DBZ_H
        DWRxk = ZeX-ZeK
        DWRkw = ZeK-ZeW
        MDVx = -dataLV2selt.X_VEL_H
        MDVk = -dataLV2selt.Ka_VEL_H
        MDVw = -dataLV2selt.W_VEL_H
        ZfluxX = Bd(ZeX)*MDVx #derive reflectivity flux
        RH = dataLV2selt.hur
        pres = dataLV2selt.pa
        presMLbot = pres.sel(range=MLbottom_t).values
        ZFtop = ZfluxX.sel(range=MLtop_t)
        ZFbottom = ZfluxX.sel(range=MLbottom_t)
        #if ZFtop<0 or ZFbottom<0 or ZeX.sel(range=MLtop_t)<0: #skip plot if ZF is small
        #    continue
        ZFR = (ZFbottom/ZFtop).values*0.23
        ##select values in time step
        temp = dataLV2selt.ta
        #spectral variables
        dataLV0selt = dataLV0.sel(time=t_spectra)
        specW   = 10*np.log10(dataLV0selt.WSpecH) #make logscale
        specW_MLbot   = specW.sel(range=MLbottom_t) #select W-Band minimum to search for Mie-notch and correct vertical velocity at ML bottom
        specKa    = 10*np.log10(dataLV0selt.KaSpecH) #make logscale
        ZeKfromSpec = 10*np.log10(dataLV0selt.KaSpecH.sum(axis=1))
        ZeKVfromSpec = 10*np.log10(dataLV0selt.KaSpecV.sum(axis=1))
        LDRfromSpec = ZeKVfromSpec-ZeKfromSpec
        specX   = 10*np.log10(dataLV0selt.XSpecH) #make logscale
        specKaV   = 10*np.log10(dataLV0selt.KaSpecV) #make logscale
        specKaHnoise   = 10*np.log10(dataLV0selt.KaSpecNoiseH) #make true linear (saved are linear units to which the log-lin transformation was applied once to much)
        specKaHnoise   = 10*np.log10(specKaHnoise)
        specKaHstrongnoisecorr   = specKa.where(specKa>(specKaHnoise+20)).copy()
        #specXKa  = dataLV0selt.XSpecH.interp_like(dataLV0.KaSpecH)-dataLV0.KaSpecH 
        LDRspecKa = specKaV - specKa #LDR normal noise reduction 
        LDRspecKaNoiseCorr = specKaV - specKaHstrongnoisecorr #LDR normal noise reduction 
        LDRspecW  = 10*np.log10(dataLV0selt.WSpecV)-10*np.log10(dataLV0.WSpecH.sel(time=t)) #make logscale

        # now I (Leonie or Jose?) need to regrid along the doppler dimension in order to calculate spectral DWR... since we already did that for w-band, I want to regrid everything to w-band. WindowWidth is the width of moving window mean 
        data_interp = post.regridSpec(dataLV0selt,windowWidth=10)
        # add offsets from LV2 file:
        dataDWR = post.addOffsets(data_interp,dataLV2selt)
        specDWRxk = dataDWR.DWR_X_Ka
        specDWRkw = dataDWR.DWR_Ka_W
        #change conventions of DV
        specKa["dopplerKa"] = -specKa["dopplerKa"] #change conventions
        specW["dopplerW"] = -specW["dopplerW"] #change conventions
        specKaV["dopplerKa"] = -specKaV["dopplerKa"] #change conventions
        specKaHstrongnoisecorr["dopplerKa"] = -specKaHstrongnoisecorr["dopplerKa"] #change conventions
        specX["dopplerX"] = -specX["dopplerX"] #change conventions
        LDRspecKa["dopplerKa"] = -LDRspecKa["dopplerKa"] #change conventions
        LDRspecKaNoiseCorr["dopplerKa"] = -LDRspecKaNoiseCorr["dopplerKa"] #change conventions
        LDRspecW["dopplerW"]  = -LDRspecW["dopplerW"] #change conventions
        specDWRxk["doppler"] = -specDWRxk["doppler"] #change conventions
        specDWRkw["doppler"] = -specDWRkw["doppler"] #change conventions

        if ~np.isnan(MLtop_t) and ~np.isnan(MLbottom_t):
            #ZFR
            ZFR_str = "ZFR={0:.2f}".format(ZFR)
            ZFtop = ZfluxX.sel(range=MLtop_t) #flux top of melting layer
            #MDVtop
            MDVxTop = MDVx.sel(range=MLtop_t).values
            #MDVxTop500 = MDVx.sel(range=MLtop+500).values #has to match an existin value #TODO: use resample.nearest ?
            MDVxBot = MDVx.sel(range=MLbottom_t).values
            MDV_str = ("MDV$_{X,top}$=" + "{0:.2f}m/s".format(MDVxTop) +
                #"\nMDV$_{X,top}+500m$=" + "{0:.2f}m/s".format(MDVxTop500) 
                    "\nMDV$_{X,bottom}$=" + "{0:.2f}m/s".format(MDVxBot))
            #DWRxkTop
            DWRxkT = DWRxk.sel(range=MLtop_t).values #(ZeK.sel(range=MLtop)-ZeW.sel(range=MLtop)).values
            DWRxk_str = "DWR$_{X,Ka,TopML}$=" + "{0:.2f}dB\n".format(DWRxkT)
            #DWRkwTop
            DWRkwT = DWRkw.sel(range=MLtop_t).values #(ZeK.sel(range=MLtop)-ZeW.sel(range=MLtop)).values
            DWRkw_str = "DWR$_{Ka,W,TopML}$=" + "{0:.2f}dB ".format(DWRkwT)
            #RH (water)
            RHb = RH.sel(range=MLbottom_t).values
            RHb_str = "RH=" + "{0:.1f}".format(RHb)+ "%"
            #Peaks
            Peak_MLtop = Peaksselt.sel(range=(MLtop_t),method="nearest").copy()
            v_Peak1 = Peak_MLtop.sel(peakIndex=1).drop("peakIndex").peakVelClass #peak with smaller DV than main peak
            v_Peak2 = Peak_MLtop.sel(peakIndex=2).drop("peakIndex").peakVelClass #peak with smaller DV than main peak
            pow_Peak1 = Peak_MLtop.sel(peakIndex=1).drop("peakIndex").peakPowClass #peak with smaller DV than main peak
            pow_Peak2 = Peak_MLtop.sel(peakIndex=2).drop("peakIndex").peakPowClass #peak with smaller DV than main peak
            if pow_Peak1>-50: #filter noise
                if pow_Peak1<-30: #probably second ice mode
                    wtop_est = -v_Peak1.values
                else:
                    wtop_est= -v_Peak2.values
            else:
                wtop_est = np.nan
            #mie-notch
            wbot_est,__ = fc.get_mie_notch_DV(specW_MLbot,presMLbot,timestr=ti_spectra)
            #get mie-notches at all heights below MLbottom
            heights = []
            theo_mie_notch_DV = []
            mie_notch_DV = []
            specW_now = specW.copy() #need to change conventions here back again
            specW_now["dopplerW"] = -specW_now["dopplerW"] #change conventions
            if False: #True: #True: #deactivat mie-notch plotting (takes a lot of time)
                for height in specW.range:
                    if height>(MLbottom_t+10):
                        continue
                    if height<(MLbottom_t-500):
                        continue
                    theo_notch_terminal_vel = fc.get_mie_notch_DV_theor(pres.sel(range=height).values) #get mie-notch DV for w_vertical=0
                    [w_est_liq,DV_notch] = fc.get_mie_notch_DV(specW_now.sel(range=height),pres.sel(range=height).values) #get actual mie-notch DV
                    #save values
                    heights.append(height.values)
                    theo_mie_notch_DV.append(theo_notch_terminal_vel)
                    mie_notch_DV.append(DV_notch)
                    print("mie-notches","h",height.values,"actual notch",DV_notch,"theor. notch",theo_notch_terminal_vel,w_est_liq)

            #Peaks_now = Peaks.sel(range=height)).copy() #all peaks in 5 min interval at ML bottom
            w_str = "w$_{top}$=" + "{0:.2f}m/s ".format(wtop_est) + "\n" + "w$_{bot}$=" + "{0:.2f}m/s ".format(wbot_est)
    
            if ~np.isnan(wtop_est) or ~np.isnan(wbot_est):
                w_detected_str = "_w"
                if ~np.isnan(wtop_est):
                    w_detected_str += "T"
                if ~np.isnan(wbot_est):
                    w_detected_str += "B"
            else:
                w_detected_str = ""

            
            ZfluxXcorrTop = (Bd(ZeX.sel(range=MLbottom_t))*(MDVx.sel(range=MLbottom_t)-wbot_est)/(Bd(ZeX.sel(range=MLtop_t))* MDVx.sel(range=MLtop_t))).values*0.23 #derive reflectivity flux
            ZfluxXcorrBot = (Bd(ZeX.sel(range=MLbottom_t))*(MDVx.sel(range=MLbottom_t))         /(Bd(ZeX.sel(range=MLtop_t))*(MDVx.sel(range=MLtop_t)-wtop_est))).values*0.23 #derive reflectivity flux
            ZfluxXcorr    = (Bd(ZeX.sel(range=MLbottom_t))*(MDVx.sel(range=MLbottom_t)-wbot_est)/(Bd(ZeX.sel(range=MLtop_t))*(MDVx.sel(range=MLtop_t)-wtop_est))).values*0.23 #derive reflectivity flux
            ZFRcorrTop_str = "ZFR$_{corrTop}$=" + " {0:.2f}".format(ZfluxXcorrTop) 
            ZFRcorrBot_str = "ZFR$_{corrBot}$=" + " {0:.2f}".format(ZfluxXcorrBot) 
            ZFRcorr_str    = "ZFR$_{corr}$=" + " {0:.2f}".format(ZfluxXcorr)
        else:
            ZFR=np.nan
            ZFR_str = ""
            ZFtop = np.nan

            DWRxk_str = ""
            DWRkw_str = ""
            MDV_str = ""
            RHb_str = ""
            w_str = ""

            ZFRcorrTop_str = ""
            ZFRcorrBot_str = ""
            ZFRcorr_str = ""

            ZFRcorrTop_str = ""
            ZFRcorrBot_str = ""
            ZFRcorr_str = ""

        import matplotlib as mpl
        mpl.style.use('seaborn')
        mpl.rcParams['font.size'] = 35
        mpl.rcParams['legend.fontsize'] = 25
        mpl.rcParams['figure.titlesize'] = 25

        mpl.rcParams['legend.fancybox'] = True
        mpl.rcParams['legend.framealpha'] = 0.7
        mpl.rcParams['legend.facecolor']='silver'
        mpl.rcParams['legend.frameon']=True

        mpl.rcParams['ytick.labelsize']= 25
        mpl.rcParams['xtick.labelsize']= 25
        mpl.rcParams['axes.titlesize']=25
        mpl.rcParams['axes.labelsize']=25

        mpl.rcParams['lines.linewidth']=5

        fig,axes = plt.subplots(ncols=2,nrows=3,figsize=(20,25),sharey=True) 
        #integral moments
        for i_var,(height,xvar,label,xlims,ax) in enumerate(zip(
                                                              [ZeX.range,ZfluxX.range], #yvar
                                                              [ZeX,ZfluxX], 
                                                              ['Ze$_{X}$ [dBz]','F$_{Z,X}$ [mm$^6$/ (m/s)]'], #label
                                                              [[15,35],[1e2,1e4]], #xlims
                                                              axes[2:3,:].flatten())):
            #plot
            if i_var==0:
                ax.plot(xvar,height,color="k")
            elif i_var==1:
                ax.semilogx(xvar,height,color="k")
                
            ax.set_xlabel(label)
            ax.set_xlim(xlims)
            ax.set_ylim([height_lims[0],height_lims[1]])
            ax.axhline(MLtop_t,color="silver")
            ax.axhline(MLbottom_t,color="silver")
            ax.grid(True) #,ls='-.')
            if i_var==1: #label=='F$_{Z,X}$ [dB m/s]': #add ML top and one-to-ine ML bottom ZF lines and ZFR value
                ax.axvline(ZFtop,color="silver",linestyle="--")
                ax.axvline(ZFtop/0.23,color="silver",linestyle=":")
                #choose color for ZFRs
                colorZFR=["k","k","k","k"] #Uncorr ="k"; colorCorrTop="k"; colorCorrBot="k"; colorCorr="k"
                for i_col,ZFR in enumerate([ZFR,ZfluxXcorrTop,ZfluxXcorrBot,ZfluxXcorr]):
                    if ZFR>1.0:
                        colorZFR[i_col]="blue"
                    elif ~np.isnan(ZFR):
                        colorZFR[i_col]="red"
                plt.text(0.99, 0.90,w_str,horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=25,color="k")
                plt.text(0.99, 0.99,ZFR_str,horizontalalignment='right',verticalalignment='top',transform = ax.transAxes,fontsize=35,color=colorZFR[0])
                plt.text(0.99, 0.1,ZFRcorrTop_str + "\n",horizontalalignment='right',verticalalignment='bottom',transform = ax.transAxes,fontsize=25,color=colorZFR[1])
                plt.text(0.99, 0.05,ZFRcorrBot_str + "\n",horizontalalignment='right',verticalalignment='bottom',transform = ax.transAxes,fontsize=25,color=colorZFR[2])
                plt.text(0.99, -0.07,ZFRcorr_str + "\n",horizontalalignment='right',verticalalignment='bottom',transform = ax.transAxes,fontsize=35,color=colorZFR[3])

        #plot spectra
        for i_var,(xvar,yvar,var,clabel,xlims,lims,ax) in enumerate(zip(
                                                              [specW.dopplerW,specKa.dopplerKa,specX.dopplerX,LDRspecKa.dopplerKa,specDWRkw.doppler,specDWRxk.doppler], #xvar 
                                                              #[specKa.dopplerKa,specKaV.dopplerKa,specKaHstrongnoisecorr.dopplerKa,LDRspecW.dopplerW,LDRspecKa.dopplerKa,LDRspecKaNoiseCorr.dopplerKa], #xvar 
                                                              [specW.range,specKa.range,specX.range,LDRspecKa.range,specDWRkw.range,specDWRxk.range], #yvar
                                                              [specW,specKa,specX,LDRspecKa], #,specDWRkw,specDWRxk], #var
                                                              ['z$_{W}$ [dB]','z$_{Ka}$ [dB]','z$_{X}$ [dB]','LDR$_{Ka}$ [dB]','dwr$_{Ka,W}$ [dB]','dwr$_{X,Ka}$ [dB]'], 
                                                              [[-1,11],[-1,11],[-1,10],[-1,11],[-1,9],[-1,9]],  #xlims
                                                              [[-45,10],[-45,10],[-45,10],[-30,-5],[-5,20],[-5,20]], #,[-5,20]],# ,[-0.5,3]], #lims (colorbar)
                                                              #[axes[1,0],axes[1,1],axes[0,1],axes[0,0]])):#axes[:,0:2].flatten())):
                                                              [axes[1,0],axes[1,1],axes[0,1],axes[0,0]])):#axes[:,0:2].flatten())):
            if len(var.shape)>2:
                print("wrong shape of spectrum:", label)
                continue
            #plot
            mesh = ax.pcolormesh(xvar,yvar,var,vmin=lims[0],vmax=lims[1],cmap=getNewNipySpectral(),rasterized=True)
            ax.set_ylim([height_lims[0],height_lims[1]])

            if i_var==0:
                ax.plot(MDVw,MDVw.range,ls="--",c="k",label="MDV$_{W}$")
                ax.plot(mie_notch_DV,heights,label="actual mie-notch",linestyle="-",c="magenta") #,lw=2)
                ax.plot(theo_mie_notch_DV,heights,label="mie-notch (w=0)",linestyle="-",c="red") #,lw=2)
                ax.legend(loc="upper center")
            elif i_var==1:
                ax.plot(MDVk,MDVk.range,ls="--",c="k",label="MDV$_{Ka}$")
                for i_peak,pI in enumerate(Peaksselt.peakIndex):
                    peak = Peaksselt.sel(peakIndex=pI).copy()
                    if all(np.isnan(peak.peakVelClass.values)) or peak.peakIndex.values<0 or peak.peakIndex.values>2:
                        label="__None"
                    else:
                        label="Peak-" + str(peak.peakIndex.values)
                    ax.plot(-peak.peakVelClass,peak.range,ls=["","","","-","-",":",""][i_peak],c="white",label=label,marker=["","","","","D","",""][i_peak],markersize=15,markevery=5) #,lw=2)
                low_Edge_t  = Edgesselt.minVelXR #.sel(time=t_spectra)
                low_Edge_av = Edgesselt.minVelXR #.sel(time=slice(t,t+av_timedelta)).mean(dim="time")
                high_Edge_t = Edgesselt.maxVelXR #.sel(time=t_spectra, method='nearest')
                high_Edge_av= Edgesselt.maxVelXR #.sel(time=slice(t,t+av_timedelta)).mean(dim="time")
                ax.plot(-high_Edge_t,Edgesselt.range,ls="--",c="red",label="SEV") #,lw=2) # + pd.to_datetime(str(t_spectra.values)).strftime('%H:%M'))
                ax.plot(-low_Edge_t,Edgesselt.range,ls="--",c="red",label="__None") #lw=2) # + pd.to_datetime(str(t_spectra.values)).strftime('%H:%M'))
                #ax.plot(-high_Edge_av,Edgesselt.range,ls="-",lw=2,c="orange",label="Edge av.")
                ax.legend(bbox_to_anchor=(0.60, 1.0),loc="upper center", bbox_transform=ax.transAxes)
            elif i_var==2:
                ax.plot(MDVx,MDVx.range,ls="--",c="k",label="MDV$_{X}$")
                ax.legend(loc="upper center")
            elif i_var==3:
                pass
                ax2 = ax.twiny()
                ax2.plot(LDR,LDR.range,ls="--",c="k",label="Integrated LDR$_{Ka}$")
                ##ax2.plot(LDRfromSpec,LDRfromSpec.range,ls="--",c="magenta")
                ax2.legend(loc="upper center")
                ax2.set_xlim([-37,-5])
                ax2.set_xlabel(clabel)
                ax2.grid(b=False,which="both")
            elif i_var==4:
                ax2 = ax.twiny()
                ax2.plot(DWRkw,DWRkw.range,ls="--",c="k")
                ax2.set_xlim(lims)
                ax2.set_xlabel(clabel)
            elif i_var==5:
                ax2 = ax.twiny()
                ax2.plot(DWRxk,DWRxk.range,ls="--",c="k")
                ax2.set_xlim(lims)
                ax2.set_xlabel(clabel)

            #add and configure colorbar
            cbaxes = ax.inset_axes([1.0,0.6,0.08,0.4],facecolor="k")

            cb     =  plt.colorbar(mesh,cax=cbaxes, orientation='vertical',label=clabel)
            cbaxes.yaxis.set_ticks_position("left")
            cbaxes.yaxis.set_label_position("left")

            #plot ML top/bottom 
            ax.axhline(MLtop_t,color="silver")
            ax.axhline(MLbottom_t,color="silver")
            #ticks    
            #labels
            ax.set_xlabel('DV [m/s]')
            #grid and lims
            ax.grid(True,zorder=-1000) #,ls='-.')
            ax.set_xlim(xlims)

        axes[0][0].set_ylabel('height [m]')
        axes[1][0].set_ylabel('height [m]')
        axes[2][0].set_ylabel('height [m]')

        for ax,letter in zip(axes.flatten(),["a)","b)","c)","d)","e)","f)"]):
            plt.text
            plt.text(0.01, 0.99, letter,horizontalalignment='left',verticalalignment='top',transform = ax.transAxes,backgroundcolor="white")

        plt.subplots_adjust(left=0.07, bottom=0.07, right=0.95, top=0.93, wspace=0.2, hspace=0.2)
        #plt.tight_layout()
        if plot_all_times:  
            plt.savefig(outPath+ "t_all" + ti_spectra +'_spectra' + w_detected_str + '.png', facecolor=fig.get_facecolor(), edgecolor='none')
        else:
            plt.savefig(outPath+ ti +'_spectra' + w_detected_str + '.png', facecolor=fig.get_facecolor(), edgecolor='none')
            plt.savefig(outPath+ ti +'_spectra' + w_detected_str + '.pdf', facecolor=fig.get_facecolor(), edgecolor='none')
            print('saved to: ' + outPath+ ti +'_spectra' + w_detected_str + '.png')
        plt.close()
        print(ti,' finished')     
