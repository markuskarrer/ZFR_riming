#!/usr/bin/env python3
# coding: utf-8

# # calculate dielectric factor for ice-air mixtures
# 
# the objective is to estimate how much of the dependecy of ZFR on the degree of riming could results from a change of the dielectric factor


import numpy as np
import snowScatt
import matplotlib.pyplot as plt
from pdb import set_trace

def mass2reff(mass):
    vol = mass/ice_density
    reff = np.cbrt(vol*3.0/(4.0*np.pi))
    return reff

freq = 9.4e9 #[Hz] X-Band
temperature = 273.15 #[K] close to melting layer
ice_density = snowScatt._compute._ice_density #917.kg/m^3 bulk ice density
water_density = 1000. #[kg/m^3]
rhoSnowArr = np.linspace(100.,911.,20) #[kg/m^3] effective snow density

fig,axes = plt.subplots(ncols=3,figsize=(12,8))
for order in ["ice in air","air in ice"]:
    for i_mf,mixing_formula in enumerate(["Bruggeman","Maxwell_Garnett","Sihvola"]):
        color=["b","g","r"][i_mf]
        K2mix = []
        for rhoSnow in rhoSnowArr:
            iceRatio = rhoSnow/ice_density


            #get eps of ice first
            epsIce = snowScatt.refractiveIndex.ice.eps(temperature, freq)
            K2Ice = snowScatt.refractiveIndex.utilities.K2(epsIce)
            #print("iceRatio",iceRatio,"K2Ice",K2Ice,ice_density)
            
            if order=="ice in air":
                volume_fractions  = (iceRatio,1-iceRatio)
                epsMix = snowScatt.refractiveIndex.mixing.eps([epsIce,complex(1.0-0.0)], volume_fractions, model=mixing_formula, ni=0.85)
            else:
                volume_fractions  = (1-iceRatio,iceRatio)
                epsMix = snowScatt.refractiveIndex.mixing.eps([complex(1.0-0.0),epsIce], volume_fractions, model=mixing_formula, ni=0.85)
            K2mix.append(snowScatt.refractiveIndex.utilities.K2(epsMix))
            print("rhoSnow",rhoSnow,"K2mix",K2mix) #,"K2mix/K2Ice",K2mix/rhoSnow**2*water_density**2)

        K2mix = np.array(K2mix) #convert to numpy array
        print(K2Ice)

        K2liq = 0.93 

        if order=="air in ice":
            linestyle="--"
        else:
            linestyle="-"

        axes[0].plot(rhoSnowArr,K2mix,label=mixing_formula + "(" + order + ")",linestyle=linestyle,color=color) #/rhoSnowArr**2*water_density**2)
        axes[0].set_xlabel(r"$\rho_{snow}$ [kg/m$^3$]")
        axes[0].set_ylabel(r"|K$_{snow}|^2$")

        axes[1].plot(rhoSnowArr,K2mix/rhoSnowArr**2*water_density**2,linestyle=linestyle,color=color)
        axes[1].set_xlabel(r"$\rho_{snow}$ [kg/m$^3$]")
        axes[1].set_ylabel(r"|K$_{snow}|^2$  $\rho_{liq}^2$ $\rho_{snow}^{-2}$")

        axes[2].plot(rhoSnowArr,K2liq/K2mix*rhoSnowArr**2/water_density**2,linestyle=linestyle,color=color)
        axes[2].set_xlabel(r"$\rho_{snow}$ [kg/m$^3$]")
        axes[2].set_ylabel(r"|K$_{liq}$|$^{2}$ |K$_{snow}$|$^{-2}$  $\rho_{snow}^2$ $\rho_{liq}^{-2}$")

axes[0].legend()

plt.tight_layout()
plt.savefig("plots/K2iceAirMixDens.pdf")
