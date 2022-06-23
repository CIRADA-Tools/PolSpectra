#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Example code demonstrating the use of the PolSpectra class.


Created on Thu Jun 23 11:37:23 2022

@author: cvaneck
"""

import numpy as np
import polspectra


def Faraday_thin_complex_polarization(freq_array,RM,Polint,initial_angle):
    """Function to produce Stokes Q/U spectra for Faraday simple source.
       freq_array = channel frequencies in Hz
       RM = source RM in rad m^-2
       Polint = polarized intensity in whatever units
       initial angle = pre-rotation polarization angle (in degrees)"""
    l2_array=(299792458./freq_array)**2
    Q=Polint*np.cos(2*(np.outer(l2_array,RM)+np.deg2rad(initial_angle)))
    U=Polint*np.sin(2*(np.outer(l2_array,RM)+np.deg2rad(initial_angle)))
    return np.squeeze(np.transpose(Q+1j*U))


#Set up channel frequencies for all sources.
N_chan=288
freq_arr=np.linspace(800e6,1088e6,num=N_chan)

N_spectra=100

#Creating random spectra. This produces lists of arrays for each Stokes parameter.
I_spectra=[]
Q_spectra=[]
U_spectra=[]
noise_amplitude=0.001
for i in range(N_spectra):
    #Stokes I spectra are power laws with random brightnesses.
    Ivalues=np.random.lognormal(mean=-2,sigma=1,size=N_chan)*(freq_arr/np.mean(freq_arr))**-0.7
    polarization=Faraday_thin_complex_polarization(freq_arr,RM=np.random.uniform(-1000,1000), #Random RM
                                                  Polint=np.random.uniform(0,0.7), #Random fractional polarization
                                                  initial_angle=np.random.uniform(0,180)) #Random angle
    I_spectra.append(Ivalues+noise_amplitude*np.random.normal(0,1,N_chan))
    Q_spectra.append(Ivalues*polarization.real+noise_amplitude*np.random.normal(0,1,N_chan))
    U_spectra.append(Ivalues*polarization.imag+noise_amplitude*np.random.normal(0,1,N_chan))

#Make the corresponding Stokes uncertainty arrays, for each source.
I_errors=[np.repeat(noise_amplitude,N_chan) for x in range(N_spectra)]
Q_errors=[np.repeat(noise_amplitude,N_chan) for x in range(N_spectra)]
U_errors=[np.repeat(noise_amplitude,N_chan) for x in range(N_spectra)]

#Random coordinates:
ra_array=np.random.uniform(0,360,N_spectra)
dec_array=np.random.uniform(-90,90,N_spectra)


#Create the PolSpectra table.
#Note that freq_arr is a single 1D array; the code knows it should be 2D
#or a list of arrays, so it assumes all sources have the same frequency 
#values and expands it to the appropriate dimensions.
#Setting the source_number column using range(N_spectra) gives each source
#a unique ID number. Quantities that are common to all sources, such as the 
#beam size, can be set for all sources using single values.
#Optional channels defined in the standard (such as coordinate_system and 
#channel_width) can also be added.
spectrumtable=polspectra.from_arrays(ra_array,dec_array,freq_arr,
                                    I_spectra,I_errors,Q_spectra,Q_errors,U_spectra,U_errors,
                                    source_number_array=range(N_spectra),
                                    beam_maj=0.01,beam_min=0.01,beam_pa=0,
                                    coordinate_system='icrs',channel_width=1e6)


#Including an extra column, that isn't part of the standard:
extra_column=np.random.randint(0,100,size=N_spectra)
spectrumtable.add_column(extra_column,name='random_integer',
                         description='A random integer for each source',units='')

#Manipulate table, extracting columns and rows:
print(spectrumtable[10])
print(spectrumtable['ra'])



print(spectrumtable.columns)


#Extract all data for a specific source:
print(spectrumtable[spectrumtable['source_number'] == 5])


#A verification function is provided which confirms that all columns
#have consistent numbers of channels (per row), and provides warnings
#if certain parameters (frequencies, beam sizes) seem like they may
#have the wrong units.
spectrumtable.verify_table()


spectrumtable.write_FITS('example_polspectra.fits',overwrite=True)
table_from_file=polspectra.from_FITS('example_polspectra.fits')




