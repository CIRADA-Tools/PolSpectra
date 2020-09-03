#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 16:45:56 2020
@author: Cameron Van Eck

A package for reading/writing polarization spectra, in a format that is
hopefully compatible with VO standards.

"""

import numpy as np
import astropy.io.fits as pf
import astropy.table as at




class polarizationspectra:
    """This class defines all the data needed for polarization spectra, along 
    with all the associated observation metadata.

    This is intended to function effectively as a 'table' format: each piece
    of data or metadata becomes a column in the table. Each row contains a
    single source spectrum, from a single observation. Sources with multiple
    observations should appear as multiple rows, each with meta data associated
    with that specific observation.
    No sub-class for single rows is being defined, since I don't think it would
    be useful. A single row can exist as a table with one row. Combining rows
    can be accomplished with table concatenation.
    
    The goal of this class is to be able to read and write FITS files with 
    polarization spectra, that are compliant with FITS standards. 
    It needs to be able to extract information into standard formats like 
    numpy arrays, or maybe AstroPy tables.
    
    Maybe internal representation should be Astropy Tables? That would make data
    access easy. Not sure if Tables support some of the variable representation
    aspects...
    Should this support custom metadata columns? Probably. How to implement?
    
    
    Different rows can have different data lengths. This could be slightly 
    tricky for numpy. Not sure how Astropy Tables handles this.
    """
    
    def __init__(self):
        self.Nrows = 0  #number of rows in the table
        self.Nsrc = 0  #number of unique sources in the table. A useful property
                     #to access for adding new sources with unique numbers.
        self.table=at.Table()  #Empty table.

    
    def create_from_arrays(self,ra_array,dec_array, freq_array, StokesI,StokesI_error,
                 StokesQ,StokesQ_error,StokesU,StokesU_error,source_number_array,
                 unit,beam_major,beam_minor,beam_pa,
                 StokesV=None,StokesV_error=None,quality=None,
                 quality_meanings=None,source_name=None,coordinate_system=None,
                 telescope=None,epoch=None,integration_time=None,
                 channel_width=None,aperture=None):
        """Inialize a polarized spectrum. Takes in combination of (channel) 
        arrays, columns, and scalars, for mandatory columns and standard 
        optional columns. Check parameter descriptions below or the 
        documentation for full descriptions.
        2D array-like inputs corresponds to quantities that vary by source and channel
            These need to be stored as 'object' columns, because the number of
            channels can vary by source/observation
        1D array-like inputs correspond to quantities that (may) vary by source
        Scalar inputs correspond to quantities that do not vary by source.
        Array-like here means something that can be converted into a Astropy
        Column object, such as a numpy array or Python list. It is not the case
        that every row must have the same number of channels!
        Required Parameters:
            ra_array: array-like object containing Right Ascensions (in degrees!)
            dec_array: array-like object containing Declinations (in degrees!)
            freq_array: 1D or 2D (source x channel) array-like containing channel
                        frequencies., in Hz.
            Stokes[I/Q/U][_error]: 2D (source x channel) array-like objects 
                        containing Stokes parameter values and their errors.
            source_number_array: array_like object containing source numbers 
                            (integers, which indicate how rows are grouped into sources).
                            Assumed to run from 0..N_src-1
            unit: scalar string containing flux unit for all Stokes arrays
            beam_major,beam_minor,beam_pa: beam parameters as scalars or 
                                            array-likes, in degrees.
        Optional parameters:
            StokesV[_error]: array-like with Stokes V values and errors.
            quality: array-like containing channel quality flags.
            quality_meanings: string or array-like of strings explaining all
                                possible quality flag codes
            source_name: array-like containing names of sources (strings)
            coordinate_system: Coordinate system (if unspecified assumed ICRS)
                            (scalar or array-like)
            telescope: Telescope name (scalar or array-like)
            epoch: scalar or array-like of (midpoint) time of observation (MJD)
            integration_time: scalar or array-like of duration of observation (in seconds)
            channel_width: scalar, 1D or 2D array_like of channel widths (in Hz)
            aperture: scalar or array-like, diameter/length of the averaging 
                    region used to determine the source spectra (in degrees)
            """

#Vector quantities are going to be arrays of N_channel elements
#Scalars take on one value per source-observation
#'Either' means it could depend on the observation. For example, if the beam isn't convolved it has a frequency dependence.
#Mandatory quantities MUST be present. The code should make sure they are present.
#Optional quantities are recommended for maximizing the value of the data. They may represent reserved keywords?
#All other quantities should be supported.

#    The IVOA standard suggests that certain things can be either single values
#     (same for all channels and sources), columns (same for all channels),
#     or columns-of-arrays (vary by source and channel). This makes it a bit 
#     complicated. Do I want to support this? It's very good for storage 
#     efficiency: things that are repeated get reduced to single values.
        #TODO: consider turning constants into metadata? Can Astropy Table
        #       metadata be converted into FITS without loss???
                 
        #Quality checks: -do all columns have the same number of rows?
        #                -do all columns have the same number of channels (within a row)?
        #                -are frequencies in Hz?
        #                -does coordinate system match expectations?
        #                -are units of beam units reasonable?
        #For the first pass, I'm not going to implement these (to save time)
        #Some, like # of row mismatches, will cause errors for the user.
        
        self.Nrows=len(ra_array)
        ra_column=at.Column(data=ra_array,name='ra',
                            description='Right Ascension',unit='deg')
        dec_column=at.Column(data=dec_array,name='dec',
                            description='Declination',unit='deg')
        
        #frequency array may be 1D (if all sources have same channels)
        if get_dimensionality(freq_array) == 2:
            freq_2D=freq_array
        elif get_dimensionality(freq_array) == 1:
            freq_2D=[freq_array for i in range(self.Nrows)]
        else:
            raise Exception('Frequency column must be channel-wise: at least 1D array!')
        
        
        #This get a bit fussy. To allow mergers of heterogenous tables down the line,
        #it is necessary to ensure that the column headers know absolutely NOTHING
        # about the number of channels.
        freq_column=at.Column(name='freq',dtype='object',shape=(),length=self.Nrows,
                            description='Channel Frequency',unit='Hz')
        freq_column[:]=freq_2D
    
        StokesI_column=at.Column(name='StokesI',dtype='object',shape=(),length=self.Nrows,
                            description='Stokes I per channel',unit=unit)
        StokesI_column[:]=StokesI
        StokesI_error_column=at.Column(name='StokesI_error',shape=(),length=self.Nrows,
                           dtype='object',description='StokesI error per channel',unit=unit)
        StokesI_error_column[:]=StokesI_error
        StokesQ_column=at.Column(name='StokesQ',dtype='object',shape=(),length=self.Nrows,
                            description='Stokes Q per channel',unit=unit)
        StokesQ_column[:]=StokesQ
        StokesQ_error_column=at.Column(name='StokesQ_error',shape=(),length=self.Nrows,
                           dtype='object',description='Stokes Q error per channel',unit=unit)
        StokesQ_error_column[:]=StokesQ_error

        StokesU_column=at.Column(name='StokesU',dtype='object',shape=(),length=self.Nrows,
                            description='Stokes U per channel',unit=unit)
        StokesU_column[:]=StokesU
        StokesU_error_column=at.Column(name='StokesU_error',shape=(),length=self.Nrows,
                           dtype='object',description='Stokes U error per channel',unit=unit)
        StokesU_error_column[:]=StokesU_error

        source_number_column=at.Column(data=source_number_array,name='source_number',
                       dtype='int',description='Source ID number in file',unit='')

        #Set the number of sources (assumed to be zero-indexed)
        self.Nsrc=len(np.unique(source_number_column))

        #Beam sizes array may be scalar or 1D
        #TODO: Support freq-dependent beam sizes??
        try:
            _=beam_major[0] #assume all beam parameters have the same dimensionality, test one.
            beam_major_1D=beam_major
            beam_minor_1D=beam_minor
            beam_pa_1D=beam_pa
        except:
            beam_major_1D=[beam_major for i in range(self.Nrows)]
            beam_minor_1D=[beam_minor for i in range(self.Nrows)]
            beam_pa_1D=[beam_pa for i in range(self.Nrows)]
            
        beam_major_column=at.Column(data=beam_major_1D,name='beam_major',
                                    dtype='float',unit='deg',
                                    description='Beam major axis in deg')
        beam_minor_column=at.Column(data=beam_minor_1D,name='beam_minor',
                                    dtype='float',unit='deg',
                                    description='Beam minor axis in deg')
        beam_pa_column=at.Column(data=beam_pa_1D,name='beam_pa',
                                    dtype='float',unit='deg',
                                    description='Beam position angle in deg')


        #Calculate N_chan and make into a column:
        N_chan_array=np.zeros(self.Nrows)
        for i in range(self.Nrows):
            N_chan_array[i]=len(freq_column[i])
        N_chan_column=at.Column(data=N_chan_array,name='num_chan',
                   dtype='int',description='Number of channels')


        #Assemble basic table:
        self.table=at.Table([source_number_column,ra_column,dec_column,freq_column,
                             StokesI_column,StokesI_error_column,
                             StokesQ_column,StokesQ_error_column,
                             StokesU_column,StokesU_error_column,
                             beam_major_column,beam_minor_column,beam_pa_column,
                             N_chan_column])
        #Now adding the optional columns:
        if StokesV is not None: #Check that both Stokes V and error are supplied?
            StokesV_column=at.Column(name='StokesV',dtype='object',shape=(),length=self.Nrows,
                            description='Stokes V per channel',unit=unit)
            StokesV_column[:]=StokesV
            self.table.add_column(StokesV_column)

            StokesV_error_column=at.Column(name='StokesV_error',shape=(),length=self.Nrows,
                           dtype='object',description='Stokes V error per channel',unit=unit)
            StokesV_error_column[:]=StokesV_error
            self.table.add_column(StokesV_error_column)
        
        if quality != None: #Check quality is 2d?
            quality_column=at.Column(name='quality',dtype='object',shape=(),length=self.Nrows,
                            description='Quality flags per channel')
            quality_column[:]=quality
            self.table.add_column(quality_column)

        if quality_meanings != None:
            quality_meanings_column=at.Column(
                        data=possible_scalar_to_1D(quality_meanings,self.Nrows),
                        name='quality_meanings',dtype='str',
                        description='Description of quality flag meanings')
            self.table.add_column(quality_meanings_column)

        if source_name != None:
            source_name_column=at.Column(data=source_name,name='source_name',dtype='str',
                            description='Source name')
            self.table.add_column(source_name_column)

        if coordinate_system != None:
            coordinate_system_column=at.Column(
                        data=possible_scalar_to_1D(coordinate_system,self.Nrows),
                        name='coordinate_system',dtype='str',
                        description='WCS Coordinate system')
            self.table.add_column(coordinate_system_column)
            
        if telescope != None:
            telescope_column=at.Column(
                        data=possible_scalar_to_1D(telescope,self.Nrows),
                        name='telescope',dtype='str',
                        description='Telescope')
            self.table.add_column(telescope_column)

        if epoch != None:
            epoch_column=at.Column(
                        data=possible_scalar_to_1D(epoch,self.Nrows),
                        name='epoch',dtype='float',
                        description='Observation Epoch (midpoint, MJD)',unit='days')
            self.table.add_column(epoch_column)
        
        if integration_time != None:
            integration_time_column=at.Column(
                        data=possible_scalar_to_1D(integration_time,self.Nrows),
                        name='integration_time',dtype='float',
                        description='Integration time (observation duration, s)',unit='s')
            self.table.add_column(integration_time_column)
            
        if channel_width != None: #Slightly complicated: may be scalar, 1D, 2D.
             #For now, just leave it as 1D or 2D, or expand to 1D if scalar.
             #Whether it's better to have 2D as the default is an open question.
             if get_dimensionality(channel_width) == 2:
                 col_dtype='object'
             else:
                 col_dtype='float'
             
             channel_width_column=at.Column(
                        data=possible_scalar_to_1D(channel_width,self.Nrows),
                        name='channel_width',dtype=col_dtype,unit='Hz',
                        description='Channel bandwidth [Hz]')
             self.table.add_column(channel_width_column)

        if aperture != None:
            aperture_column=at.Column(
                        data=possible_scalar_to_1D(aperture,self.Nrows),
                        name='aperture',dtype='float',unit='deg',
                        description='Integration aperture (diameter, deg)')
            self.table.add_column(aperture_column)
        
        #Done!
           
    def add_column(self,values,name,description,units=''):
        """Add a column with a single value per row, or a scalar.
        Parameters:
            values : scalar or array-like
                value(s) for the column to hold
            name : str 
                Column name and key for reference within table
            description : str
                Description of the column contents (human readable)
            units : str
                Physical unit of the column quantity, if any"""
        new_column=at.Column(
                        data=possible_scalar_to_1D(values,self.Nrows),
                        name=name,unit=units,
                        description=description)
        self.table.add_column(new_column)

    def add_channel_column(self,values,name,description,units):
        """Add a column with a value for every channel in each row.
        Parameters:
            values : 2D array-like
                value(s) for the column to hold
            name : str 
                Column name and key for reference within table
            description : str
                Description of the column contents (human readable)
            units : str
                Physical unit of the column quantity, if any
            dtype : str
                Data type to be stored as. Channel-wise columns MUST have
                dtype='object' in order to be saved properly."""
        #Verify the size is correct:
        if len(values) != self.Nrows:
            raise Exception('New column does not have same number of rows as table!')
        for i in range(self.Nrows):
            if len(values[i]) != self['num_chan'][i]:
                raise Exception("New column doesn't have same number of channels as rest of table, on row {}.".format(i))
        
        new_column=at.Column(length=self.Nrows,
                        name=name,unit=units,dtype='object',
                        description=description)
        new_column[:]=values
        self.table.add_column(new_column)        


    def __repr__(self):
        #pass through directly to the table
        return self.table.__repr__()

    def __str__(self):
        #pass through directly to the table
        return self.table.__str__()
    
    def __len__(self):
        return self.Nrows
    
    def __getitem__(self,key):
        """Depending on type of key, returns column, or polarizationspectra 
        with selected rows.
        Generally passes item selection through to the underlying table,
        except for single row access (which is converted to a one-row polarizationspectra)
        and multiple row access (which is converted from a table to a polarizationspectra)."""

        if isinstance(key, (int, np.integer)):
            key=[key,]
        
        val=self.table[key]
        if isinstance(val,at.table.Table):
            polspec=polarizationspectra()
            polspec.table=val
            polspec.Nrows=len(val)
            polspec.Nsrc=len(np.unique(val['source_number']))
            return polspec
        else:
            return val

    def __setitem__(self,key,item):
        """Passes through directly to the underlying Astropy Table. Should not
        be used often, hopefully. Maybe I should just make the table immutable?"""
        self.table[key]=item

    def write_FITS(self,filename,overwrite=False):
        """Write the polspectra to a FITS file.
        Parameters:
            filename : str
                Name and relative path of the file to save to.
            overwrite : bool [False]
                Overwrite the file if it already exists?"""
        #This is going to be complicated, because the automatic write algorithm
        # doesn't like variable length arrays. pyfits can support it, it just
        # needs a little TLC to get it into the correct format.
        #TODO: Converting to FITSrec format loses the column description...
        # can this be preserved in metadata/header? Will think about.
        fits_columns=[]
        col_descriptions=[]
        
        #per column, convert to fits column:
        for col in self.table.colnames:
            tabcol=self.table[col]
            if tabcol.dtype != np.dtype('object'): #Normal columns
                col_format=pf.column._convert_record2fits(tabcol.dtype)
            else: #Channelized columns
                subtype=np.result_type(tabcol[0][0]) #get the type of each element in 2D array
                col_format='P'+pf.column._convert_record2fits(subtype)+'()'
            if tabcol.unit != None:
                unit=tabcol.unit.to_string()
            else:
                unit=''
            pfcol=pf.Column(name=tabcol.name,unit=unit,
                            array=tabcol.data,format=col_format)
            fits_columns.append(pfcol)
            col_descriptions.append(tabcol.description)
        
        tablehdu=pf.BinTableHDU.from_columns(fits_columns)
        tablehdu.writeto(filename,overwrite=overwrite)
        
    
    def read_FITS(self,filename):
        """Read in a polarization spectrum table from a FITS file.
        Parameters:
            filename: str
                Relative path and name of file to read from."""
        hdu=pf.open(filename)
        data=hdu[1].data
        header=hdu[1].header
        self.table=at.Table(data)
        #The FITS to table conversion doesn't carry the units, so this needs
        #to be done manually:
        for i in range(1,header['TFIELDS']+1):
            if 'TUNIT{}'.format(i) in header.keys():
                self.table[header['TTYPE{}'.format(i)]].unit=header['TUNIT{}'.format(i)]
        
        self.Nrows=len(data)
        self.Nsrc=len(np.unique(self.table['source_number']))
        hdu.close()
        

    
    def merge_tables(self,table2,merge_type='exact',source_numbers='keep'): 
        """Merge another polarization spectrum table into this one, even if
        the columns aren't identical. User selects how mis-matched columns are
        integrated using the merge_type parameter:
            'exact': all column names/types must match exactly.
            'outer': all columns are kept; missing values are masked out.
            'inner': only columns common to both inputs are kept.
            
        Parameters:
            table2 : polarizationspectra object: 
                the second table to be added to this one.
            merge_type : str (only accepted values are 'inner','exact','outer')
                Desired behaviour for dealing with mis-matched columns.
            source_numbers : str ('keep or 'concat')
                Desired behaviour for source_number column.
                'keep' does not change the source_number column at all; this
                    is desireable if you want to combine different observations
                    of the same source(s).
                'concat' shifts the source numbers of the new rows up to be 
                    unique; this is desireable if you want to combine different
                    sources together but haven't manually set the source numbers
                    to be distinct.
            """
        if (source_numbers != 'keep') and (source_numbers != 'concat'):
            raise Exception('Incorrect option given for source_number behaviour in table merging!')

        if merge_type not in ['exact','inner','outer']:
            raise Exception('Incorrect option given for merge_type in table merging')

        add_table=table2.table.copy() #Copy the 2nd table to avoid accidently writing to it.
        if source_numbers == 'concat': #Reset the numbering in the 2nd table 
            #so that continues where the 1st table ends. Doesn't prevent gaps!
            add_table['source_number']=(add_table['source_number']-
                np.min(add_table['source_number'])+np.max(self.table['source_number'])+1 )

        merged_table=at.vstack([self.table,add_table],
                               join_type=merge_type,metadata_conflicts='error')
        self.table=merged_table
        self.Nsrc=len(np.unique(self.table['source_number']))
        self.Nrows=self.Nrows+table2.Nrows

        
    
    def renumber_sources(self):
        """Re-orders the source_number column to be consecutive numbers from
        0 to N_src-1. Useful when extracting a subset of sources."""
        source_numbers,indices=np.unique(self.table['source_number'],return_index=True)
        source_numbers=source_numbers[np.argsort(indices)]
        old_source_number_column=self.table['source_number'].copy()
        for i in range(len(source_numbers)):
            w=np.where(old_source_number_column == source_numbers[i])
            self.table['source_number'][w] = i
            
    def copy(self):
        """Returns a copy of the current instance."""
        new_copy=polarizationspectra()
        new_copy.table=self.table.copy()
        new_copy.Nrows=self.Nrows
        new_copy.Nsrc=self.Nsrc
        return new_copy
    

def from_FITS(filename):
    """Read in a polarization spectrum table from a FITS file.
    Parameters:
        filename: str
            Relative path and name of file to read from."""
    polspec=polarizationspectra()
    polspec.read_FITS(filename)
    return polspec

def from_arrays(ra_array,dec_array, freq_array, StokesI,StokesI_error,
                 StokesQ,StokesQ_error,StokesU,StokesU_error,source_number_array,
                 unit,beam_major,beam_minor,beam_pa,
                 StokesV=None,StokesV_error=None,quality=None,
                 quality_meanings=None,source_name=None,coordinate_system=None,
                 telescope=None,epoch=None,integration_time=None,
                 channel_width=None,aperture=None):
    
    new_spectra=polarizationspectra()
    new_spectra.create_from_arrays(ra_array, dec_array, freq_array, StokesI, 
                                   StokesI_error, StokesQ, StokesQ_error, 
                                   StokesU, StokesU_error, source_number_array, 
                                   unit, beam_major, beam_minor, beam_pa,
                                   StokesV,StokesV_error,quality,
                                   quality_meanings,source_name,coordinate_system,
                                   telescope,epoch,integration_time,
                                   channel_width,aperture)
    return new_spectra
from_arrays.__doc__=polarizationspectra.create_from_arrays.__doc__

def possible_scalar_to_1D(scalar,Nrows):
    """If the input value is not indexable (and not a string), extend it to a list
    of Nrows values. If it is indexable, leave it as is (unless it's a string)"""
    try: #Test 1D vs scalar
        _=scalar[0]
        if type(scalar) == str:  #String are indexible scalars, so check first!
            column=[scalar for i in range(Nrows)]
        else:
            column=scalar
    except: #If it's not indexible, assume it's a scalar:
        column=[scalar for i in range(Nrows)]
    return column

    
def get_dimensionality(column):
    """This function determines the dimensionality of a variable, to see if
    it contains scalar, 1D, or 2D data. Current version fails on strings."""
    try:
        _=column[0]
        try:
            _=column[0][0]
            return 2
        except: return 1
    except:
        return 0
    



