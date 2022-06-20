#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A package for reading/writing polarization spectra following the PolSpectra
standard. Includes functions to create PolSpectra tables in Python (as Astropy
Tables), read and write them as FITS files, and basic manipulation (extracting
selections of the table, merging tables, etc).


"""

import numpy as np
import astropy.io.fits as pf
import astropy.table as at
import astropy.coordinates as ac
import astropy.io.votable as vot



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
    
    This is all built on top of astropy Tables. The main complicating factor is
    that the array columns can have different lengths in different rows. This
    is because different observations (with differing number of frequencies)
    can be concatenated into the same table. To support this, all the array
    columns are turned into 'object' columns, where the object in each row
    is an array of arbitrary length.
    
    Astropy.io.fits has partial support for these kinds of columns, but it takes
    a little finessing to get them to read/write properly. This class has
    functions to take care of that.
    
    """
    
    def __init__(self):
        self.Nrows = 0  #number of rows in the table
        self.Nsrc = 0  #number of unique sources in the table. A useful property
                     #to access for adding new sources with unique numbers.
        self.table=at.Table()  #Empty table.
        self.columns=[]  #A convnient list of column names (as strings)
        self.table.meta['VERSION']=0.2  
        #Version 0.2: updated after first round of coauthor comments
        #             New columns, new functions for quality control and such.

    
    def create_from_arrays(self,long_array,lat_array, freq_array, stokesI,stokesI_error,
                 stokesQ,stokesQ_error,stokesU,stokesU_error,source_number_array,
                 beam_maj,beam_min,beam_pa,coordinate_system='icrs',
                 stokesV=None,stokesV_error=None,quality=None,
                 quality_meanings=None,ionosphere=None,cat_id=None,dataref=None,
                 telescope=None,epoch=None,integration_time=None,interval=None,
                 leakage=None,channel_width=None,flux_type=None,aperture=None):
        """Inialize a polarized spectrum. Takes in combination of (channel) 
        arrays, columns, and scalars, for mandatory columns and standard 
        optional columns. Check parameter descriptions below or the 
        documentation for full descriptions.
        2D array-like inputs corresponds to quantities that vary by source and 
            channel (in that order: source first, then channel)
            These need to be stored internally as 'object' columns, because the 
            number of channels can vary by source/observation
        1D array-like inputs correspond to quantities that (may) vary by source,
            with one entry per row.
        Scalar inputs correspond to quantities that do not vary by source.
        Array-like here means something that can be converted into a Astropy
        Column object, such as a numpy array or Python list. This code has been
        tested with Python lists/nested lists, and with numpy arrays. Other
        input types are not guaranteed to work.
        It is not the case that every row must have the same number of channels.
        Required Parameters:
            long_array: array-like object containing Right Ascensions 
                        or Galactic longitudes, in decimal degrees.
                        Coordinate system is set with the coordinate_system
                        keyword. Will be converted to ICRS and Galactic for
                        the final table.
            lat_array: array-like object containing Declinations 
                       or Galactic latitudes, in decimal degrees.
                       Coordinate system is set with the coordinate_system
                       keyword. Will be converted to ICRS and Galactic for
                        the final table.
            freq_array: 1D or 2D (source x channel) array-like containing channel
                        frequencies, in Hz. If 1D, it is assumed each row
                        has the same frequency channels, and it will be expanded
                        out to a 2D array.
            Stokes[I/Q/U][_error]: 2D (source x channel) array-likes  
                        containing Stokes parameter values and their errors.
            source_number_array: array_like object containing source numbers 
                            (integers, which indicate how rows are grouped into sources).
                            For example, could run from 0..N_src-1, but is not
                            required to.
            beam_maj,beam_min,beam_pa: beam parameters as scalars, 1D or 2D
                        array-likes, in degrees. If scalar, will be expanded out
                        to each row. If 1D, will be assumed to be 1 value per row.                        
            coordinate_system: string containing coordinate system name, as
                               recognized by astropy.coordinates. Defaults
                               to 'icrs', common alternatives include
                               'fk4' (B1950), 'fk5' (J2000), 'galactic'

        Optional parameters:
            stokesV[_error]: array-like with Stokes V values and errors.
            quality: array-like containing channel quality flags.
            quality_meanings: string or array-like of strings explaining all
                                possible quality flag codes
            cat_id: array-like containing names of sources (strings)
            telescope: Telescope name (scalar or array-like)
            epoch: scalar or array-like of (midpoint) time of observation (MJD)
            integration_time: scalar or array-like of duration of observation (in seconds)
            interval: scalar or array-like of interval of observation (in days)
            leakage: scalar, 1D or 2D array-like of estimated leakage from 
                    Stokes I into Q and U, as a fraction of Stokes I.
            channel_width: scalar, 1D or 2D array_like of channel widths (in Hz)
            flux_type: scalar or 1D array-like of strings describing how the
                        Stokes spectra were extracted (peak pixel, integrated, fit, etc)
            aperture: scalar or array-like, diameter/length of the averaging 
                    region used to determine the source spectra (in degrees)
            """


                 
        
        self.Nrows=len(long_array)
        #If different columns have different numbers of rows, table creation
        #will fail. So no explicit check of vector length is needed.
        
        #Convert coordinates into ICRS and Galactic:
        coordinates=ac.SkyCoord(long_array,lat_array,
                                frame=coordinate_system,unit='deg')        
        ra_column=at.Column(data=coordinates.icrs.ra.deg,name='ra',
                            description='Right Ascension',unit='deg')
        dec_column=at.Column(data=coordinates.icrs.dec.deg,name='dec',
                            description='Declination',unit='deg')
        glon_column=at.Column(data=coordinates.galactic.l.deg,name='l',
                            description='Galactic Longitude',unit='deg')
        glat_column=at.Column(data=coordinates.galactic.b.deg,name='b',
                            description='Galactic Latitude',unit='deg')
        
        
        #frequency array may be 1D (if all sources have same channels), 
        #and must support expansion to 2D as needed.
        if _get_dimensionality(freq_array) == 2:
            freq_2D=freq_array
        elif _get_dimensionality(freq_array) == 1:
            freq_2D=[freq_array for i in range(self.Nrows)]
        else:
            raise Exception('Frequency column must be channel-wise: at least 1D array!')
        
        
        #This gets a bit fussy. To allow mergers of heterogenous tables down the line,
        #it is necessary to ensure that the column headers know absolutely NOTHING
        # about the number of channels. The list comprehension is necesary to pass
        #2D numpy arrays into the column.
        freq_column=at.Column(name='freq',dtype='object',shape=(),length=self.Nrows,
                            description='Channel Frequency',unit='Hz')
        freq_column[:]=[x for x in freq_2D]
    
        stokesI_column=at.Column(name='stokesI',dtype='object',shape=(),length=self.Nrows,
                            description='Stokes I per channel')
        stokesI_column[:]=[x for x in stokesI]

        stokesI_error_column=at.Column(name='stokesI_error',shape=(),length=self.Nrows,
                           dtype='object',description='stokesI error per channel')
        stokesI_error_column[:]=[x for x in stokesI_error]

        stokesQ_column=at.Column(name='stokesQ',dtype='object',shape=(),length=self.Nrows,
                            description='Stokes Q per channel')
        stokesQ_column[:]=[x for x in stokesQ]
        stokesQ_error_column=at.Column(name='stokesQ_error',shape=(),length=self.Nrows,
                           dtype='object',description='Stokes Q error per channel')
        stokesQ_error_column[:]=[x for x in stokesQ_error]

        stokesU_column=at.Column(name='stokesU',dtype='object',shape=(),length=self.Nrows,
                            description='Stokes U per channel')
        stokesU_column[:]=[x for x in stokesU]
        stokesU_error_column=at.Column(name='stokesU_error',shape=(),length=self.Nrows,
                           dtype='object',description='Stokes U error per channel')
        stokesU_error_column[:]=[x for x in stokesU_error]

        source_number_column=at.Column(data=source_number_array,name='source_number',
                       dtype='int',description='Source ID number in file',unit='')

        #Set the number of sources (based on unique entries in the source number column).
        self.Nsrc=len(np.unique(source_number_column))

        #Beam sizes array may be scalar, 1D, or 2D. To accomodate mixing different
        #data types together (when merging tables), as well as to be able to save
        #to FITS format properly, the columns need to be made 'empty' (otherwise)
        #astropy assigns a length), and each element needs to be an array or list
        # (otherwise pyFITS cries). This needs to be done for any column that
        #might be a mixture of single entries and arrays (mixed 1D and 2D).
        beam_maj_column=at.Column(length=self.Nrows,
               name='beam_maj',dtype='object',unit='deg',
               description='Beam major axis in deg')
        beam_maj_column[:]=[ [x] if np.array(x).ndim == 0 else x for x in _possible_scalar_to_1D(beam_maj,self.Nrows)] 

        
        beam_min_column=at.Column(length=self.Nrows,
               name='beam_min',dtype='object',unit='deg',
               description='Beam minor axis in deg')
        beam_min_column[:]=[ [x] if np.array(x).ndim == 0 else x for x in _possible_scalar_to_1D(beam_min,self.Nrows)] 
        
        beam_pa_column=at.Column(length=self.Nrows,
               name='beam_pa',dtype='object',unit='deg',
               description='Beam position angle in deg')
        beam_pa_column[:]=[ [x] if np.array(x).ndim == 0 else x for x in _possible_scalar_to_1D(beam_pa,self.Nrows)] 



        #Calculate Nchan and make into a column:
        Nchan_array=np.zeros(self.Nrows)
        for i in range(self.Nrows):
            Nchan_array[i]=len(freq_column[i])
        Nchan_column=at.Column(data=Nchan_array,name='Nchan',
                   dtype='int',description='Number of channels')

        #Assemble basic table:
        self.table=at.Table([source_number_column,ra_column,dec_column,
                             glon_column, glat_column,
                             freq_column,
                             stokesI_column,stokesI_error_column,
                             stokesQ_column,stokesQ_error_column,
                             stokesU_column,stokesU_error_column,
                             beam_maj_column,beam_min_column,beam_pa_column,
                             Nchan_column])
        
        
        #Now adding the optional columns:
        if stokesV is not None: #Check that both Stokes V and error are supplied?
            stokesV_column=at.Column(name='stokesV',dtype='object',shape=(),length=self.Nrows,
                            description='Stokes V per channel')
            stokesV_column[:]=[x for x in stokesV]
            self.table.add_column(stokesV_column)

            stokesV_error_column=at.Column(name='stokesV_error',shape=(),length=self.Nrows,
                           dtype='object',description='Stokes V error per channel')
            stokesV_error_column[:]=[x for x in stokesV_error]
            self.table.add_column(stokesV_error_column)
        
        if quality is not None: #Check quality is 2d?
            quality_column=at.Column(name='quality',dtype='object',shape=(),length=self.Nrows,
                            description='Quality flags per channel')
            quality_column[:]=[x for x in quality]
            self.table.add_column(quality_column)

        if quality_meanings is not None:
            quality_meanings_column=at.Column(
                        data=_possible_scalar_to_1D(quality_meanings,self.Nrows),
                        name='quality_meanings',dtype='str',
                        description='Description of quality flag meanings')
            self.table.add_column(quality_meanings_column)
            
        if ionosphere is not None:
            ionosphere_column=at.Column(
                        data=_possible_scalar_to_1D(ionosphere,self.Nrows),
                        name='ionosphere',dtype='str',
                        description='Ionospheric correction method')
            self.table.add_column(ionosphere_column)
        

        if cat_id is not None:
            source_name_column=at.Column(data=cat_id,name='cat_id',dtype='str',
                            description='Source name')
            self.table.add_column(source_name_column)

        if dataref is not None:
            dataref_column=at.Column(data=_possible_scalar_to_1D(dataref,self.Nrows),
                                         name='dataref',dtype='str',
                            description='Reference to data paper')
            self.table.add_column(dataref_column)
            
        if telescope is not None:
            telescope_column=at.Column(
                        data=_possible_scalar_to_1D(telescope,self.Nrows),
                        name='telescope',dtype='str',
                        description='Telescope')
            self.table.add_column(telescope_column)

        if epoch is not None:
            epoch_column=at.Column(
                        data=_possible_scalar_to_1D(epoch,self.Nrows),
                        name='epoch',dtype='float',
                        description='Observation Epoch (midpoint, MJD)',unit='d')
            self.table.add_column(epoch_column)
        
        if integration_time is not None:
            integration_time_column=at.Column(
                        data=_possible_scalar_to_1D(integration_time,self.Nrows),
                        name='integration_time',dtype='float',
                        description='Integration time (observation duration, s)',unit='s')
            self.table.add_column(integration_time_column)
            
        if interval is not None:
            interval_column=at.Column(
                        data=_possible_scalar_to_1D(interval,self.Nrows),
                        name='interval',dtype='float',
                        description='Interval of observation (days)',unit='d')
            self.table.add_column(interval_column)


        if leakage is not None: #Like the beam columns, could be scalar, 1D, or 2D.
        #Thus slightly complicated to deal with.
             leakage_column=at.Column(length=self.Nrows,
                        name='leakage',dtype='object',unit='',
                        description='Estimated leakage fraction')
             leakage_column[:]=[ [x] if np.array(x).ndim == 0 else x for x in _possible_scalar_to_1D(leakage,self.Nrows)] 
             self.table.add_column(leakage_column)


        if channel_width is not None:  #Like the beam columns, could be scalar, 1D, or 2D.
        #Thus slightly complicated to deal with.
             channel_width_column=at.Column(length=self.Nrows,
                        data=_possible_scalar_to_1D(channel_width,self.Nrows),
                        name='channel_width',dtype='object',unit='Hz',
                        description='Channel bandwidth [Hz]')
             channel_width_column[:]=[ [x] if np.array(x).ndim == 0 else x for x in _possible_scalar_to_1D(channel_width,self.Nrows)] 
             self.table.add_column(channel_width_column)


        if flux_type is not None:
            flux_type_column=at.Column(
                        data=_possible_scalar_to_1D(flux_type,self.Nrows),
                        name='flux_type',dtype='str',
                        description='Stokes extraction method')
            self.table.add_column(flux_type_column)


        if aperture is not None:
            aperture_column=at.Column(
                        data=_possible_scalar_to_1D(aperture,self.Nrows),
                        name='aperture',dtype='float',unit='deg',
                        description='Integration aperture (diameter, deg)')
            self.table.add_column(aperture_column)

        #Set list of column names.        
        self.columns=self.table.colnames
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
                Physical unit of the column quantity, if any (optional)
            """
        new_column=at.Column(
                        data=_possible_scalar_to_1D(values,self.Nrows),
                        name=name,unit=units,
                        description=description)
        self.table.add_column(new_column)
        self.columns=self.table.colnames



    def add_channel_column(self,values,name,description,units=''):
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
        """
        #Verify the size is correct:
        if len(values) != self.Nrows:
            raise Exception('New column does not have same number of rows as table!')
        for i in range(self.Nrows):
            if len(values[i]) != self['Nchan'][i]:
                raise Exception("New column doesn't have same number of channels as rest of table, on row {}.".format(i))
        
        new_column=at.Column(length=self.Nrows,
                        name=name,unit=units,dtype='object',
                        description=description)
        new_column[:]=[x for x in values]
        self.table.add_column(new_column)        
        self.columns=self.table.colnames


    def __repr__(self):
        #pass through directly to the table
        return self.table.__repr__()
    
    def _repr_html_(self):
        return self.table._repr_html_()


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
            polspec.columns=val.colnames
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
        #Converting to FITSrec format loses the column description...
        # Is that important?
        fits_columns=[]
        col_descriptions=[]
        
        #per column, convert to fits column:
        for col in self.table.colnames:
            tabcol=self.table[col]
            if tabcol.dtype != np.dtype('object'): #Normal columns
                col_format=pf.column._convert_record2fits(tabcol.dtype)
            else: #Channelized columns
                subtype=np.result_type(np.array(tabcol[0])) #get the type of each element in 2D array
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
        self.columns=self.table.colnames

        hdu.close()
        


    def write_VOTable(self,filename):
        """Write the polspectra to a VOTable (.xml) file. Note that this will
        automatically overwrite any existing file with the same name.
        Parameters:
            filename : str
            Name and relative path of the file to save to.
        
        """
        VOtable=vot.from_table(self.table)
        VOtable.description='PolSpectra'
        VOtable.coordinate_systems.append(vot.tree.CooSys(ID='equatorial_coordinates',system='ICRS',epoch='J2000.0'))
        tab=VOtable.get_first_table()
        VOtable.to_xml(filename)

    
    def read_VOTable(self,filename):
        """Read in a polarization spectrum table from a VOTable file.
        Parameters:
            filename: str
                Relative path and name of file to read from."""
        readin=vot.parse(filename)
        table=readin.get_first_table()
        self.table=table.to_table()
        self.Nrows = len(self.table)
        self.Nsrc = np.unique(self['source_number']).size
        self.columns=self.table.dtype.names


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
            source_numbers : str ('keep' or 'concat')
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
        self.columns=self.table.colnames
        

        
    
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
        new_copy.columns=new_copy.table.colnames
        return new_copy
    
    
    def verify_table(self):
        """Checks for most common possible problems that could occur when 
        generating a PolSpecta table:
            1. Inconsistent number of channels for columns in a single row.
            2. Unit-mismatches (frequencies not in Hz, beam size not in deg, etc)
        Can't check custom columns; only verifies standard columns.'
        """
        
        #Generate a list of of all columns that are channel-wise.
        channel_columns=['freq','stokesI','stokesI_error','stokesQ',
                         'stokesQ_error','stokesU','stokesU_error']
        #Add stokesV, error if present
        if 'stokesV' in self.table.colnames:
            channel_columns.append('stokesV')
            channel_columns.append('stokesV_error')
        
        badvalues=[]
        for i in range(self.Nrows):
            for column in channel_columns:
                if len(self.table[i][column]) != self.table[i]['Nchan']:
                    badvalues.append((column,i))

        #Beam, leakage, channel_width are problematic - could be single-valued or channel-wise
        #They are allowed to be either Nchan or 1.

        for i in range(self.Nrows):
            if len(self.table[i]['beam_maj']) not in [self.table[i]['Nchan'],1] :
                badvalues.append(('beam_maj',i))
            if len(self.table[i]['beam_min']) not in [self.table[i]['Nchan'],1] :
                badvalues.append(('beam_min',i))
            if len(self.table[i]['beam_pa']) not in [self.table[i]['Nchan'],1] :
                badvalues.append(('beam_pa',i))


        if ('leakage' in self.table.colnames):
            for i in range(self.Nrows):
                if len(self.table[i]['leakage']) not in [self.table[i]['Nchan'],1] :
                    badvalues.append(('leakage',i))


            pass
        if ('channel_width' in self.table.colnames) and (self.table['channel_width'].dtype.type==np.object_):
            for i in range(self.Nrows):
                if len(self.table[i]['channel_width']) not in [self.table[i]['Nchan'],1] :
                    badvalues.append(('channel_width',i))




        
        if len(badvalues) == 0:
            print('All columns have consistent numbers of channels.')
        else:
            print('Some columns have mis-matched numbers of channels.\n'
                  'Below are column-row pairs of mismatched entries:')
            print(badvalues)

        #Check for unit mismatches, by flagging values that are several
        #orders of magnitude larger/smaller than expected.
        
        #Frequency shouldn't be below 1 MHz, I expect, so check for small frequencies
        if np.min([np.min(x) for x in self.table['freq'] ] ) < 1e6:
            print('Some frequency values are much smaller than expected.'
                  ' Are they in the correct units (Hz)?')

        #Beam size shouldn't be more than a degree, generally
        if np.max([np.max(x) for x in self.table['beam_maj'] ] ) > 1:
            print('Some beam size values are much larger than expected.'
                  'Are they in the correct units (deg)?')
    
    
    
    def crossmatch_sources(self,radius,consecutive=True):
        """Groups together sources (by giving them the same source_number)
        based on position crossmatching, using a user-supplied crossmatch
        radius. Modifies the source_number column in-place.
        Note that it if the radius is large is possible to 'chain together' 
        sources such that each group member is close enough to at least one
        other, but not necessarily to all others.
        Inputs:
            radius (float): cross-matching radius in degrees. Sources closer than
                    this radius will be grouped.
            consecutive (bool): Renumber sources to be consecutive from 0 to 
                                N_src-1. If false, source numbers will be kept
                                as-is (after grouping).
        """
        #Create SkyCoord objects for all rows:
        positions=ac.SkyCoord(ra=self.table['ra'],dec=self.table['dec'],
                    frame='icrs',unit='deg')

        #Per row, find neighbours in later part of table.
        for i in range(self.Nrows-1):
            sep=positions[i+1:].separation(positions[i])
            w=np.where(sep.deg < radius)[0]
            if w.size > 0:
                self.table['source_number'][w+i+1]=self.table['source_number'][i]
                
        if consecutive == True:
            self.renumber_sources()
            
        self.Nsrc=len(np.unique(self.table['source_number']))
    
        

def from_FITS(filename):
    """Read in a polarization spectrum table from a FITS file.
    Parameters:
        filename: str
            Relative path and name of file to read from."""
    polspec=polarizationspectra()
    polspec.read_FITS(filename)
    return polspec

def from_VOTable(filename):
    """Read in a polarization spectrum table from a VOTable file.
    Parameters:
        filename: str
            Relative path and name of file to read from."""
    polspec=polarizationspectra()
    polspec.read_VOTable(filename)
    return polspec
    

def from_arrays(long_array,lat_array, freq_array, stokesI,stokesI_error,
                 stokesQ,stokesQ_error,stokesU,stokesU_error,source_number_array,
                 beam_maj,beam_min,beam_pa,coordinate_system='icrs',
                 stokesV=None,stokesV_error=None,quality=None,
                 quality_meanings=None,ionosphere=None,cat_id=None,dataref=None,
                 telescope=None,epoch=None,integration_time=None,interval=None,
                 leakage=None,channel_width=None,flux_type=None,aperture=None):
    
    new_spectra=polarizationspectra()
    new_spectra.create_from_arrays(long_array,lat_array, freq_array, stokesI,stokesI_error,
                 stokesQ,stokesQ_error,stokesU,stokesU_error,source_number_array,
                 beam_maj,beam_min,beam_pa,coordinate_system,
                 stokesV,stokesV_error,quality,
                 quality_meanings,ionosphere,cat_id,dataref,
                 telescope,epoch,integration_time,interval,
                 leakage,channel_width,flux_type,aperture)
    return new_spectra
from_arrays.__doc__=polarizationspectra.create_from_arrays.__doc__

def _possible_scalar_to_1D(scalar,Nrows):
    """If the input value is not indexable (and not a string), extend it to a list
    of Nrows values. If it is indexable, leave it as is (unless it's a string).
    Should also pass 2D arrays through without changes (may fail for 2D arrays
    of strings, have not tested).
    """
    try: #Test array vs scalar
        _=scalar[0]  #scalars can't be indexed, so this will fail for scalars.
        if type(scalar) == str:  #String are indexible scalars, so check first!
            column=[scalar for i in range(Nrows)]
        else:
            column=scalar
    except: #If it's not indexible, assume it's a scalar:
        column=[scalar for i in range(Nrows)]
    return column

    
def _get_dimensionality(column):
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
    




