#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QA tests for the polarizedspectra class.

These tests will produce a few small simulated data sets and try to create
polarizedspectra tables and manipulate them in typical ways.


"""

import unittest
import os
import shutil
import numpy as np
import astropy.table as at

def make_simdata_1():
    """Create test data arrays with 2 sources. Only basic columns, all as lists."""
    simsrc={}
    simsrc['source_number']=[0,1]
    simsrc['ra']=[15.75,133.3]
    simsrc['dec']=[+13.53,-13.333]
    simsrc['freq']=[[800e6,801e6],[800e6,803e6]]
    simsrc['StokesI']=[[1e-3,1.1e-3],[2e-3,2.1e-3]]
    simsrc['StokesI_error']=[[1e-5,1e-5],[1e-5,1.1e-5]]
    simsrc['StokesQ']=[[1e-4,1.1e-4],[2e-4,2.1e-4]]
    simsrc['StokesQ_error']=[[1e-5,1e-5],[1e-5,1.1e-5]]
    simsrc['StokesU']=[[1e-5,1.1e-5],[2e-5,2.1e-5]]
    simsrc['StokesU_error']=[[10e-5,1.1e-5],[1e-5,1.1e-5]]
    simsrc['unit']='Jy/beam'
    simsrc['beam_major']=[0.01,0.01]
    simsrc['beam_minor']=[0.01,0.01]
    simsrc['beam_pa']=[0,0]
    return simsrc

def make_simdata_2():
    """Make test data that has 2 sources with different channel lengths.
    Otherwise, everything kept to the basics."""
    simsrc={}
    simsrc['source_number']=[0,1]
    simsrc['ra']=[15.75,133.3]
    simsrc['dec']=[+13.53,-13.333]
    simsrc['freq']=[[800e6,801e6],[800e6,801e6,802e6]]
    simsrc['StokesI']=[[1e-3,1.1e-3],[2e-3,2.1e-3,2.2e-3]]
    simsrc['StokesI_error']=[[1e-5,1e-5],[1e-5,1.1e-5,1.2e-5]]
    simsrc['StokesQ']=[[1e-4,1.1e-4],[2e-4,2.1e-4,2.2e-4]]
    simsrc['StokesQ_error']=[[1e-5,1e-5],[1e-5,1.1e-5,1.2e-5]]
    simsrc['StokesU']=[[1e-5,1.1e-5],[2e-5,2.1e-5,2.2e-5]]
    simsrc['StokesU_error']=[[1e-5,1e-5],[1e-5,1.1e-5,1.2e-5]]
    simsrc['unit']='Jy/beam'
    simsrc['beam_major']=[0.01,0.01]
    simsrc['beam_minor']=[0.01,0.01]
    simsrc['beam_pa']=[0,0]
    return simsrc
    
def make_simdata_3():
    """Make test data that has only 1 source (scalars) and 1D arrays."""
    simsrc={}
    simsrc['source_number']=[0]
    simsrc['ra']=[15.75]
    simsrc['dec']=[13.53]
    simsrc['freq']=[[800e6,801e6]]
    simsrc['StokesI']=[[1e-3,1.1e-3]]
    simsrc['StokesI_error']=[[1e-5,1e-5]]
    simsrc['StokesQ']=[[1e-4,1.1e-4]]
    simsrc['StokesQ_error']=[[1e-5,1e-5]]
    simsrc['StokesU']=[[1e-5,1.1e-5]]
    simsrc['StokesU_error']=[[1e-5,1e-5]]
    simsrc['unit']='Jy/beam'
    simsrc['beam_major']=0.01
    simsrc['beam_minor']=0.01
    simsrc['beam_pa']=0
    return simsrc




class test_polspectra(unittest.TestCase):
    def test_01_create_from_arrays(self):
        source_arrays=make_simdata_1()
        pol_spec=polspectra.from_arrays(source_arrays['ra'],source_arrays['dec'],
                                source_arrays['freq'], source_arrays['StokesI'],
                                source_arrays['StokesI_error'],source_arrays['StokesQ'],
                                source_arrays['StokesQ_error'],
                                source_arrays['StokesU'],source_arrays['StokesU_error'],
                                source_arrays['source_number'],
                                source_arrays['unit'],source_arrays['beam_major'],
                                source_arrays['beam_minor'],source_arrays['beam_pa'])
        self.assertIsInstance(pol_spec,polspectra.polarizationspectra,'Failed to create 2-row simple polarizationspectra.')

    def test_02_creation_from_arrays_is_correct(self):
        source_arrays=make_simdata_1()
        pol_spec=polspectra.from_arrays(source_arrays['ra'],source_arrays['dec'],
                                source_arrays['freq'], source_arrays['StokesI'],
                                source_arrays['StokesI_error'],source_arrays['StokesQ'],
                                source_arrays['StokesQ_error'],
                                source_arrays['StokesU'],source_arrays['StokesU_error'],
                                source_arrays['source_number'],
                                source_arrays['unit'],source_arrays['beam_major'],
                                source_arrays['beam_minor'],source_arrays['beam_pa'])
        self.assertEqual(len(pol_spec),len(source_arrays['ra']),'Simple polarizationspectra contains wrong number of rows.')
        self.assertEqual(pol_spec.Nsrc,len(source_arrays['ra']),'Simple polarizationspectra contains wrong number of sources.')
        source_arrays['num_chan']=len(source_arrays['freq'][0])
        for colname in pol_spec.table.colnames:
            try: #For channelized columns, check row by row for equivalence.
                _=pol_spec[colname][0][0]
                for i in range(pol_spec.Nsrc):
                    self.assertTrue(np.array_equiv(source_arrays[colname][i],pol_spec[colname][i]),'{} Array not stored properly!'.format(colname))
            except: #For 1D columns, check entire column at once:
                self.assertTrue(np.array_equiv(source_arrays[colname],pol_spec[colname]),'{} Array not stored properly!'.format(colname))


    def test_03_creation_from_variable_length_arrays(self):
        #Check that table is created properly and can be accessed giving the correct results.
        source_arrays=make_simdata_2()
        pol_spec=polspectra.from_arrays(source_arrays['ra'],source_arrays['dec'],
                                source_arrays['freq'], source_arrays['StokesI'],
                                source_arrays['StokesI_error'],source_arrays['StokesQ'],
                                source_arrays['StokesQ_error'],
                                source_arrays['StokesU'],source_arrays['StokesU_error'],
                                source_arrays['source_number'],
                                source_arrays['unit'],source_arrays['beam_major'],
                                source_arrays['beam_minor'],source_arrays['beam_pa'])
        self.assertIsInstance(pol_spec,polspectra.polarizationspectra,'Failed to create 2-row different-channels polarizationspectra.')
        self.assertEqual(len(pol_spec),len(source_arrays['ra']),'Simple polarizationspectra contains wrong number of rows.')
        self.assertEqual(pol_spec.Nsrc,len(source_arrays['ra']),'Simple polarizationspectra contains wrong number of sources.')
        source_arrays['num_chan']=[len(source_arrays['freq'][i]) for i in range(len(source_arrays['ra']))]
        for colname in pol_spec.table.colnames:
            self.assertTrue(np.array_equiv(source_arrays[colname],pol_spec[colname]),'{} Array not stored properly!'.format(colname))
        
    
    
    def test_04_merging_basic_tables(self):
        #Test that tables can be merged, especially if the channel lengths don't match.
        source_arrays1=make_simdata_1()
        pol_spec1=polspectra.from_arrays(source_arrays1['ra'],source_arrays1['dec'],
                                source_arrays1['freq'], source_arrays1['StokesI'],
                                source_arrays1['StokesI_error'],source_arrays1['StokesQ'],
                                source_arrays1['StokesQ_error'],
                                source_arrays1['StokesU'],source_arrays1['StokesU_error'],
                                source_arrays1['source_number'],
                                source_arrays1['unit'],source_arrays1['beam_major'],
                                source_arrays1['beam_minor'],source_arrays1['beam_pa'])
        source_arrays2=make_simdata_2()
        pol_spec2=polspectra.from_arrays(source_arrays2['ra'],source_arrays2['dec'],
                                source_arrays2['freq'], source_arrays2['StokesI'],
                                source_arrays2['StokesI_error'],source_arrays2['StokesQ'],
                                source_arrays2['StokesQ_error'],
                                source_arrays2['StokesU'],source_arrays2['StokesU_error'],
                                source_arrays2['source_number'],
                                source_arrays2['unit'],source_arrays2['beam_major'],
                                source_arrays2['beam_minor'],source_arrays2['beam_pa'])
        #Create merged version with concatenated source lists 
        #(confirm number of sources matches expectations)
        pol_spec_merge=pol_spec1.copy()
        pol_spec_merge.merge_tables(pol_spec2,merge_type='exact',source_numbers='concat')
        self.assertIsInstance(pol_spec_merge,polspectra.polarizationspectra,'Failed to create simple merged polarizationspectra.')
        self.assertEqual(pol_spec_merge.Nrows,pol_spec1.Nrows+pol_spec2.Nrows,'Merged version does not have expected number of rows!')
        self.assertEqual(pol_spec_merge.Nsrc,pol_spec1.Nsrc+pol_spec2.Nsrc,'Merged (concat) version does not have expected number of rows!')
        #Create merged version with overlapping sources:
        pol_spec_merge=pol_spec1.copy()
        pol_spec_merge.merge_tables(pol_spec2,merge_type='exact',source_numbers='keep')
        self.assertIsInstance(pol_spec_merge,polspectra.polarizationspectra,'Failed to create simple merged polarizationspectra.')
        self.assertEqual(pol_spec_merge.Nrows,pol_spec1.Nrows+pol_spec2.Nrows,'Merged version does not have expected number of rows!')
        self.assertEqual(pol_spec_merge.Nsrc,pol_spec1.Nsrc,'Merged (keep) version does not have expected number of rows!')
            
    

    def test_05_readwrite_FITS(self):
        #Test that FITS file can be saved, read in again, and all information is preserved.
        source_arrays2=make_simdata_2()
        pol_spec2=polspectra.from_arrays(source_arrays2['ra'],source_arrays2['dec'],
                                source_arrays2['freq'], source_arrays2['StokesI'],
                                source_arrays2['StokesI_error'],source_arrays2['StokesQ'],
                                source_arrays2['StokesQ_error'],
                                source_arrays2['StokesU'],source_arrays2['StokesU_error'],
                                source_arrays2['source_number'],
                                source_arrays2['unit'],source_arrays2['beam_major'],
                                source_arrays2['beam_minor'],source_arrays2['beam_pa'])
        pol_spec2.write_FITS('./tests/simdata/test05.fits')
        self.assertTrue(os.path.exists('./tests/simdata/test05.fits'),'FITS file has not been saved!')
        readin_polspec=polspectra.from_FITS('./tests/simdata/test05.fits')
        self.assertEqual(readin_polspec.table.colnames,pol_spec2.table.colnames,'Table column names not preserved under FITS read/write.')
        self.assertEqual(readin_polspec.Nrows,pol_spec2.Nrows,'Number of rows not preserved under FITS read/write')
        self.assertEqual(readin_polspec.Nsrc,pol_spec2.Nsrc,'Number of sources not preserved under FITS read/write')
        self.assertTrue(np.array_equal(pol_spec2.table.as_array(),readin_polspec.table.as_array()),'Table contents are not preserved under FITS read/write')


    def test06_expand_scalars_to_columns(self):
        source_arrays2=make_simdata_2()
        source_arrays2['beam_pa']=30
        source_arrays2['beam_major']=0.2
        source_arrays2['beam_minor']=0.1
        pol_spec2=polspectra.from_arrays(source_arrays2['ra'],source_arrays2['dec'],
                                source_arrays2['freq'], source_arrays2['StokesI'],
                                source_arrays2['StokesI_error'],source_arrays2['StokesQ'],
                                source_arrays2['StokesQ_error'],
                                source_arrays2['StokesU'],source_arrays2['StokesU_error'],
                                source_arrays2['source_number'],
                                source_arrays2['unit'],source_arrays2['beam_major'],
                                source_arrays2['beam_minor'],source_arrays2['beam_pa'])
        self.assertIsInstance(pol_spec2,polspectra.polarizationspectra,'Failed to convert scalars to columns.')
        self.assertTrue(np.array_equal(pol_spec2['beam_pa'].data,np.repeat(source_arrays2['beam_pa'],pol_spec2.Nrows)),'Scalar was not accurately converted to column')

    
    def test_07_optional_columns(self):
        #Test that optional columns can be entered properly.
        source_arrays2=make_simdata_2()
        source_arrays2['StokesV']=[[1e-5,-1.1e-5],[2e-5,-2.1e-5,2.2e-5]]
        source_arrays2['StokesV_error']=[[1e-5,1e-5],[1e-5,2.1e-5,1.2e-5]]
        source_arrays2['quality']=[[0,0],[0,1,0]]
        source_arrays2['quality_meanings']='0=good, 1=RFI'
        source_arrays2['source_name']=['Source 1','Source 2']
        source_arrays2['coordinate_system']='FK5'
        source_arrays2['telescope']='Simulation'
        source_arrays2['epoch']=59003.45
        source_arrays2['integration_time']=1.0
        source_arrays2['channel_width']=1e6
        source_arrays2['aperture']=0.01
        pol_spec2=polspectra.from_arrays(source_arrays2['ra'],source_arrays2['dec'],
                                source_arrays2['freq'], source_arrays2['StokesI'],
                                source_arrays2['StokesI_error'],source_arrays2['StokesQ'],
                                source_arrays2['StokesQ_error'],
                                source_arrays2['StokesU'],source_arrays2['StokesU_error'],
                                source_arrays2['source_number'],
                                source_arrays2['unit'],source_arrays2['beam_major'],
                                source_arrays2['beam_minor'],source_arrays2['beam_pa'],
                                StokesV=source_arrays2['StokesV'],
                                StokesV_error=source_arrays2['StokesV_error'],
                                quality=source_arrays2['quality'],
                                quality_meanings=source_arrays2['quality_meanings'],
                                source_name=source_arrays2['source_name'],
                                coordinate_system=source_arrays2['coordinate_system'],
                                telescope=source_arrays2['telescope'],
                                epoch=source_arrays2['epoch'],
                                integration_time=source_arrays2['integration_time'],
                                channel_width=source_arrays2['channel_width'],
                                aperture=source_arrays2['aperture'])
        source_arrays2['num_chan']=[len(source_arrays2['freq'][i]) for i in range(pol_spec2.Nrows)]
        for colname in pol_spec2.table.colnames:
            try: #For channelized columns, check row by row for equivalence.
                _=pol_spec2[colname][0][0]
                for i in range(pol_spec2.Nsrc):
                    self.assertTrue(np.array_equiv(source_arrays2[colname][i],pol_spec2[colname][i]),'{} Array not stored properly!'.format(colname))
            except: #For 1D columns, check entire column at once:
                self.assertTrue(np.array_equiv(source_arrays2[colname],pol_spec2[colname]),'{} Array not stored properly!'.format(colname))

    
    
    def test_08_unspecified_columns(self):
        #Test that columns that aren't part of the standard can be created, and are 
        #handled properly by read/write operations.
        source_arrays2=make_simdata_2()
        pol_spec2=polspectra.from_arrays(source_arrays2['ra'],source_arrays2['dec'],
                                source_arrays2['freq'], source_arrays2['StokesI'],
                                source_arrays2['StokesI_error'],source_arrays2['StokesQ'],
                                source_arrays2['StokesQ_error'],
                                source_arrays2['StokesU'],source_arrays2['StokesU_error'],
                                source_arrays2['source_number'],
                                source_arrays2['unit'],source_arrays2['beam_major'],
                                source_arrays2['beam_minor'],source_arrays2['beam_pa'])
        scalar=7.0
        pol_spec2.add_column(values=scalar,name='scalar_column',
                             description='Test scalar column',units='m/s')
        self.assertTrue('scalar_column' in pol_spec2.table.colnames,'Custom scalar column not added to table!')
        self.assertEqual(pol_spec2['scalar_column'][0],scalar,'Custom scalar column values not propagated into table!')

        vector_col=[13,6]
        pol_spec2.add_column(values=vector_col,name='vector_column',
                             description='Test vector column',units='kg')
        self.assertTrue('vector_column' in pol_spec2.table.colnames,'Custom vector column not added to table!')
        self.assertTrue(np.array_equiv(pol_spec2['vector_column'],vector_col),'Custom vector column values not propagated into table!')

        #It should fail if given vector of the wrong length:
        vector_col=[13,6,16]
        with self.assertRaises(ValueError,msg='Adding custom column allowed column of wrong size!'):
            pol_spec2.add_column(values=vector_col,name='vector_column_wrongsize',
                             description='Test vector column 2',units='kg')

        #Add channel column with correct shape:
        channelized_column=[[0,1],[0,1,2]]
        pol_spec2.add_channel_column(channelized_column,name='Channel_column',
                                         description='Test channel-wise column',units='')
        self.assertTrue('Channel_column' in pol_spec2.table.colnames,'Custom channel-wise column not added to table!')
        self.assertTrue(np.array_equiv(pol_spec2['Channel_column'],channelized_column),'Custom channel-wise column values not propagated into table!')
        
        #Should fail if incorrect number of channels:
        channelized_column=[[0,1],[0,1]]
        with self.assertRaises(Exception,msg='Adding custom column allowed column of wrong size!'):
            pol_spec2.add_channel_column(channelized_column,name='Channel_column_wrong',
                                         description='Test channel-wise column',units='')
            
        #Should write and read custom columns successfully:
        pol_spec2.write_FITS('./tests/simdata/test08.fits')
        self.assertTrue(os.path.exists('./tests/simdata/test08.fits'),'Custom column FITS file has not been saved!')
        readin_polspec=polspectra.from_FITS('./tests/simdata/test08.fits')
        self.assertEqual(readin_polspec.table.colnames,pol_spec2.table.colnames,'Custom columns lost in write-read process')
        for colname in pol_spec2.table.colnames:
            try: #For channelized columns, check row by row for equivalence.
                _=pol_spec2[colname][0][0]
                for i in range(pol_spec2.Nsrc):
                    self.assertTrue(np.array_equiv(readin_polspec[colname][i],pol_spec2[colname][i]),'{} Array not stored properly!'.format(colname))
            except: #For 1D columns, check entire column at once:
                self.assertTrue(np.array_equiv(readin_polspec[colname],pol_spec2[colname]),'{} Array not stored properly!'.format(colname))


    
    def test_09_single_source_table(self):
        #Test that things still work when creating a polarizationspectrum with a single row.
        source_arrays1=make_simdata_3()
        pol_spec1=polspectra.from_arrays(source_arrays1['ra'],source_arrays1['dec'],
                                source_arrays1['freq'], source_arrays1['StokesI'],
                                source_arrays1['StokesI_error'],source_arrays1['StokesQ'],
                                source_arrays1['StokesQ_error'],
                                source_arrays1['StokesU'],source_arrays1['StokesU_error'],
                                source_arrays1['source_number'],
                                source_arrays1['unit'],source_arrays1['beam_major'],
                                source_arrays1['beam_minor'],source_arrays1['beam_pa'])
        self.assertIsInstance(pol_spec1,polspectra.polarizationspectra,'Failed to create 1-row simple polarizationspectra.')
        
        #Test that 1-row table can be properly combined with other tables:
        source_arrays2=make_simdata_2()
        pol_spec2=polspectra.from_arrays(source_arrays2['ra'],source_arrays2['dec'],
                                source_arrays2['freq'], source_arrays2['StokesI'],
                                source_arrays2['StokesI_error'],source_arrays2['StokesQ'],
                                source_arrays2['StokesQ_error'],
                                source_arrays2['StokesU'],source_arrays2['StokesU_error'],
                                source_arrays2['source_number'],
                                source_arrays2['unit'],source_arrays2['beam_major'],
                                source_arrays2['beam_minor'],source_arrays2['beam_pa'])
        pol_spec_merge=pol_spec1.copy()
        pol_spec_merge.merge_tables(pol_spec2,merge_type='exact',source_numbers='concat')
        self.assertIsInstance(pol_spec_merge,polspectra.polarizationspectra,'Failed to create simple merged polarizationspectra.')
        self.assertEqual(pol_spec_merge.Nrows,pol_spec1.Nrows+pol_spec2.Nrows,'Merged version does not have expected number of rows!')
        self.assertEqual(pol_spec_merge.Nsrc,pol_spec1.Nsrc+pol_spec2.Nsrc,'Merged (concat) version does not have expected number of rows!')

        
    
    def test_10_merge_different_length_string_columms(self):
        source_arrays1=make_simdata_1()
        source_arrays1['telescope']='Simulation'
        source_arrays1['source_name']=['Source 1','Source 2']
        pol_spec1=polspectra.from_arrays(source_arrays1['ra'],source_arrays1['dec'],
                                source_arrays1['freq'], source_arrays1['StokesI'],
                                source_arrays1['StokesI_error'],source_arrays1['StokesQ'],
                                source_arrays1['StokesQ_error'],
                                source_arrays1['StokesU'],source_arrays1['StokesU_error'],
                                source_arrays1['source_number'],
                                source_arrays1['unit'],source_arrays1['beam_major'],
                                source_arrays1['beam_minor'],source_arrays1['beam_pa'],
                                telescope=source_arrays1['telescope'],source_name=source_arrays1['source_name'])
        source_arrays2=make_simdata_1()
        source_arrays2['telescope']='Simulation long string'
        source_arrays2['source_name']=['Source 1 with long name','Source 2 with long name']
        pol_spec2=polspectra.from_arrays(source_arrays2['ra'],source_arrays2['dec'],
                                source_arrays2['freq'], source_arrays2['StokesI'],
                                source_arrays2['StokesI_error'],source_arrays2['StokesQ'],
                                source_arrays2['StokesQ_error'],
                                source_arrays2['StokesU'],source_arrays2['StokesU_error'],
                                source_arrays2['source_number'],
                                source_arrays2['unit'],source_arrays2['beam_major'],
                                source_arrays2['beam_minor'],source_arrays2['beam_pa'],
                                telescope=source_arrays2['telescope'],source_name=source_arrays2['source_name'])
        pol_spec_merge=pol_spec1.copy()
        pol_spec_merge.merge_tables(pol_spec2,merge_type='exact',source_numbers='concat')
        self.assertIsInstance(pol_spec_merge,polspectra.polarizationspectra,'Failed to create merged table with different string lengths.')
        self.assertEqual(pol_spec_merge.Nsrc,pol_spec1.Nsrc+pol_spec2.Nsrc,'Merged (concat) version does not have expected number of rows!')
        self.assertEqual(pol_spec_merge['telescope'][2],source_arrays2['telescope'],'Merged string column has incorrect entries!')
        self.assertEqual(pol_spec_merge['source_name'][2],source_arrays2['source_name'][0],'Merged string column has incorrect entries!')
        
        

    def test_11_extract_subset_and_reindex(self):
        #Does it support sub-selection of only some sources, then 
        source_arrays1=make_simdata_1()
        pol_spec1=polspectra.from_arrays(source_arrays1['ra'],source_arrays1['dec'],
                                source_arrays1['freq'], source_arrays1['StokesI'],
                                source_arrays1['StokesI_error'],source_arrays1['StokesQ'],
                                source_arrays1['StokesQ_error'],
                                source_arrays1['StokesU'],source_arrays1['StokesU_error'],
                                source_arrays1['source_number'],
                                source_arrays1['unit'],source_arrays1['beam_major'],
                                source_arrays1['beam_minor'],source_arrays1['beam_pa'])
        self.assertIsInstance(pol_spec1[0],polspectra.polarizationspectra,'Single-row selection does not return polarizationspectra object.')
        self.assertIsInstance(pol_spec1['num_chan'],at.column.Column,'Column access does not return astropy.table Column object!')
        
        #Testing source extraction and reindexing:
        source_arrays2=make_simdata_1()
        pol_spec2=polspectra.from_arrays(source_arrays2['ra'],source_arrays2['dec'],
                                source_arrays2['freq'], source_arrays2['StokesI'],
                                source_arrays2['StokesI_error'],source_arrays2['StokesQ'],
                                source_arrays2['StokesQ_error'],
                                source_arrays2['StokesU'],source_arrays2['StokesU_error'],
                                source_arrays2['source_number'],
                                source_arrays2['unit'],source_arrays2['beam_major'],
                                source_arrays2['beam_minor'],source_arrays2['beam_pa'])
        pol_spec_merge=pol_spec1.copy()
        pol_spec_merge.merge_tables(pol_spec2,merge_type='exact',source_numbers='concat')
        pol_spec_merge.merge_tables(pol_spec2,merge_type='exact',source_numbers='keep')
        source_zero=pol_spec_merge[np.where(pol_spec_merge['source_number'] == 0)]
        self.assertTrue(len(source_zero) == 2, 'Incorrectly extracted multi-row single source.')
        disordered_subselection=pol_spec_merge[[3,5,0]]
        disordered_subselection.renumber_sources()
        self.assertTrue(np.array_equiv(disordered_subselection['source_number'],range(len(disordered_subselection))),'Attempt at re-ordering source numbers has failed!')

    
if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.realpath(__file__)))
    os.chdir('../')
    import polspectra

    #Clean up in preparation for tests:    
    if os.path.exists('tests/simdata'):
        shutil.rmtree('tests/simdata')
    if not os.path.isdir('tests/simdata'):
        os.makedirs('tests/simdata')



    unittest.TestLoader.sortTestMethodsUsing=None
    suite = unittest.TestLoader().loadTestsFromTestCase(test_polspectra)
    unittest.TextTestRunner(verbosity=2).run(suite)
    
    
#import importlib
#importlib.reload(polspectra)