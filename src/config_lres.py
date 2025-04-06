import os
from grizli.utils import get_line_wavelengths
import routines_spec as rspec

BAD_VALUE = -999.0
gratings = rspec.GRATINGS_LRES

# label of the current sample (used in filenames and paths)
label_catalog = 'highz_n_emitters_prism'

# filenames
fname_sample = 'sample_spec.csv' # spec sample filename
fname_lines_sn = f'sample-{label_catalog}-nirspec_lines-sn.csv'
fname_lines_flux = 'catalog_line_fluxes.csv'
fname_abundances = 'catalog_abundances.csv'
fname_abundances_mean = 'catalog_abundances_mean.csv'

# filnames: data tables and catalogs
date_DJA_version = '2025_01_13'
fpath_data = os.path.join(os.getenv('data'), 'spectra/data/catalogs',
                          f'nirspec_tables-{date_DJA_version}')
fname_redshifts_manual = f'nirspec_redshifts_manual-{date_DJA_version}.fits'
fname_msaexp = f'nirspec_redshifts-{date_DJA_version}.fits'
fname_extractions = 'nirspec_extractions.fits'
fpath_redshifts_manual = os.path.join(fpath_data, fname_redshifts_manual)
fpath_msaexp = os.path.join(fpath_data, fname_msaexp)
fpath_extractions = os.path.join(fpath_data, fname_extractions)

# PyMC sampler
step_method = 'DEMZ'

# location of line fitting outputs
#fdir_outputs = f"pymc_outputs-step{step_method}
fdir_outputs = f"pymc_outputs-stepDEMZ-uncover_24175"


#=== dictionary of spectroscopic lines
""" Load spec line list """
lw, lr = get_line_wavelengths()
lw['OIII-1660'] = [1660.809]
lw['OIII-1665'] = [1665.85]
lw['NeIII-3343'] = [3343.4]
lw['OII-3727'] = [3727.092]
lw['OII-3729'] = [3729.875]
lw['NII-5756'] = [5756.2]

lw['HeI-4121'] = [4120.8154]
lw['HeI-4144'] = [4143.761]
lw['HeI-4169'] = [4168.967]
lw['HeI-4388']  = [4387.9296]
#lw['NeI-4412']  = [4412.285]
#lw['FeII-4413']  = [4413.600] # found in some objects
#lw['NeII-4413']  = [4413.113]
#lw['FeII-4205']  = [4205.5382] # found in some objects
lw['FeII-4205'] = [4452.098] # found in some objects

# Ne IV] 1601.5, 1601.7, 2421.8, 2424.4
# [S II] 4077.5, 4069.8
# [O II] 2471.0, 2471.1
# [N I] 5200.3, 5197.9, 3466.5
#mylw = {}
#mylw['NeIV'] = [1601.5, 2421.8]
#mylw['ArIV'] = [4740, 4711, 2689.0, 2854.5]
#mylw['OII'] = [2471.0]
#mylw['NI'] = [5200.3, 3466.5]
#mylw['NII'] = [3062.8]
#mylw['OIII'] = [2321.0]
#mylw['NeV'] = [3426, 1574.7]
#mylw['OI'] =  [6363.8, 2972.3]
#mylw['NeIII'] =  [3967.5, 1814.6]
#mylw['MgV'] =  [2782.7, 1324.6]
#mylw['SIII'] =  [3721.6]
#mylw['ArV'] =  [6133.8, 2691.1]
#mylw['ArIII'] =  [7135.8, 3005.2]

# lam_vaccuum = lam_air * n, n=1.000293 for air

lw['CII-4267'] = [4268.51]
lw['CIV-1549'] = [1548.20]
lw['CIV-1551'] = [1550.784]

lw['NeIV-1601'] = [1601.5]
lw['NeIV-2421'] = [2421.8]
lw['ArIV-4740'] = [4740.0]
lw['ArIV-4711'] = [4711.0]
lw['ArIV-2689'] = [2689.0]
lw['ArIV-2854'] = [2854.5]
lw['OII-2471'] = [2471.0]
lw['NI-5200'] = [5200.3]
lw['NI-3466'] = [3466.5]
lw['NII-3062'] = [3062.8]
lw['OIII-2321'] = [2321.0]
lw['NeV-3426'] = [3426.0]
lw['NeV-1574'] = [1574.7]
lw['OI-6363'] =  [6363.8]
lw['OI-2972'] =  [2972.3]
lw['NeIII-3967'] =  [3967.5]
lw['NeIII-1814'] =  [1814.6]
lw['MgV-2928'] = [2928.0]
lw['MgV-2782'] =  [2782.7]
lw['MgV-1324'] =  [1324.6]
lw['SIII-3721'] =  [3721.6]
lw['ArV-6133'] =  [6133.8]
lw['ArV-2691'] =  [2691.1]
lw['ArIII-7135'] =  [7135.8]
lw['ArIII-3005'] =  [3005.2]
#lw['NIV-1487']
lw['NIV-1483'] = [1483.321]
lw['NIV-1487'] = [1486.496]


lines_dict = {'Lya_1215':    [ lw['Lya'],       ],
              'HeII_1640':    [ lw['HeII-1640'], ],
              'OIII_1660':    [ lw['OIII-1660'], ],
              'OIII_1665':    [ lw['OIII-1665'], ],
              'NIII_1750':    [ lw['NIII-1750'], ],
              'NeIII_3343':   [ lw['NeIII-3343'] ],
              'OII_3727':     [ lw['OII-3727'],  ],
              'OII_3729':     [ lw['OII-3729'],  ],
              'NeIII_3867':   [ lw['NeIII-3867'] ],
              'SII_4070':     [ lw['SII-4070'],  ],
              'SII_4078':     [ lw['SII-4078'],  ],
              'OIII_4363':    [ lw['OIII-4363'], ],
              'HeII_4687':    [ lw['HeII-4687'], ],
              'OIII_4959':    [ lw['OIII-4959'], ],
              'OIII_5007':    [ lw['OIII-5007'], ],
              'NII_5756':     [ lw['NII-5756'],  ],
              'SIII_6314':    [ lw['SIII-6314'], ],
              'NII_6549':     [ lw['NII-6549'],  ],
              'NII_6584':     [ lw['NII-6584'],  ],
              'SII_6717':     [ lw['SII-6717'],  ],
              'SII_6731':     [ lw['SII-6731'],  ],
              'OII_7323':     [ lw['OII-7323'],  ],
              'OII_7332':     [ lw['OII-7332'],  ],
              'SIII_9531':    [ lw['SIII-9531'], ],  
              
              'Ha_6565':      [ lw['Ha'],        ],
              'Hb_4861':      [ lw['Hb'],        ],
              'Hg_4342':      [ lw['Hg'],        ],
              'Hd_4103':      [ lw['Hd'],        ],
              
              'CIV_1549':    [ lw['CIV-1549'],  ],
              'CIV_1551':    [ lw['CIV-1551'],  ],
              'CIII_1906':    [ lw['CIII-1906'],  ],
              'CIII_1908':    [ lw['CIII-1908'],  ],
              'CII_4267':    [ lw['CII-4267'],  ],
                          
              'NIV_1483':    [ lw['NIV-1483'], ],
              'NIV_1487':    [ lw['NIV-1487'], ],
              'NeIV_1602':   [ lw['NeIV-1601'], ],
              'NeIV_2422':   [ lw['NeIV-2421'], ],
              'ArIV_4740':   [ lw['ArIV-4740'], ],
              'ArIV_4711':   [ lw['ArIV-4711'], ],
              'ArIV_2689':   [ lw['ArIV-2689'], ],
              'ArIV_2854':   [ lw['ArIV-2854'], ],
              'OII_2471':    [ lw['OII-2471'], ],
              'NI_5200':     [ lw['NI-5200'], ],
              'NI_3466':     [ lw['NI-3466'], ],
              'NII_3063':    [ lw['NII-3062'], ],
              'OIII_2321':   [ lw['OIII-2321'], ],
              'NeV_3426':    [ lw['NeV-3426'], ],
              'NeV_1575':    [ lw['NeV-1574'], ],
              'OI_6364':     [ lw['OI-6363'], ],
              'OI_2972':     [ lw['OI-2972'], ],
              'NeIII_3967':  [ lw['NeIII-3967'], ],
              'NeIII_1815':  [ lw['NeIII-1814'], ],
              'MgV_2783':    [ lw['MgV-2782'], ],
              'MgV_1325':    [ lw['MgV-1324'], ],
              'SIII_3722':   [ lw['SIII-3721'], ],
              'ArV_6134':    [ lw['ArV-6133'], ],
              'ArV_2691':    [ lw['ArV-2691'], ],
              'ArIII_7136':  [ lw['ArIII-7135'], ],
              'ArIII_3005':  [ lw['ArIII-3005'], ],
              }

lw_exclude_sets = ['QSO-UV-lines', 'Gal-UV-lines', 'full', 'highO32', 'lowO32',
                  'Lya+CIV', 'Balmer 10kK', 'Balmer 10kK + MgII', 
                  'Balmer 10kK + MgII Av=0.5', 'Balmer 10kK + MgII Av=1.0', 
                  'Balmer 10kK + MgII Av=2.0', 'Balmer 10kK t0.5', 
                  'Balmer 10kK t1', 'OIII+Hb', 'OIII+Hb+Hg+Hd', 'OIII+Hb+Ha', 
                  'OIII+Hb+Ha+SII', 'HeI-series',
                  'QSO-Narrow-lines', 'Ha+SII', 'Ha+SII+SIII+He', 
                  'Ha+NII+SII+SIII+He', 'Ha+NII+SII+SIII+He+PaB', 
                  'Ha+NII+SII+SIII+He+PaB+PaG', 'Ha+NII', 'NII', 'SII',
                  'OIII+OII', 'OII+Ne', 'CIII-1906x', 'OIII', 
                  
                  ]

lw_exclude_lines = ['MgII',
                    'NeV-3346', 'NeVI-3426',
                    'HeI-3820', 'HeI-3889',
                    'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
                    #'SII-4075', 'SII-4070', 'SII-4078'
                    ]

lw_exclude = lw_exclude_sets + lw_exclude_lines
