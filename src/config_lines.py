import numpy as np

from grizli.utils import get_line_wavelengths


#=== dictionary of spectroscopic lines
""" Load spec line list """
lw, lr = get_line_wavelengths()
lw['OIII-1660'] = [1660.809]
lw['OIII-1665'] = [1665.85]
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

# lam_vaccuum = lam_air * n, n=1.000293 for air

lw['CII-4267'] = [4268.51]
lw['CIV-1549'] = [1548.20]
lw['CIV-1551'] = [1550.784]

lw['NeIV-1601'] = [1601.5]
lw['NeIV-2422'] = [2421.8]
lw['NeIV-2425'] = [2424.42]
lw['NeIV-4714'] = [4714.17]
lw['NeIV-4725'] = [4725.64]
lw['ArIV-4740'] = [4740.0]
lw['ArIV-4711'] = [4711.0]
lw['ArIV-2854'] = [2854.5]
lw['ArIV-2868'] = [2868.21976353]
lw['OII-2471'] = [2471.0]
lw['NI-5200'] = [5200.3]
lw['NI-3466'] = [3466.5]
lw['NII-3062'] = [3062.8]
lw['OIII-2321'] = [2321.0]
lw['NeV-3426'] = [3426.0]
lw['NeV-3346'] = [3345.83]
lw['NeV-1574'] = [1574.7]
lw['OI-6363'] = [6363.8]
lw['OI-2972'] = [2972.3]
lw['NeIII-1814'] = [1814.6]
lw['NeIII-3343'] = [3343.4]
lw['NeIII-3869'] = [3868.76]
lw['NeIII-3967'] = [3967.47]
lw['MgV-2928'] = [2928.0]
lw['MgV-2782'] =  [2782.7]
lw['MgV-1324'] =  [1324.6]
lw['SIII-3721'] =  [3721.6]
lw['SIII-9069'] =  [9068.60]
lw['ArV-6133'] =  [6133.8]
lw['ArV-2691'] =  [2691.1]
lw['ArIII-7751'] =  [7751.10]
lw['ArIII-7135'] =  [7135.8]
lw['ArIII-3005'] =  [3005.2]
lw['NIV-1483'] = [1483.321]
lw['NIV-1487'] = [1486.496]
lw['SiIII-1883'] = [1882.71]
lw['SiIII-1892'] = [1892.03]


lines_dict = {'Lya_1215':     [ lw['Lya'],       ],
              'HeII_1640':    [ lw['HeII-1640'], ],
              'OIII_1660':    [ lw['OIII-1660'], ],
              'OIII_1665':    [ lw['OIII-1665'], ],
              'NIII_1750':    [ lw['NIII-1750'], ],
              'NeIII_1815':   [ lw['NeIII-1814'],],
              'NeIII_3343':   [ lw['NeIII-3343'] ],
              'NeIII_3869':   [ lw['NeIII-3869'] ],
              'NeIII_3967':   [ lw['NeIII-3968'],],
              'OII_3727':     [ lw['OII-3727'],  ],
              'OII_3729':     [ lw['OII-3729'],  ],
              'SII_4070':     [ lw['SII-4070'],  ],
              'SII_4078':     [ lw['SII-4078'],  ],
              'OIII_4363':    [ lw['OIII-4363'], ],
              'HeII_4687':    [ lw['HeII-4687'], ],
              'OIII_4959':    [ lw['OIII-4959'], ],
              'OIII_5007':    [ lw['OIII-5007'], ],
              'NII_5756':     [ lw['NII-5756'],  ],
              'NII_6549':     [ lw['NII-6549'],  ],
              'NII_6584':     [ lw['NII-6584'],  ],
              'SII_6717':     [ lw['SII-6717'],  ],
              'SII_6731':     [ lw['SII-6731'],  ],
              'OII_7323':     [ lw['OII-7323'],  ],
              'OII_7332':     [ lw['OII-7332'],  ],
              'SIII_3722':    [ lw['SIII-3721'], ],
              'SIII_6312':    [ lw['SIII-6314'], ],
              'SIII_9069':    [ lw['SIII-9069'], ],  
              'SIII_9531':    [ lw['SIII-9531'], ],  
              'SiIII_1883':   [ lw['SiIII-1883'] ],
              'SiIII_1892':   [ lw['SiIII-1892'] ],
              
              'Ha_6565':      [ lw['Ha'],        ],
              'Hb_4861':      [ lw['Hb'],        ],
              'Hg_4342':      [ lw['Hg'],        ],
              'Hd_4103':      [ lw['Hd'],        ],
              
              'CIV_1549':     [ lw['CIV-1549'],  ],
              'CIV_1551':     [ lw['CIV-1551'],  ],
              'CIII_1906':    [ lw['CIII-1906'], ],
              'CIII_1908':    [ lw['CIII-1908'], ],
              'CII_4267':     [ lw['CII-4267'],  ],
                          
              'NIV_1483':     [ lw['NIV-1483'],  ],
              'NIV_1487':     [ lw['NIV-1487'],  ],
              'NeIV_1602':    [ lw['NeIV-1601'], ],
              'NeIV_2422':    [ lw['NeIV-2422'], ],
              'NeIV_2425':    [ lw['NeIV-2425'], ],
              'NeIV_4714':    [ lw['NeIV-4714'], ],
              'NeIV_4725':    [ lw['NeIV-4725'], ],
              'ArIV_4740':    [ lw['ArIV-4740'], ],
              'ArIV_4711':    [ lw['ArIV-4711'], ],
              'ArIV_2868':    [ lw['ArIV-2868'], ],
              'ArIV_2854':    [ lw['ArIV-2854'], ],
              'OII_2471':     [ lw['OII-2471'],  ],
              'NI_5200':      [ lw['NI-5200'],   ],
              'NI_3466':      [ lw['NI-3466'],   ],
              'NII_3063':     [ lw['NII-3062'],  ],
              'OIII_2321':    [ lw['OIII-2321'], ],
              'NeV_3426':     [ lw['NeV-3426'],  ],
              'NeV_3346':     [ lw['NeV-3346'],  ],
              'NeV_1575':     [ lw['NeV-1574'],  ],
              'OI_6364':      [ lw['OI-6363'],   ],
              'OI_2972':      [ lw['OI-2972'],   ],
              'MgV_2783':     [ lw['MgV-2782'],  ],
              'MgV_1325':     [ lw['MgV-1324'],  ],
              'ArV_6134':     [ lw['ArV-6133'],  ],
              'ArV_2691':     [ lw['ArV-2691'],  ],
              'ArIII_3005':   [ lw['ArIII-3005'],],
              'ArIII_7136':   [ lw['ArIII-7135'],],
              'ArIII_7751':   [ lw['ArIII-7751'],],
              }

lw_exclude_sets = [
    'QSO-UV-lines', 'Gal-UV-lines', 'full', 'highO32', 'lowO32',
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

lw_exclude_lines = [
    'MgII',
    'HeI-3820', 'HeI-3889',
    'H7', 'H8', 'H9', 'H10', 'H11', 'H12',
    #'SII-4075', 'SII-4070', 'SII-4078'
]

lw_exclude = lw_exclude_sets + lw_exclude_lines


#=== line sets by element or by application
# lines to fit
lines_neutral = [
    'OI_2972', 'OI_6364',
    'NI_3466', 'NI_5200',
    
    # recomb lines
    # OI 8446,8447
]

lines_MgS = [
    'MgV_2783', 'MgV_1325', 
    'SII_4070', 'SII_4078',
    'SIII_3722','SIII_6312','SIII_9069','SIII_9532',
]
lines_Ar = [
    'ArIII_3005','ArIII_7136','ArIII_7751',
    'ArIV_4740','ArIV_4711','ArIV_2868','ArIV_2854',
    'ArV_6134','ArV_2691'
]
lines_Ne = [
    'NeIII_1815','NeIII_3343','NeIII_3869','NeIII_3967',
    'NeIV_1602','NeIV_2422','NeIV_2425','NeIV_4714','NeIV_4725',
    'NeV_1575','NeV_3346','NeV_3426',
]
lines_cnohe = [
    'CIV_1549', 'CIV_1551', 'CIII_1906', 'CIII_1908', # 'CII_4267' (not in pn),
    'NIV_1483', 'NIV_1487', 
    'NIII_1750', 'NII_3063', 'NII_5756', 
    'NII_6549', 'NII_6584',
    'OIII_1660', 'OIII_1665', 
    'OIII_2321', 'OII_2471', 
    'OII_3727', 'OII_3729', 
    'OIII_4363', 'OIII_4959', 'OIII_5007', 
    'OII_7323', 'OII_7332',
    'HeII_1640', 'HeII_4687',
    
    # recomb lines
    # OII 4639,4642,4649
    # OIII 3265
    # OIV 4631
    # NII 4237,4242
    # NIII 4379
    # CII 4267
    # CIII 4647
    # CIV 4657
]

diag_lines = {
    'Te_diag': [
        ['OIII_4363', 'OIII_4959', 'OIII_5007',],
        ['NII_5756', 'NII_6549', 'NII_6584',],
        ['NeV_1575', 'NeV_3426',],
    ],
    'ne_diag': [
        ['CIII_1906', 'CIII_1908',],
        ['OII_3727', 'OII_3729',],
        ['SII_4070', 'SII_4078',],
        ['ArIV_4711', 'ArIV_4740',],
        ['SII_6717', 'SII_6731',],
        #['SiIII_1883', 'SiIII_1892',],
    ],
}

cols_tem_diag = [lines for lset in diag_lines['Te_diag'] for lines in lset]
cols_den_diag = [lines for lset in diag_lines['ne_diag'] for lines in lset]

cols_hydrogen = [
    'Hd_4103', 'Hg_4342', 'Hb_4861', 'Ha_6565'
]

cols_doublets = [
    'NIV_1483', 'NIV_1487',
    'CIV_1549', 'CIV_1551',
    'OIII_1660', 'OIII_1665',
    'CIII_1906', 'CIII_1908',
    'OII_3727', 'OII_3729',
    'SII_4070', 'SII_4078',
    'ArIV_4711', 'ArIV_4740',
    'SII_6717', 'SII_6731',
    'OII_7323', 'OII_7332',
]

#line_keys = cols_tem_diag + cols_den_diag + cols_hydrogen
line_keys = lines_MgS + lines_Ar + lines_Ne +\
            lines_cnohe + cols_tem_diag + cols_den_diag + cols_hydrogen
line_keys = np.unique(line_keys)

#===============================================================================
#=== for interfacing with PyNeb ================================================
#===============================================================================
# translate my line labels to PyNeb convension
translator_my2pn = {
    'GroupID': 'NAME',
    'HeII_1640_flux': 'He2r_1640A',
    'HeII_4687_flux': 'He2r_4686A',
    'NI_3466_flux': 'N1_3466A',
    'NI_5200_flux': 'N1_5200A',
    'NII_3063_flux': 'N2_3063A',
    'NII_6549_flux': 'N2_6548A',
    'NII_6584_flux': 'N2_6584A',
    'NIII_1750_flux': 'N3_1750A',
    'NIV_1483_flux': 'N4_1483A',
    'NIV_1487_flux': 'N4_1487A',
    'OI_2972_flux': 'O1_2973A',
    'OI_6364_flux': 'O1_6364A',
    'OII_2471_flux': 'O2_2470A',
    'OII_3727_flux': 'O2_3726A',
    'OII_3729_flux': 'O2_3729A',
    'OII_7323_flux': 'O2_7319A',
    'OII_7332_flux': 'O2_7330A',
    'OIII_1660_flux': 'O3_1661A',
    'OIII_1665_flux': 'O3_1666A',
    'OIII_2321_flux': 'O3_2321A',
    'OIII_4363_flux': 'O3_4363A',
    'OIII_4959_flux': 'O3_4959A',
    'OIII_5007_flux': 'O3_5007A',
    #'OIII_5007+_flux': 'O3_5007A+',
    'Ha_6565_flux': 'H1r_6563A',
    'Hb_4861_flux': 'H1r_4861A',
    'Hg_4861_flux': 'H1r_4861A',
    'Hd_4861_flux': 'H1r_4861A',
    'Hg_4342_flux': 'H1r_4341A',
    'Hd_4103_flux': 'H1r_4102A',
    'NeIII_1815_flux': 'Ne3_1815A',
    'NeIII_3343_flux': 'Ne3_3343A',
    'NeIII_3867_flux': 'Ne3_3869A',
    'NeIII_3967_flux': 'Ne3_3968A',
    'SII_4070_flux': 'S2_4069A',
    'SII_4078_flux': 'S2_4076A',
    'NII_5756_flux': 'N2_5755A',
    'SIII_3722_flux': 'S3_3722A',
    'SIII_6312_flux': 'S3_6312A',
    'SIII_9069_flux': 'S3_9069A',
    'SIII_9531_flux': 'S3_9531A',
    'SII_6717_flux': 'S2_6716A',
    'SII_6731_flux': 'S2_6731A',
    'ArIII_7751_flux': 'Ar3_7751A',
    'ArIII_7136_flux': 'Ar3_7136A',
    'ArIII_3005_flux': 'Ar3_3005A',
    'ArIV_4740_flux': 'Ar4_4740A',
    'ArIV_4711_flux': 'Ar4_4711A',
    'ArIV_2868_flux': 'Ar4_2868A',
    'ArIV_2854_flux': 'Ar4_2854A',
    'ArV_6134_flux': 'Ar5_6133A',
    'ArV_2691_flux': 'Ar5_2692A',
    'NeIV_1602_flux': 'Ne4_1602A',
    'NeIV_2422_flux': 'Ne4_2422A',
    'NeIV_2425_flux': 'Ne4_2425A',
    'NeIV_4714_flux': 'Ne4_4714A',
    'NeIV_4725_flux': 'Ne4_4724A',
    'NeV_1575_flux': 'Ne5_1575A',
    'NeV_3426_flux': 'Ne5_3426A',
    'NeV_3346_flux': 'Ne5_3346A',
    'MgV_2783_flux': 'Mg5_2783A', 
    'MgV_1325_flux': 'Mg5_1325A',
    'CIV_1549_flux': 'C4_1548A',
    'CIV_1551_flux': 'C4_1551A', 
    'CIII_1908_flux': 'C3_1909A',
    'CIII_1906_flux': 'C3_1907A',
    'SiIII_1883_flux': 'Si3_1883A',
    'SiIII_1892_flux': 'Si3_1892A',

}

translator_pn2my = dict(zip(translator_my2pn.values(), translator_my2pn.keys()))

#=== Te,ne diagnostics
# Te temperature diagnostics and assumed typical density
# see e.g. # Kewley+2019 10.3847/1538-4357/ab16ed
temp_diag = {
    '[SII] 4072+/6720+': 1e2,
    '[NII] 5755/6584': 400,
    #'[NII] 5755/6584+': 400,
    '[OIII] 1666/4363': 1e3,
    '[OIII] 1666/5007': 1e3,
    #'[OIII] 1666/5007+': 1e3,
    #'[OIII] 4363/5007': 1e3,
    '[OIII] 4363/5007+': 1e3,
    #'[SIII] 9069/6312': 1e4,
    #'[SIII] 6312/9531+': 1e4,
    '[NeV] 1575/3426': 1e4,
    '[ArIV] 2860+/4720+': 1e4,
}

# ne density diagnostics and assumed typical temperature
# see e.g. # Kewley+2019 10.3847/1538-4357/ab16ed
den_diag = {
    '[OII] 3726/3729': 1e4,
    '[SII] 4069/4076': 1e4,
    '[SII] 6731/6716': 1e4,
    '[CIII] 1909/1907': 2e4, 
    '[ArIV] 4740/4711': 2e4,
    ##'[ClIII] 5538/5518': 1e4,
    ##'[OIV] 1405/1401': 1e4,
}

# ionization zones (ref. Berg+21)
zone_ions = {
    
    'low': 
        ['OII', 'NII', 'SII', 'FeII', 'MgII', 'SiII',],
    
    'intermediate':
        ['HeII', 'CIII', 'NIII', 'OIII', 'SiIII', 'SiIV', 
         'SIII', 'SIV', 'ClIII', 'ArIII', 'FeIII',],
    
    'high':
        ['CIV', 'NIV', 'NeIII', 'ArIV',],

    'very_high':
        ['HeIII', 'OIV', 'NeV',],

}

ion_abundances = {
    'Ar3': [['Ar3_7136A','Ar3_7751A']],
    'Ar4': [['Ar4_4711A','Ar4_4740A'], ['Ar4_2854A','Ar4_2868A']],
    #'Ar5': [['Ar5_6133A']],
    'C3':  [['C3_1907A','C3_1909A']],
    'C4':  [['C4_1548A','C4_1551A']],
    'O2':  [['O2_3726A','O2_3729A'], ['O2_7319A','O2_7330A']], # OK
    'O3':  [['O3_1661A','O3_1666A'], ['O3_4959A','O3_5007A']], # OK
    #'Mg5': [['Mg5_2783A','Mg5_2929A']],
    'N2':  [['N2_6548A','N2_6584A']], # OK
    'N3':  [['N3_1750A']],
    'N4':  [['N4_1483A', 'N4_1487A']],
    'Ne3': [['Ne3_3869A','Ne3_3968A']],
    'Ne4': [['Ne4_2422A','Ne4_2425A'], ['Ne4_4714A','Ne4_4724A']],
    'Ne5': [['Ne5_3426A','Ne5_3346A']],
    'S2':  [['S2_4069A','S2_4076A'], ['S2_6716A','S2_6731A']],
    'S3':  [['S3_9069A','S3_9531A']],
    'Si3': [['Si3_1883A','Si3_1892A']],
    
}

elem_abundances = {
    'Ar/Hb': ['Ar3_7751A/Hb','Ar4_4740A/Hb'],
    'C/Hb':  ['C3_1909A/Hb','C4_1551A/Hb'],
    'N/Hb':  ['N2_6584A/Hb','N3_1750A/Hb','N4_1487A/Hb'],
    'O/Hb':  ['O2_3729A/Hb','O3_5007A/Hb'],
    'S/Hb':  ['S2_6731A/Hb','S3_9531A/Hb'],
    'Ne/Hb': ['Ne3_3968A/Hb','Ne4_2425A/Hb','Ne5_3346A/Hb'],
}

# Create a reverse mapping for ion-dependent switch
zone_switch = {ion: zone for zone, ions in zone_ions.items() for ion in ions}
