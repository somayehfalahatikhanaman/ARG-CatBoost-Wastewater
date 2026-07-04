# Environmental variables
env_features = [
    'LATITUDE', 'LONGITUDE', 'T2M', 'T2M_MAX', 'T2M_MIN', 'TS',
    'RH2M', 'QV2M', 'T2MDEW', 'T2MWET', 'PRECTOTCORR',
    'GWETTOP', 'GWETROOT', 'GWETPROF', 'EVPTRNS',
    'WS2M', 'PS', 'ALLSKY_SFC_SW_DWN',
    'ALLSKY_SFC_LW_DWN', 'CLOUD_AMT',
    'PW', 'relative_week', 'weekofyear'
]

# ARG classes
gene_classes = [
    'BL', 'AMG', 'COL', 'TET', 'SUL',
    'FQ', 'GLY', 'FOS', 'PHEN',
    'OXA', 'TMP', 'MLS', 'NI',
    'RIF', 'MDR-EP'
]

print("Number of environmental features:", len(env_features))
print("Number of ARG classes:", len(gene_classes))
