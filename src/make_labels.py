from __future__ import print_function
import os

import numpy as np
import pandas as pd

def _overwrite_line(msg):
    print(msg, end='\r')



# should be the directory of ADMISSIONS.csv
data_dir = '~/mimic3/mimic3/demo/'

print('Getting data...')

data = pd.read_csv(os.path.join(data_dir, 'ADMISSIONS.csv'))

# convert relevant date columns into datetimes
for col in ['DISCHTIME','DEATHTIME']:
    data[col] = pd.to_datetime(data[col])
    
# when a person is admitted more then once and dies, the deathtime is only
# reflected in the last admittance, so get all admittances per patient and
# generate the labels for each admittance wrt to the deathtime for that patient
print('Creating labels...')
#data_grp_sub_id = data.groupby(['SUBJECT_ID'])

cols = ['SUBJECT_ID','HADM_ID','IN_HOSPITAL','30_DAYS','1_YEAR']
labels = pd.DataFrame(index=np.arange(0,len(data)), columns=cols)

idx = 0
for subject_id, row in data.iterrows():
    #_overwrite_line('{}% complete...'.format(round(idx/float(len(data)),4)*100))

    # if this patient died get the time and set the labels
    #print (row['DEATHTIME'])
    if (str(row['DEATHTIME']) != "NaT"):
        deathtime = pd.Timestamp(row['DEATHTIME'])
        
        #for row in rows.iterrows():
        r = row
        dischtime = r['DISCHTIME']
        in_hospital = deathtime <= dischtime
        days30 = (~in_hospital & (deathtime < (dischtime + np.timedelta64(30,'D')))) | in_hospital
        years1 = (~in_hospital & ~days30 & (deathtime < (dischtime + np.timedelta64(365,'D')))) | days30 | in_hospital

        labels.iloc[idx] = [row['SUBJECT_ID'], r['HADM_ID'], int(in_hospital), int(days30), int(years1)]
        idx += 1
        
    # they lived! all zeros
    else:
#        for row in rows.iterrows():
        r = row
        labels.iloc[idx] = [row['SUBJECT_ID'], r['HADM_ID'], 0, 0, 0]
        idx += 1

if len(labels) != len(data):
    print('\nWARNING: We may have lost some data. data:{} labels:{}'.format(len(data),len(labels)) )

#labels.to_csv(os.path.join(data_dir,'LABELS.csv'))
np.save('../data/labels.npy', labels.as_matrix()) 
print('\nDone!')
