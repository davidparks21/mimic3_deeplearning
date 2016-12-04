
# coding: utf-8

# In[1]:

class Feature(object):
    def __init__(self, path="~/mimic3/mimic3/demo/"):
        self.path = path
        
    def get_admission(self):
        import pandas as pd
        admission_df = pd.read_csv(self.path+'ADMISSIONS.csv', sep=',',header=0)
        return admission_df             .drop('ROW_ID', 1)             .drop('DISCHTIME', 1)             .drop('DEATHTIME', 1)             .drop('ADMISSION_TYPE', 1)             .drop('ADMISSION_LOCATION', 1)             .drop('DISCHARGE_LOCATION', 1)             .drop('INSURANCE', 1)             .drop('LANGUAGE', 1)             .drop('RELIGION', 1)             .drop('MARITAL_STATUS', 1)             .drop('ETHNICITY', 1)             .drop('EDREGTIME', 1)             .drop('EDOUTTIME', 1)             .drop('DIAGNOSIS', 1)             .drop('HOSPITAL_EXPIRE_FLAG', 1)             .drop('HAS_CHARTEVENTS_DATA', 1)
    
    def get_chartevents(self):
        import pandas as pd
        chartevents_df = pd.read_csv(self.path+'CHARTEVENTS.csv')
        return chartevents_df             .drop('ROW_ID', 1)             .drop('STORETIME', 1)             .drop('CGID', 1)             .drop('VALUE', 1)             .drop('VALUEUOM', 1)             .drop('WARNING', 1)             .drop('ERROR', 1)             .drop('RESULTSTATUS', 1)             .drop('STOPPED', 1)
    
    def get_items(self):
        HR = [220045 ,211] #[Heart Rate, Heart Rate]
        NIBP_S = [220179] #[Non Invasive Blood Pressure systolic]
        NIBP_D = [220180] #[Non Invasive Blood Pressure diastolic]
        CAP_Refill = [115, 3348, 8377, 223951, 224308] #[Capillary Refill [Right], Capillary Refill, Capillary Refill [Left], Capillary Refill R, Capillary Refill L]
        Glucose = [227015] #[Glucose_ApacheIV]
        pH = [227037] #[PH_ApacheIV]
        Temp = [227054] #[TemperatureF_ApacheIV]
        UrineScore = [227059] #[UrineScore_ApacheIV]
        Oxygen = [227035] #[OxygenScore_ApacheIV]
        RespiratoryRate = [224688] #[Respiratory Rate (Set)]
        GCS = [198] #[GCS Total]
        FiO2 = [1040] #[BIpap FIO2]
        ETCO2 = [1817] #[ETCO2]
#         items = [HR, NIBP_S, NIBP_D, CAP_Refill, Glucose, pH, Temp, UrineScore, Oxygen, RespiratoryRate, GCS, FiO2, ETCO2]
        return {
            'HR': HR,
            'NIBP_S': NIBP_S,
            'NIBP_D': NIBP_D,
            'CAP_Refill': CAP_Refill,
            'Glucose': Glucose,
            'pH': pH,
            'Temp': Temp,
            'UrineScore': UrineScore,
            'Oxygen': Oxygen,
            'RespiratoryRate': RespiratoryRate,
            'GCS': GCS,
            'FiO2': FiO2,
            'ETCO2': ETCO2
        }
    
    def get_relative_time(self, admit_time, chart_time):
        """
        admit_time, chart_time str
        :return value in seconds
        """
        from datetime import datetime
        # Accepts string in "2144-07-24 09:00:00" format and returns difference in seconds
        adm_time = datetime.strptime(admit_time, '%Y-%m-%d %H:%M:%S')
        char_time = datetime.strptime(chart_time, '%Y-%m-%d %H:%M:%S')
        del_time = char_time - adm_time
        rel_time = del_time.days*86400 + del_time.seconds
        return rel_time
    
    def get_relative_time_per_admid(self, df, admit_time):
        """
        :param df: chartevent df for particular HADM_ID
        :param admit_time: str
        :return tuple (new df with extra column, max time patient was monitored)
        """
        rlt = []
        max_time = 0
        for cTime in df['CHARTTIME']:
            temp = self.get_relative_time(admit_time, cTime)
            rlt.append(temp)
            if (temp > max_time):
                max_time = temp
        
        df['RELTIME'] = rlt
        return (df, max_time)
    
    def get_patient_event(self, df, item_ids, from_time_seconds, to_time_seconds, old_val):
        """
        :param df: chartevent df for particular HADM_ID
        :param item_ids: list of ids whose value between from_time_seconds and to_time_seconds is averaged and returned
        :param old_val: if patient not monitored during given time then use this value (typically previous value)
        """
        import math
        
        m = df[df.ITEMID.isin(item_ids)]
        l = m[(m['RELTIME'] >= from_time_seconds) & (m['RELTIME'] < to_time_seconds)]
        j = l['VALUENUM'].astype('float64').mean()
        
        if math.isnan(j): 
            return [old_val, 0]
        else:
            return [j, 1]
    
    def get_max_time(self, chartevents_df, admission_df):
        max_t = 0
        for idx, admission in admission_df.iterrows():
            (chartevents_perAdm_df, max_time) = self.get_relative_time_per_admid(
                chartevents_df[chartevents_df['HADM_ID'] == admission['HADM_ID']],
                admission['ADMITTIME'])# max_time is in seconds
            if(max_time > max_t):
                max_t = max_time
        return int(max_t/(60*10)) + 1
        
    
    def get_features(self):
        """TODO: convert list of list of list to numpy 3d matrix as admit*time*featurevec"""
        import numpy as np
        import math
         
        admission_df = self.get_admission()    
        chartevents_df = self.get_chartevents()
        
        features = np.zeros(shape=(admission_df.shape[0], 
                                  self.get_max_time(chartevents_df, admission_df),
                                  len(self.get_items())*2))
        z = np.zeros(shape=(admission_df.shape[0]))
        
        cnt2 = 0
        for idx, admission in admission_df.iterrows():
            print(str(admission['HADM_ID']) + ' started')
            import sys
            sys.stdout.flush()
            
            (chartevents_perAdm_df, max_time) = self.get_relative_time_per_admid(
                    chartevents_df[chartevents_df['HADM_ID'] == admission['HADM_ID']],
                    admission['ADMITTIME'])# max_time is in seconds
            
            z[cnt2] = math.ceil(max_time/(60*10))
            feature_patient = np.zeros(shape=(features.shape[1], features.shape[2]))
            prev_val = {}
            
            cnt = 0
            for it in range(0, max_time, 60*10):
                feature_patient_time = []
                items = self.get_items()
                for item_name in sorted(items.keys()):
                    prev_val[item_name] = self.get_patient_event(chartevents_perAdm_df, 
                                            items[item_name], it, it+(60*10), 
                                                       prev_val[item_name][0] if item_name in prev_val else 0)
                    feature_patient_time.extend(prev_val[item_name])
                feature_patient[cnt,:] = np.array(feature_patient_time)
                cnt+=1
            features[cnt2,:] = feature_patient
            cnt2+=1
            
        return features, z
    
    


# In[ ]:

# feature = Feature()


# In[109]:

# feature = Feature()
# admission_df = feature.get_admission()    
# chartevents_df = feature.get_chartevents()
# admission_df = admission_df[admission_df['HADM_ID'] == 142345]
# chartevents_df = chartevents_df[chartevents_df['HADM_ID'] == 142345].head(5)
# # chartevents_df
# (a, max_time) = feature.get_relative_time_per_admid(
#                     chartevents_df,
#                     '2164-10-23 21:09:00')# max_time is in seconds
# a = feature.get_patient_event(a, [599, 617], 96061, 96661, 4)


# In[110]:

# import unittest

# class TestFeature(unittest.TestCase):
#     def test_get_patient_event(self):
#         import pandas as pd
#         feature = Feature()
#         admission_df = feature.get_admission()    
#         chartevents_df = feature.get_chartevents()
#         admission_df = admission_df[admission_df['HADM_ID'] == 142345]
#         chartevents_df = chartevents_df[chartevents_df['HADM_ID'] == 142345]
        
#         feature.get_relative_time_per_admid(chartevents_df, '2164-10-23 21:09:00')
#         self.assertEqual(1, 1)

# suite = unittest.TestLoader().loadTestsFromTestCase(TestFeature)
# unittest.TextTestRunner(verbosity=2).run(suite)


# In[111]:

feature = Feature()
x, z = feature.get_features()
import numpy as np
np.save('x', x)
np.save('z', z)
print("done")

