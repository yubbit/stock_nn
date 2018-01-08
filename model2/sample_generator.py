import pandas as pd
import numpy as np
import datetime

pse_data_loc = 'data/pse_data.csv'
wb_data_loc = 'data/wb_data.csv'
labels_loc = 'data/output_data.csv'

class sampleGenerator:
    def __init__(self, pse_data_loc, wb_data_loc, labels_loc):
        self.pse_data = pd.read_csv(pse_data_loc, header=0, index_col=0, parse_dates=True)
        self.wb_data = pd.read_csv(wb_data_loc, header=0, index_col=0, parse_dates=True)
        self.labels = pd.read_csv(labels_loc, header=0, index_col=0, parse_dates=True)
        self.ts_index = self.pse_data.index
        self.index_sz = len(self.ts_index)

    def retrieve_sample(self, num=1):
        pse_data = self.pse_data
        wb_data = self.wb_data
        labels = self.labels
        ts_index = self.ts_index

        for i in range(num):
            if len(ts_index) <= 90:
                self.ts_index = pse_data.index
                ts_index = self.ts_index

            sample_index = np.random.randint(90, len(ts_index))
            end_date = ts_index[sample_index]
            beg_date = ts_index[sample_index-90]
            wb_date = datetime.datetime(ts_index[sample_index].year, 1, 1)
            wb_index = wb_data.index.get_loc(wb_date) + 1

            pse_sample = np.array([pse_data[beg_date:end_date].as_matrix()])
            labels_sample = np.array([labels[end_date].as_matrix()])
            wb_sample = np.array([wb_data.iloc[wb_index-5:wb_index].as_matrix()])

            if i == 0:
                pse_samples = pse_sample
                labels_samples = labels_sample
                wb_samples = wb_sample
            else:
                pse_samples = np.append(pse_samples, pse_sample, axis=0)
                wb_samples = np.append(wb_samples, wb_sample, axis=0)
                labels_samples = np.append(labels_samples, labels_sample, axis=0)
    
            self.ts_index = ts_index.drop(ts_index[sample_index])

        return pse_samples, wb_samples, labels_samples

