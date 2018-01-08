import pandas as pd
import numpy as np
import datetime

pse_src = 'data/pse_data.csv'
wb_src = 'data/wb_data.csv'
label_src = 'data/output_data.csv'

class sampleGenerator():
    def __init__(self, pse_src=pse_src, wb_src=wb_src, label_src=label_src):
        pse = pd.read_csv(pse_src, header=0, index_col=0, parse_dates=True)
        label = pd.read_csv(label_src, header=0, index_col=0, parse_dates=True)
        wb = pd.read_csv(wb_src, header=0, index_col=0, parse_dates=True)
        wb = wb.reindex(pse.index, method='pad')

        self.pse = pse
        self.label = label
        self.wb = wb
        self.ts_index = pse.index

    def retrieve_sample(self, num=1, num_period=90, ordered=False, index=None):
        pse = self.pse
        wb = self.wb
        label = self.label
        if index is None:
            index = self.ts_index
        ts_index = index

        for i in range(num):
            if len(ts_index) <= num_period:
                self.ts_index = pse.index
                ts_index = index

            if ordered:
                sample_ix = num_period
            else:
                sample_ix = np.random.randint(num_period, len(ts_index))

            end = pse.index.get_loc(ts_index[sample_ix])
            beg = end - num_period
            ix = pse.index.tolist()[beg:end]

            pse_sample = np.array([pse.loc[ix].as_matrix()])
            wb_sample = np.array([wb.loc[ix].as_matrix()])
            label_sample = np.array([label.loc[ix].as_matrix()])

            if i == 0:
                pse_samples = pse_sample
                wb_samples = wb_sample
                label_samples = label_sample
            else:
                pse_samples = np.append(pse_samples, pse_sample, axis=0)
                wb_samples = np.append(wb_samples, wb_sample, axis=0)
                label_samples = np.append(label_samples, label_sample, axis=0)

            ts_index = ts_index.drop(ts_index[sample_ix])

        self.ts_index = ts_index

        return pse_samples, wb_samples, label_samples

    def generate_batch(self, batch_sz=5, num_period=90):
        pse = self.pse
        wb = self.wb
        label = self.label
        ts_index = np.array(pse.index.tolist())

        trim = len(ts_index) % batch_sz
        if trim != 0:
            ts_index = ts_index[0:-trim]

        ts_index = ts_index.reshape([batch_sz, -1])

        for i in range(ts_index.shape[1] // num_period):
            for j in range(batch_sz):
                ix = ts_index[j,(i*num_period):(i+1)*num_period]
                pse_sample = np.array([pse.loc[ix].as_matrix()])
                wb_sample = np.array([wb.loc[ix].as_matrix()])
                label_sample = np.array([label.loc[ix].as_matrix()])
                
                if j == 0:
                    pse_samples = pse_sample
                    wb_samples = wb_sample
                    label_samples = label_sample
                else:
                    pse_samples = np.append(pse_samples, pse_sample, axis=0)
                    wb_samples = np.append(wb_samples, wb_sample, axis=0)
                    label_samples = np.append(label_samples, label_sample, axis=0)

            yield pse_samples, wb_samples, label_samples

