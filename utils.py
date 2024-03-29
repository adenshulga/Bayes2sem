import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.stats import ks_2samp

import matplotlib.cm as cm

DATA_PATH = 'data/'

class TimeSeries:

    def __init__(self,  ts : np.ndarray, dataset_num: str, ts_num: int) -> None:

        self.ts = ts
        self.dim : int = ts.shape[1]
        self.len : int = len(ts)
        self.dataset_num = dataset_num
        self.ts_num = ts_num

    def __str__(self) -> str:
        
        string = f'''
        Dataser number: {self.dataset_num}
        TS number: {self.ts_num}
        TS len: {self.len}
        TS dim: {self.dim}
        '''
        
        return string
    
    def __repr__(self) -> str:
        return str(self)
    
    def __len__(self) -> int:
        return self.len
    
    def plot(self, borders : list[int] = None) -> None:

        if borders:
            begin = borders[0]
            end = borders[1]
            plt.plot(np.arange(begin, end, 1), self.ts[begin:end])
        else:
            plt.plot(np.arange(0, self.len, 1), self.ts)
        plt.show()

    def plot_distribution(self, 
                          changepoints: list[int] = None, 
                          bins: int = 100) -> None:
        if changepoints:
            color_map = cm.get_cmap('rainbow')
            ranges = [0] + changepoints + [self.len]
            p_matrix = np.zeros((len(ranges) - 1, len(ranges) - 1))

            plt.figure(figsize=(10, 5)) 

            for i in range(len(ranges) - 1):
                segment = self.ts[ranges[i]:ranges[i+1]]
                segment_color = color_map(i / (len(ranges) - 2))

                mean = np.mean(segment)
                variance = np.var(segment)

                plt.hist(segment,
                         bins=bins,
                         color=segment_color,
                         alpha=0.3,
                         label=f'Segment {i} - Mean: {mean:.2f}, Var: {variance:.2f}',
                         density=True)
                
                for j in range(i + 1, len(ranges) - 1):
                    other_segment = self.ts[ranges[j]:ranges[j+1]]
                    ks_statistic, p_value = ks_2samp(segment, other_segment)
                    p_matrix[i][j] = p_value
                    p_matrix[j][i] = p_value  # Symmetric matrix

            # Print the p-value matrix
            print("Matrix of p-values for the hypothesis that pairs of segments have the same distribution:")
            print(p_matrix)
                
        else:
            mean = np.mean(self.ts)
            variance = np.var(self.ts)

            plt.hist(self.ts,
                     bins=bins,
                     color='blue',
                     alpha=0.5,
                     label=f'Mean: {mean:.2f}, Var: {variance:.2f}',
                     density=True)

        plt.legend(loc='upper right')

        plt.show()



class Dataset:

    def __init__(self, num) -> None:
        
        self.num = num
        file_name = f'Learning_data_part{num}.xlsx'
        file_path = os.path.join(DATA_PATH, file_name)

        df = pd.read_excel(file_path)

        def extract_names(df: pd.DataFrame) -> dict[list]:
            columns = {}
            prev_series_number = -1
            for column in df.columns[1:]:
                parts = column.split('_')
                series_number = int(parts[1])
                coord_number = int(parts[3])
                if series_number == prev_series_number:
                    columns[series_number].append(column)
                else:
                    columns[series_number] = [column]
                prev_series_number = series_number

            return list(columns.values())
        

        names = extract_names(df)

        self.ts_list = []
        for i, name in enumerate(names):
            ts = df[name].to_numpy()
            ts = TimeSeries(ts, self.num, i)
            self.ts_list.append(ts)

    def __getitem__(self, i) -> TimeSeries:
        try:
            item = self.ts_list[i]
        except Exception as e:
            raise Exception(f'Choose value between 0 and {len(self.ts_list)}')
        return item

    def __iter__(self) -> 'Dataset':
        self.iter_index = 0
        return self

    def __next__(self) -> TimeSeries:
        if self.iter_index < len(self.ts_list):
            result = self.ts_list[self.iter_index]
            self.iter_index += 1
            return result
        else:
            raise StopIteration
        
    def sample_plots(self, 
                     ind : int = 0,
                     borders: list[int] = None,
                     changepoints: list[int] = None) -> None:
        
        self[ind].plot(borders=borders)
        self[ind].plot_distribution(changepoints=changepoints)


        

        









