import numpy as np
from scipy.stats import rv_continuous
from typing import Callable
from utils import TimeSeries



'''
Есть prior на среднее и дисперсию
Есть prior на длину интервала

Я генерирую длины отрезков. Для каждого отрезка генерирую параметры 


я передаю prior на длины отрезков, и prior на параметры распределения
'''

class SyntheticTimeSeries(TimeSeries):

    def __init__(self, ts: np.ndarray, dataset_num: str, ts_num: int, true_cps : list[int]) -> None:
        super().__init__(ts, dataset_num, ts_num)
        self.true_cps = true_cps

    def __str__(self) -> str:
        string =  super().__str__()
        return string + f'True cps: {self.true_cps}' 


class SyntheticChangepointData:
    def __init__(self, 
                 changepoint_prior: rv_continuous, 
                 params_prior: list[rv_continuous],
                 ts_gen_distribution : Callable,
                 timespan: int = 10000,
                 ) -> None:
        self.timespan = timespan
        self.changepoint_prior = changepoint_prior
        self.params_prior = params_prior
        self.ts_gen_distribution = ts_gen_distribution

    def generate_segments(self):
        
        segments = [0]

        while True:
            # print(self.timespan)
            if segments[-1] < self.timespan:
                next_changepoint = self.changepoint_prior.rvs() + segments[-1]
                
                if next_changepoint < self.timespan:
                    segments.append(next_changepoint)
                else:
                    segments.append(self.timespan)
                    break
            else:
                break

        segments = [int(cp) for cp in segments]
        return segments

    def generate_parameters(self, num_segments: int):
        params_list = [[dist.rvs() for dist in self.params_prior] for _ in range(num_segments)]
        return params_list
    
    def generate_data(self, dataset_num: str = 'Synthetic', ts_num: int = None):
        
        segments = self.generate_segments()
        print('Generated segments :', segments)
        true_cps = segments[1:-1]
        time_series_data = np.empty((0,1))

        segment_parameters = self.generate_parameters(len(segments))

        for i, (start, end) in enumerate(zip(segments[:-1], segments[1:])):
            segment_length = end - start
            params = segment_parameters[i]
            segment_data = self.ts_gen_distribution(*params, size=segment_length)
            
            time_series_data = np.concatenate((time_series_data, segment_data))
            # print(time_series_data.shape)

        return SyntheticTimeSeries(time_series_data, dataset_num, ts_num, true_cps=true_cps)


















