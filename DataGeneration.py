import numpy as np
from scipy.stats import expon, norm, rv_continuous
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
        # Initialize the list of segment boundaries starting with 0
        segments = [0]

        # Keep sampling changepoints and growing the list
        while True:
            if segments[-1] < self.timespan:
                # Generate the next changepoint
                next_changepoint = self.changepoint_prior.rvs() + segments[-1]
                
                if next_changepoint < self.timespan:
                    segments.append(next_changepoint)
                else:
                    # If the next changepoint is beyond the timespan, end the generation process
                    segments.append(self.timespan)
                    break
            else:
                break

        # Making sure the segments are integers
        segments = [int(cp) for cp in segments]
        return segments

    def generate_parameters(self, num_segments: int):
        # Sample parameters using params_prior for the given number of segments
        param_sets = [dist.rvs(size=num_segments) for dist in self.params_prior]
        return list(zip(*param_sets))  # Transpose the list to get a list of parameter sets

    def generate_data(self, dataset_num: str = 'Synthetic', ts_num: int = None):
        # Generate the time series based on the generated segments and parameters
        segments = self.generate_segments()
        true_cps = [segment[-1] for segment in segments]
        time_series_data = np.array([])

        # Generate parameters for each segment
        segment_parameters = self.generate_parameters(len(segments) - 1)

        for i, (start, end) in enumerate(zip(segments[:-1], segments[1:])):
            segment_length = end - start
            params = segment_parameters[i]
            # Generate the data for the current segment using the sampled parameters
            segment_data = self.ts_gen_distribution(*params, size=segment_length)
            time_series_data = np.concatenate((time_series_data, segment_data))

        return SyntheticTimeSeries(time_series_data, dataset_num, ts_num, true_cps=true_cps)



# Example usage
lambda_param = 0.05  # Lambda parameter for exponential distribution
mean_var_pairs = [(0, 1), (5, 1.5), (10, 2)]
num_segments = 3
segment_length = 100

# We create distribution objects for each segment and the changepoints.
changepoint_dist = expon(scale=1/lambda_param)
segment_dists = [norm(loc=mean, scale=np.sqrt(var)) for mean, var in mean_var_pairs]

generator = SyntheticChangepointData(segment_dists, changepoint_dist)
time_series = generator.generate_data(num_segments=num_segments, segment_length=segment_length)
print(time_series)

# Visualize the changepoints
import matplotlib.pyplot as plt

changepoints = generator.generate_changepoints(num_segments=num_segments)
for cp in changepoints:
    plt.axvline(x=cp, color='r', linestyle='--')
plt.plot(time_series)
plt.show()














