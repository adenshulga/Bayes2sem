from abc import ABC, abstractmethod
import numpy as np
from tqdm import tqdm
from utils import TimeSeries
from scipy.stats import anderson, norm, kstest

class ChangePointModel(ABC):

    @abstractmethod
    def find_cp(ts: TimeSeries) -> int:
        pass

def anderson_p_value(sample: np.ndarray) -> float:
    mean, std = np.mean(sample), np.std(sample)

    norm_sample = (sample - mean) / std
    
    ad_statistic, critical_values, significance_levels = anderson(norm_sample, dist='norm')
    p_value = significance_levels[np.where(ad_statistic <= critical_values)][0]

    return p_value

class KolmogorovSmirnovModel(ChangePointModel):

    def compute_uncertainty(self, probs: np.ndarray, max_prob: np.ndarray, cp: int) -> int:
        
        half_max = max_prob / 2

        cross_points = np.where(np.diff(np.sign(probs - half_max)))[0]

        left_index = None
        right_index = None

        # Check if there are any cross points less than cp
        if np.any(cross_points < cp):
            left_index = cross_points[cross_points < cp].max()
        else:
            left_index = cp - 1  
        # Check if there are any cross points greater than cp
        if np.any(cross_points > cp):
            right_index = cross_points[cross_points > cp].min()
        else:
            # Handle case where there are no cross points greater than cp
            right_index = cp + 1  # or some other default value or handling logic

        # Ensure both indices are found before calculating width
        if left_index is not None and right_index is not None:
            width = right_index - left_index
            return width
        # else:
        #     # Handle case where we could not find a proper width
        #     return 0  # or some other default value or error handling


    def find_cp(self, ts: TimeSeries) -> int:
        max_prob = -np.inf
        best_changepoint = None
        
        def kolmogorov_smirnov_test(sample: np.ndarray) -> float:
            # print(type(sample))
            # print(sample.shape)
            if sample.shape[0] == 0:
                return 1 
            
            mean, std = np.mean(sample), np.std(sample)
            norm_sample = (sample - mean)/std

            p_value = kstest(norm_sample.ravel(), norm.cdf).pvalue
            return p_value

        time_series = ts.ts

        probs = []
        for t_cp in tqdm(range(10, len(time_series) - 1)):  # Iterate over possible changepoint locations
            # Split the time series into two segments
            segment1 = time_series[:t_cp]
            segment2 = time_series[t_cp:]

            probability = kolmogorov_smirnov_test(segment1) * kolmogorov_smirnov_test(segment2)
            probs.append(probability)

            if probability > max_prob:
                max_prob = probability
                best_changepoint = t_cp

        if max(probs[-50:]) > 0.9:
            best_changepoint = ts.len + 1
            uncertainty = 10
            return best_changepoint, uncertainty, probs

        
        uncertainty = self.compute_uncertainty(probs, max_prob, best_changepoint)

        return best_changepoint, uncertainty, probs




        




















