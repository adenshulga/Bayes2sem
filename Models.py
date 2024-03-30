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

class MaxLikelyhoodModel(ChangePointModel):

    def find_cp(self, ts: TimeSeries) -> int:
        max_prob = -np.inf
        best_changepoint = None
        
        def kolmogorov_smirnov_test(sample: np.ndarray) -> float:
            # print(type(sample))
            # print(sample.shape)
            
            mean, std = np.mean(sample), np.std(sample)
            norm_sample = (sample - mean)/std

            p_value = kstest(norm_sample.ravel(), norm.cdf).pvalue
            return p_value

        time_series = ts.ts

        probs = []
        for t_cp in tqdm(range(2, len(time_series) - 1)):  # Iterate over possible changepoint locations
            # Split the time series into two segments
            segment1 = time_series[:t_cp]
            segment2 = time_series[t_cp:]

            probability = kolmogorov_smirnov_test(segment1) * kolmogorov_smirnov_test(segment2)
            probs.append(probability)

            if probability > max_prob:
                max_prob = probability
                best_changepoint = t_cp

        return best_changepoint, max_prob, probs



        #     # Compute empirical mean and standard deviation for each segment
        #     mean1, std1 = np.mean(segment1), np.std(segment1)
        #     mean2, std2 = np.mean(segment2), np.std(segment2)

        #     # Compute likelihood for each segment
        #     likelihood1 = np.log(np.prod(norm.pdf(segment1, mean1, std1)))
        #     likelihood2 = np.log(np.prod(norm.pdf(segment2, mean2, std2)))

        #     # Multiply the likelihoods for the two segments
        #     total_likelihood = likelihood1 + likelihood2

        #     likelihoods.append(total_likelihood)

        #     # Update maximum likelihood and best changepoint if necessary
        #     if total_likelihood > max_likelihood:
        #         max_likelihood = total_likelihood
        #         best_changepoint = t_cp

        # return best_changepoint, likelihoods

        




















