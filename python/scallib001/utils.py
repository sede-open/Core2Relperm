import numpy as np
import pandas as pd

class dictn(dict):
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        del self[name]

    def copy(self):
        return dictn(self)


def make_movie_schedule(schedule, time_end, start_step_size, increment_factor):
    '''Create time series for schedule each period starting with small step size

    Parameters:
    ----------
    schedule: pandas dataframe
       data frame containing time sequence for experiment
    time_end: float [hour]
       end time of experiment
    start_step_size: fload [hour]
       timestep size at start of each period
    increment_factor: float
       timestep increment factor

    Returns:
    -------
    array: numpy array of floats [hour]
       time sequence
    '''
    
    assert start_step_size > 0.0
    assert increment_factor > 1.0
    assert schedule.StartTime.values[0] == 0.0
    
    # Get start time of each period
    start_times = np.unique(np.sort(schedule.StartTime.values))
    start_times = start_times[start_times < time_end]

    # Get end time of each period
    t_ends = np.hstack( (start_times, [time_end]) )[1:]

    movie_time = []
    time = 0.0
    t_start = 0.0
    for t_end in t_ends:
        i = 0
        while True:
            dtime = np.power(increment_factor, i) * start_step_size
            time = t_start + dtime
            if time > t_end:
                t_start = t_end
                movie_time.append(t_end)
                break
            movie_time.append(time)
            i += 1

    return np.array(movie_time) 
