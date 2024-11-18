'''3 sigma 异常值检测'''
import numpy as np
import matplotlib.pyplot as plt


def find_anomalies(data_in):
    # Set upper and lower limit to 3 standard deviation
    anomalies = []
    random_data_std = np.std(data_in)
    random_data_mean = np.mean(data_in)
    sigma3 = random_data_std * 3

    lower_limit  = random_data_mean - sigma3 
    upper_limit = random_data_mean + sigma3
    print(lower_limit)
    print(upper_limit)
    # Generate outliers
    for outlier in data_in:
        if outlier > upper_limit or outlier < lower_limit:
            anomalies.append(outlier)
    return anomalies



# 3sigma原则
data = [1,4,8,90,98,44,35,56,2,41,11,24,23,45,500, 150]
error = find_anomalies(data)

print(error)
