import os
import statistics
import numpy as np


def calculate_throughput(t_class, t_comp, comp_ratio, network_speed, path):
    data_size = os.stat(path).st_size / (1024 * 1024)  # MB
    class_throughput = data_size / t_class
    comp_throughput = data_size / t_comp
    network_throughput = network_speed * comp_ratio
    if class_throughput < comp_throughput:
        throughput = class_throughput
        bottleneck = 'classification'
    else:
        throughput = comp_throughput
        bottleneck = 'compression'
    if network_throughput < throughput:
        throughput = network_throughput
        bottleneck = 'network speed'
    return throughput, bottleneck


'''
    if pipelines == 1:
    else:
        min_throughput = min(class_throughput, comp_throughput)
        pipeline_ratio = (min_throughput / (network_speed * comp_ratio)) * pipelines
        if pipeline_ratio > 1:
            # ave_comp_ratio = statistics.harmonic_mean(compression_ratios)
            throughput = comp_ratio * network_speed
        else:
            throughput = min_throughput
'''