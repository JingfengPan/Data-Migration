import os


def cost_model(
        classification_time: float,
        compression_time: float,
        compression_ratio: float,
        data_path: str,
        n_processors: int,
        price_cpu: float = 0.048,  # the price of using one CPU core per hour
        price_net: float = 0.05,  # the price of transforming data per gigabyte
        cost_scale: int = 1024 * 1024  # calculate the cost of TBs
):
    original_size = os.stat(data_path).st_size / (1024 * 1024)  # MB
    compressed_size = original_size / compression_ratio
    base_cost = ((price_cpu / 3600) * n_processors) * (classification_time + compression_time) + (compressed_size * price_net / 1024)
    cost = (cost_scale / original_size) * base_cost
    return cost
