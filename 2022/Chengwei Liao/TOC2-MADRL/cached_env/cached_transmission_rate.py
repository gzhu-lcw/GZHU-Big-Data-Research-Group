import math

"""
计算数据传输速率
"""


def calculate_data_rate(distance, power, channel_bandwidth):
    noise_square = 1e-10
    channel_gain = 128.1 + 37.6 * math.log10(distance)
    signal_to_noise_ratio = (power * channel_gain) / noise_square

    max_modulation_order = math.log2(1 + signal_to_noise_ratio)

    max_data_rate = channel_bandwidth * max_modulation_order

    return max_data_rate

