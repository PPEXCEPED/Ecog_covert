#!/Users/DELL/anaconda3/envs/ECOG/python.exe
# -*- coding: utf-8 -*-
# @Time : 2023/10/30 1:26
# @Author : Zhenjie Wang
from HS_reading import pairs_sound_avg
from HS_reading import pairs

import datetime
import multiprocessing as mp




if __name__ == '__main__':
    clean_data_path = "D:/BaiduSyncdisk/code"
    start_t = datetime.datetime.now()

    num_cores = int(mp.cpu_count()) - 1
    print("本地计算机有: " + str(num_cores) + " 核心")
    pool = mp.Pool(num_cores)
    param_dict = {
                    # 'task1': [44],
                  'task2': [47],
                  'task3': [50],
                  'task4': [71],
                  # 'task5': [45],
                  'task6': [48],
                  'task7': [54],
                  'task8': [73],
                  'task9': [76],
                  'task10': [78]}
    results = [pool.apply_async(pairs, args=(75,param,clean_data_path)) for name, param in param_dict.items()]
    results = [p.get() for p in results]

    end_t = datetime.datetime.now()
    elapsed_sec = (end_t - start_t).total_seconds()
    print("多进程计算 共消耗: " + "{:.2f}".format(elapsed_sec) + " 秒")