import os

import numpy as np
from pathlib import Path
from evaluator_cross_city import Evaluator
from ProtoRA import ProtoRA
import pickle


if __name__ == '__main__':
    evaluator = Evaluator(data_path='./region_data/')
    save_flag = False
    print_flag = True
    city_pairs = [('xa', 'cd'), ('cd', 'xa'), ('bj', 'cd'), ('cd', 'bj'), ('xa', 'bj'), ('bj', 'xa')]
    save_path = './output/'
    pretrained_emb = 'hrep'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for seed in [1,]:
        for city_pair in city_pairs:
            source_city, target_city = city_pair
            city_p = '{}-{}'.format(source_city, target_city)

            s_emb_aligned, t_emb_aligned = ProtoRA(source_city, target_city, seed, backbone=pretrained_emb, info_flag=True,
                                                        info_source=['poi', 'mob'])
            evaluator.evaluate(source_city, target_city, s_emb_aligned, t_emb_aligned, save_flag=save_flag,
                               save_path=save_path, method='w_poi_mob', seed=seed, print_flag=print_flag)

            s_emb_aligned, t_emb_aligned = ProtoRA(source_city, target_city, seed, backbone=pretrained_emb, info_flag=True, info_source=['poi'])
            evaluator.evaluate(source_city, target_city, s_emb_aligned, t_emb_aligned, save_flag=save_flag,
                               save_path=save_path, method='w_poi', seed=seed, print_flag=print_flag)

            s_emb_aligned, t_emb_aligned = ProtoRA(source_city, target_city, seed, backbone=pretrained_emb, info_flag=True, info_source=['mob'])
            evaluator.evaluate(source_city, target_city, s_emb_aligned, t_emb_aligned, save_flag=save_flag,
                               save_path=save_path, method='w_mob', seed=seed, print_flag=print_flag)

            s_emb_aligned, t_emb_aligned = ProtoRA(source_city, target_city, seed, backbone=pretrained_emb, info_flag=False)
            evaluator.evaluate(source_city, target_city, s_emb_aligned, t_emb_aligned, save_flag=save_flag,
                               save_path=save_path, method='ProtoRA', seed=seed, print_flag=print_flag)
