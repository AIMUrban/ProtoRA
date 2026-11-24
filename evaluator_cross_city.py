import json
import pandas as pd
from sklearn import linear_model, svm
from sklearn.metrics import mean_absolute_error, mean_squared_error, top_k_accuracy_score, root_mean_squared_error
import numpy as np
import argparse
import os
import random





def metrics_local(y_truths, y_preds):
    y_preds[y_preds < 0] = 0

    mae = mean_absolute_error(y_truths, y_preds)
    rmse = root_mean_squared_error(y_truths, y_preds)
    return mae, rmse


def scale_transform(f, min_o, max_o, min_t, max_t):
    transformed_values = (f-min_o)/(max_o-min_o)*(max_t-min_t) + min_t
    return transformed_values

def evaluation_reg(s_X, s_y, t_X, t_y):
    X_train, X_test = s_X, t_X
    y_train, y_test = s_y, t_y

    reg = linear_model.Ridge(alpha=1.0)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    y_truth = y_test

    mae, rmse = metrics_local(y_truth, y_pred)
    return mae, rmse


class Evaluator(object):
    def __init__(self, data_path):
        self.result = {}
        # self.dataset = config.get('dataset', '')
        self.s_dataset = None
        self.t_dataset = None
        self.data_path = data_path
        self.s_data_path = None
        self.t_data_path = None
        self.best_epoch = None

    def collect(self, batch):
        pass

    def _valid_cross_region_gdp(self, s_region_emb, t_region_emb, print_flag):

        s_gdp = np.load(self.data_path + '{}/gdp_sum.npy'.format(
            self.s_dataset)).astype('float32')
        t_gdp = np.load(self.data_path + '{}/gdp_sum.npy'.format(
            self.t_dataset)).astype('float32')

        mae, rmse = evaluation_reg(s_region_emb, s_gdp, t_region_emb, t_gdp)
        if print_flag:
            print('======================================================')
            print('Evaluating Region GDP Prediction')
            print("Result of {} estimation in {} based on {}:".format('GDP', self.t_dataset, self.s_dataset))
            print('MAE = {:.3f}, RMSE = {:.3f}'.format(mae, rmse))
        return mae, rmse

    def _valid_cross_region_pp(self, s_region_emb, t_region_emb, print_flag):

        s_gdp = np.load(self.data_path + '{}/population_sum.npy'.format(
            self.s_dataset)).astype('float32')
        t_gdp = np.load(self.data_path + '{}/population_sum.npy'.format(
            self.t_dataset)).astype('float32')

        mae, rmse = evaluation_reg(s_region_emb, s_gdp, t_region_emb, t_gdp)
        if print_flag:
            print('======================================================')
            print('Evaluating Region Population Prediction')
            print("Result of {} estimation in {} based on {}:".format('population', self.t_dataset, self.s_dataset))
            print('MAE = {:.3f}, RMSE = {:.3f}'.format(mae, rmse))
        return mae, rmse

    def _valid_cross_region_co2(self, s_region_emb, t_region_emb, print_flag):

        s_gdp = np.load(self.data_path + '{}/co2_sum.npy'.format(
            self.s_dataset)).astype('float32')
        t_gdp = np.load(self.data_path + '{}/co2_sum.npy'.format(
            self.t_dataset)).astype('float32')

        mae, rmse = evaluation_reg(s_region_emb, s_gdp, t_region_emb, t_gdp)
        if print_flag:
            print('======================================================')
            print('Evaluating Region CO2 Prediction')
            print("Result of {} estimation in {} based on {}:".format('CO2', self.t_dataset, self.s_dataset))
            print('MAE = {:.3f}, RMSE = {:.3f}'.format(mae, rmse,))
        return mae, rmse


    def evaluate_region_embedding(self, s_region_emb, t_region_emb, print_flag=False):
        if print_flag:
            print('======================================================')
            print('source region emb shape = {}, target region shape = {}'.format(
                s_region_emb.shape, t_region_emb.shape))

        # region_mae, region_rmse, region_mape, region_r2 = self._valid_cross_region_flow(s_region_emb, t_region_emb, print_flag)
        gdp_mae, gdp_rmse, = self._valid_cross_region_gdp(s_region_emb, t_region_emb, print_flag)
        pp_mae, pp_rmse, = self._valid_cross_region_pp(s_region_emb, t_region_emb, print_flag)
        co2_mae, co2_rmse, = self._valid_cross_region_co2(s_region_emb, t_region_emb, print_flag)

        self.result['gdp_mae'] = gdp_mae
        self.result['gdp_rmse'] = gdp_rmse

        self.result['pp_mae'] = pp_mae
        self.result['pp_rmse'] = pp_rmse

        self.result['co2_mae'] = co2_mae
        self.result['co2_rmse'] = co2_rmse



    def evaluate(self, s_dataset, t_dataset, s_region_emb, t_region_emb, save_flag=False, save_path=None, method=None, seed=None, print_flag=False):
        # self.best_epoch = best_epoch
        self.s_dataset = s_dataset
        self.t_dataset = t_dataset
        self.s_data_path = self.data_path + self.s_dataset + '/'
        self.t_data_path = self.data_path + self.t_dataset + '/'

        self.clear()
        self.evaluate_region_embedding(s_region_emb, t_region_emb, print_flag=print_flag)

        if save_flag:
            save_path = os.path.join(save_path, '{}/{}-{}'.format(method, s_dataset, t_dataset))
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            result_path = os.path.join(save_path, 'result_seed_{}.json'.format(seed))
            # print(self.result)
            json.dump(self.result, open(result_path, 'w'), indent=4)
            print('Evaluate result is saved at {}'.format(result_path))

        return self.result

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        self.result = {}

