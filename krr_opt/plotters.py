# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error


def plot_metrics(yp_val, yp_test, y_val, y_test,
                 n_train, n_val, dataset_name,
                 results_dir, lim=(None, None),
                 yp_train=None, y_train=None):

    def add_col(col_num, title, y_true, y_pred, lim,
                coords_mse=(0, 0), coords_mae=(0, 0)):
        font = dict(
            color='black',
            weight='normal',
            size=10
        )

        mse = mean_squared_error(y_true, y_pred, squared=False)
        mae = mean_absolute_error(y_true, y_pred)

        axs[col_num].axline([0, 0], [1, 1])
        axs[col_num].set(xlim=lim, ylim=lim)

        axs[col_num].text(*coords_mse,
                          'RMSE: {:.3e}'.format(mse),
                          fontdict=font, ha='left', va='top',
                          transform=axs[col_num].transAxes)
        axs[col_num].text(*coords_mae,
                          'MAE: {:.3e}'.format(mae),
                          fontdict=font, ha='left', va='top',
                          transform=axs[col_num].transAxes)

        axs[col_num].scatter(y_true, y_pred, c='black', zorder=3, marker='.')
        axs[col_num].set_xlabel('true values')
        axs[col_num].set_title(title)

    x1, x2 = 0, 0  # setting text coordinates
    if yp_train is not None and y_train is not None:  # 3 subplots (train/val/test)
        y1, y2 = -0.2, -0.3

        fig, axs = plt.subplots(3, figsize=(8, 10))
        fig.suptitle(f'n_train: {n_train}; n_val: {n_val}')
        # train: we might need to visualize train predictions,
        # since initial weights are changing during optimization
        add_col(0, 'train', y_train, yp_train, lim,
                coords_mse=(x1, y1), coords_mae=(x2, y2))
        # validation
        add_col(1, 'validation', y_val, yp_val, lim,
                coords_mse=(x1, y1), coords_mae=(x2, y2))
        # test
        add_col(2, 'test', y_test, yp_test, lim,
                coords_mse=(x1, y1), coords_mae=(x2, y2))
        plt.subplots_adjust(hspace=0.5, wspace=0.5)

    else:  # 2 subplots (val/test)
        y1, y2 = -0.15, -0.22

        fig, axs = plt.subplots(2, figsize=(8, 8))
        fig.suptitle(f'n_train: {n_train}; n_val: {n_val}')
        # validation
        add_col(0, 'validation', y_val, yp_val, lim,
                coords_mse=(x1, y1), coords_mae=(x2, y2))
        # test
        add_col(1, 'test', y_test, yp_test, lim,
                coords_mse=(x1, y1), coords_mae=(x2, y2))
        plt.subplots_adjust(hspace=0.4)

    plt.savefig(f'{results_dir}/{dataset_name}/plots/'
                f'{n_train}_train_{n_val}_val.jpeg', dpi=500)
    plt.close(fig)


def plot_performance(points, combined_mse, combined_mae,
                     dataset_name, results_dir):
    """https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.loglog.html"""
    fig = plt.figure(figsize=(12, 12))
    plt.rc('xtick', labelsize=14)
    plt.rc('ytick', labelsize=14)

    plt.loglog(points, combined_mse, '--ro', linewidth=2, markersize=14, label='RMSE')
    plt.loglog(points, combined_mae, '--bo', linewidth=2, markersize=14, label='MAE')
    plt.xlabel('number of training points', fontsize=14)
    plt.ylabel('metric, kcal/mol', fontsize=14)

    plt.legend(fontsize=14)
    plt.savefig(f'{results_dir}/{dataset_name}'
                f'/plots/performance.jpeg', dpi=500)
    plt.close(fig)
