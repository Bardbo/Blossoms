import os
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
from tqdm import tqdm
from itertools import product

def get_algorithm_df(func_ind, ndim, metrics="min", runs=51, year=2017):
    algorithm_df = dict()
    for algorithm_name in os.listdir("result"):
        file_name = f"result/{algorithm_name}/F{func_ind}{year}-{ndim}-{runs}.csv"
        data = pd.read_csv(file_name)
        match metrics:
            case "min":
                col_index = data.ffill().iloc[-1].argmin()
                data = data.iloc[:, col_index]
            case "max":
                col_index = data.ffill().iloc[-1].argmax()
                data = data.iloc[:, col_index]
            case "mean":
                data = data.mean(axis=1)
        algorithm_df[algorithm_name] = data
    algorithm_df = pd.DataFrame(algorithm_df)
    return algorithm_df

if __name__ == "__main__":
    func_ids = list(range(1, 30))
    ndims = [10, 30, 50, 100]
    for func_id, ndim in tqdm(product(func_ids, ndims)):
        algorithm_df = get_algorithm_df(func_id, ndim)

        plot_top_n = 5
        with plt.style.context(['science', 'grid']):
            fig, ax = plt.subplots(dpi=300)
            plot_df = algorithm_df[algorithm_df.min(axis=0).sort_values().index[:plot_top_n]]
            min_value = plot_df.min().min()
            max_value = plot_df.max().max()
            ax.axhline(y=min_value, linestyle='--', color="r", lw=0.3)
            plot_df.plot(ax=ax)
            ax.text(2, min_value-0.03*(max_value-min_value), f'$min={min_value:.2f}$', fontsize=4)
            ax.set_xlabel("Generation Number")
            ax.set_ylabel(f"F{func_id}, ndim={ndim}")
            plt.savefig(fname=f"fig\F{func_id}-{ndim}.png")
            plt.close()
        print(f"F{func_id}-{ndim} Finished .")