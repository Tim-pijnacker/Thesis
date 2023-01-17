import torch
import pickle
import os
import matplotlib.pyplot as plt
import numpy as np

import torch.utils.benchmark as benchmark
from itertools import product
from collections import defaultdict

# functions to benchmark
from cpp.nsection import entmax_nsect_cpp
from python.nsection import entmax_nsect
from cuda.nsection import entmax_nsect_cuda
from entmax import entmax_bisect
from torch.nn.functional import softmax

class benchmarker():
    def __init__(
        self,
        alpha = 1.5,
        nsct_iter = 5, 
        bisct_iter = 25,
        n_sections = 32, 
        rows = [10,100],
        cols = [1000,10000],
        models = ["py", "cpp", "cuda", "bisct", "soft"]
    ) -> None:

        self.alpha = alpha
        self.nsct_iter=nsct_iter
        self.bisct_iter=bisct_iter
        self.n_sections=n_sections
        self.rows = rows
        self.cols = cols
        self.models=models

        self.label = 'entmax, bisect and softmax time'
        self._settings_dict = self._create_settings_dict()

        self.row_str = str(self.rows).replace(" ", "")
        self.col_str = str(self.cols).replace(" ", "")
        self.path_bench = f"benchmark/bench_{self.row_str}_{self.col_str}"
        self.path_plot = f"benchmark/plot_{self.row_str}_{self.col_str}"

        self._load_bench()
        self._load_plot()

    def _save_bench(self) -> bool:
        """Save index to disk."""

        if self.path_bench is None:
            return False

        if not os.path.isdir(self.path_bench):
            os.makedirs(self.path_bench)

        with open(os.path.join(self.path_bench, f"bench_{self.row_str}_{self.col_str}.pkl"), "wb") as f:
            pickle.dump(self._bench_list, f)

        return True

    def _load_bench(self) -> bool:
        """Load index from disk."""

        if self.path_bench is None:
            return False

        if not os.path.isdir(self.path_bench):
            return False

        with open(os.path.join(self.path_bench, f"bench_{str(self.row_str)}_{str(self.col_str)}.pkl"), "rb") as f:
            self._bench_list = pickle.load(f)

        return True
    
    def _save_plot(self) -> bool:
        """Save index to disk."""

        if self.path_plot is None:
            return False

        if not os.path.isdir(self.path_plot):
            os.makedirs(self.path_plot)

        with open(os.path.join(self.path_plot, f"plot_{self.row_str}_{self.col_str}.pkl"), "wb") as f:
            pickle.dump(self._plot_dict, f)

        return True

    def _load_plot(self) -> bool:
        """Load index from disk."""

        if self.path_plot is None:
            return False

        if not os.path.isdir(self.path_plot):
            return False

        with open(os.path.join(self.path_plot, f"plot_{str(self.row_str)}_{str(self.col_str)}.pkl"), "rb") as f:
            self._plot_dict = pickle.load(f)
            self.x_vals = self._plot_dict["x"]
            del self._plot_dict["x"]

        return True

    def _create_settings_dict(self):
        settings_dict = defaultdict()
        for model in self.models:
            model_dict = defaultdict()
            if model == "py":
                model_dict['stmt'] = f'entmax_nsect(x, alpha = {self.alpha}, n_iter={self.nsct_iter}, n_sections={self.n_sections})'
                model_dict['setup'] = 'from python.nsection import entmax_nsect'
            elif model == "cpp":
                model_dict['stmt'] = f'entmax_nsect_cpp(x, alpha = {self.alpha}, n_iter={self.nsct_iter}, n_sections={self.n_sections})'
                model_dict['setup'] = 'from cpp.nsection import entmax_nsect_cpp'
            elif model == "cuda":
                model_dict['stmt'] = f'entmax_nsect_cuda(x, alpha = {self.alpha}, n_iter={self.nsct_iter}, n_sections={self.n_sections})'
                model_dict['setup'] = 'from cuda.nsection import entmax_nsect_cuda'
            elif model == "cuda1":
                model_dict['stmt'] = f'entmax_nsect_cuda1(x, alpha = {self.alpha}, n_iter={self.nsct_iter}, n_sections={self.n_sections})'
                model_dict['setup'] = 'from cuda.nsection import entmax_nsect_cuda1'
            elif model == "bisct":
                model_dict['stmt'] = f'entmax_bisect(x, alpha = {self.alpha}, n_iter={self.bisct_iter})'
                model_dict['setup'] = 'from entmax import entmax_bisect'
            elif model == "soft":
                model_dict['stmt'] = f'softmax(x, dim=-1)'
                model_dict['setup'] = 'from torch.nn.functional import softmax'
            settings_dict[model] = model_dict
        return settings_dict

    def _time_model(self, x, model, sub_label, num_threads):
        timer = benchmark.Timer(
            stmt=self._settings_dict[model]['stmt'],
            setup=self._settings_dict[model]['setup'],
            globals={'x': x},
            num_threads=num_threads,
            label=self.label,
            sub_label=sub_label,
            description=model,
        ).timeit(100)

        return timer

    def _create_bench_list(self, threads):
        self._bench_list = []

        prod = product(self.rows, self.cols)
        prod = list(prod)
        prod.append((32000, 64))
        for r, c in prod:
            x = torch.randn(r, c, device=torch.device("cuda:0"), dtype=torch.float32)
            sub_label = f'[{r}, {c}]'
            for thread in threads:
                for model in self.models:
                    self._bench_list.append(self._time_model(x, model, sub_label, thread))
    
    def _create_plot_list(self, x_len,base):
        self._plot_dict = defaultdict()
        
        self.x_vals = []
        for i in range(x_len):
            self.x_vals.append(base ** (7 + i))

        self._plot_dict["x"] = self.x_vals

        for r in self.rows:
            row_dict = defaultdict(list)
            for c in self.x_vals:
                torch.manual_seed(40)
                x = torch.randn(r, c, device=torch.device("cuda:0"), dtype=torch.float32)
                sub_label = f'[{r}, {c}]'
                for model in self.models:
                    t = self._time_model(x, model, sub_label, 1)
                    row_dict[model].append(t.mean)
            self._plot_dict[r] = row_dict
            
    def add_model_bench(self, new_model, stmt, setup, threads=[1,4,16,32]):
        model_dict = defaultdict()
        model_dict["stmt"] = stmt
        model_dict["setup"] = setup
        self._settings_dict[new_model] = model_dict
        
        for r, c in product(self.rows, self.cols):
            torch.manual_seed(40)
            x = torch.randn(r, c, device=torch.device("cuda:0"), dtype=torch.float32)
            sub_label = f'[{r}, {c}]'
            for thread in threads:
                self._bench_list.append(self._time_model(x, new_model, sub_label, thread))
                
    def del_model_bench(self, del_model, threads=[1,4,16,32]):
        del self._settings_dict[del_model]
        
        for thread in threads:
            self._bench_list.pop(-1)

    def initialise_bench(self, threads=[1,4,16,32]):
        self._create_bench_list(threads=threads)
        self._save_bench()
    
    def initialise_plot(self, x_len=9, base=2):
        self._create_plot_list(x_len=x_len,base=base)
        self._save_plot()

    def compare(self):
        compare = benchmark.Compare(self._bench_list)
        compare.colorize()
        compare.print()

    def plot(self):
        if "x" in self._plot_dict.keys():
            self.x_vals = self._plot_dict["x"]
            del self._plot_dict["x"]

        n_plots = len(self._plot_dict.keys())
        fig, axs = plt.subplots(1, n_plots, figsize=(10, 6))
        fig.suptitle("Time plots for differnet input sizes")
        for idx, n_rows in enumerate(self._plot_dict.keys()):
            for model in self.models:
                axs[idx].plot(self.x_vals, self._plot_dict[n_rows][model])
                axs[idx].set(xlabel='input dimension', ylabel='time (s)')
                axs[idx].set_xscale('log', base=2)
            axs[idx].set_title(f'{n_rows} input rows ')
        fig.legend(labels=self.models)
        plt.show()


def main():
    bench = benchmarker(alpha = 1.5, nsct_iter = 5, bisct_iter = 25, n_sections = 32, rows = [10, 100], cols = [100, 1000, 10000], models=["py", "cpp", "cuda", "cuda1", "bisct", "soft"])
    bench.initialise_bench(threads=[1])
    bench.compare()
    bench = benchmarker(alpha = 1.5, nsct_iter = 5, bisct_iter = 25, n_sections = 32, rows = [100, 400, 1000], cols = [32000], models=["py", "cpp", "cuda", "cuda1", "bisct", "soft"])
    bench.initialise_bench(threads=[1])
    # bench.initialise_plot()
    bench.compare()
    # bench.plot()

if __name__ == "__main__":
    main()