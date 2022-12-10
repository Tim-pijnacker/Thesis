import torch
import pickle
import os

import torch.utils.benchmark as benchmark
from itertools import product
from collections import defaultdict

# functions to benchmark
# from python.nsection import entmax_nsect
# from cpp.nsection import entmax_nsect_cpp
# from cuda.nsection import entmax_nsect_cuda
# from entmax import entmax_bisect
# from torch.nn.functional import softmax

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
        self._settings_dict = self._create_settings_dict()

        self.row_str = str(self.rows).replace(" ", "")
        self.col_str = str(self.cols).replace(" ", "")
        self.path = f"benchmark/bench_{self.row_str}_{self.col_str}"

        self._load()


    def _save(self) -> bool:
        """Save index to disk."""

        if self.path is None:
            return False

        if not os.path.isdir(self.path):
            os.makedirs(self.path)

        with open(os.path.join(self.path, f"bench_{self.row_str}_{self.col_str}.pkl"), "wb") as f:
            pickle.dump(self._bench_list, f)

        return True


    def _load(self) -> bool:
        """Load index from disk."""

        if self.path is None:
            return False

        if not os.path.isdir(self.path):
            return False

        with open(os.path.join(self.path, f"bench_{str(self.row_str)}_{str(self.col_str)}.pkl"), "rb") as f:
            self._bench_list = pickle.load(f)

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
        self.label = 'entmax, bisect and softmax time'

        for r, c in product(self.rows, self.cols):
            x = torch.randn(r, c, device=torch.device("cuda:0"), dtype=torch.float32)
            sub_label = f'[{r}, {c}]'
            for thread in threads:
                for model in self.models:
                    self._bench_list.append(self._time_model(x, model, sub_label, thread))


    def initialise(self, threads=[1,4,16,32]):
        self._create_bench_list(threads=threads)
        self._save()
    

    def compare(self):
        compare = benchmark.Compare(self._bench_list)
        compare.colorize()
        compare.print()


    def plots(self):
        pass