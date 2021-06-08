input_file = "stabilize_some_results.json"

import json
with open(input_file, "r") as in_handle:
    results = json.load(in_handle)

from tabulate import tabulate

sizes = results["sizes"][:-1]

headers = ["$\\beta$", *[f"${size}$ neurons" for size in sizes]]

betas = results["betas"]

table = [[beta, *[
    f"${results['results'][size_index][beta_index]:.2f}$" for size_index, _ in enumerate(sizes)
]] for beta_index, beta in enumerate(betas)]

print(tabulate(table, headers=headers, tablefmt="latex_raw"))