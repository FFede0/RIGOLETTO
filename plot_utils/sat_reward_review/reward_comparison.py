from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import parse
import os


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Plot training or evaluation results", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-i", "--input_files",
    help = "Input files",
    nargs = "+",
    type = str,
    required = True
  )
  parser.add_argument(
    "-o", "--output_folder",
    help = "Folder to store plots",
    type = str,
    required = True
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def load_data(input_files: list) -> pd.DataFrame:
  data = pd.DataFrame()
  for filename in input_files:
    _, rf = parse.parse("{}_{}.csv", os.path.basename(filename))
    rf_data = pd.read_csv(filename)[[
      "iteration", 
      "pct_concluded", 
      "pct_optimal",
      "mean_diff_iter",
      "mean_reward_iter",
      "min_reward_iter",
      "max_reward_iter"
    ]]
    rf_data["method"] = rf
    # join
    data = pd.concat([data, rf_data], ignore_index = True)
  return data


def load_dijkstra_data() -> pd.DataFrame:
  dijkstra_data_group = pd.read_csv(
    "plot_utils/sat_reward_review/report_reward_jumps.csv"
  )[["function_name","total_reward"]].groupby("function_name")
  dijkstra_data = pd.DataFrame(
    dijkstra_data_group.min()
  ).join(
    pd.DataFrame(dijkstra_data_group.max()),
    lsuffix = "_min",
    rsuffix = "_max"
  ).join(
    pd.DataFrame(dijkstra_data_group.mean())
  ).reset_index().rename(
    columns = {"function_name": "method", "total_reward": "total_reward_mean"}
  ).set_index("method")
  return dijkstra_data


def main(input_files: list, output_folder: str):
  reward_data = load_data(input_files)
  dijkstra_reward_data = load_dijkstra_data()
  # plot
  _, ax = plt.subplots(figsize = (12,8))
  for method, method_reward in reward_data.groupby("method"):
    # moving average
    reward_data_ma = method_reward.rolling(
      window = 100
    ).mean(numeric_only = True)
    # plot
    reward_data_ma.plot(
      x = "iteration",
      y = "mean_reward_iter",
      label = method,
      linewidth = 2,
      ax = ax,
      grid = True,
      fontsize = 14
    )
    ax.fill_between(
      x = reward_data_ma["iteration"].values,
      y1 = reward_data_ma["min_reward_iter"].values,
      y2 = reward_data_ma["max_reward_iter"].values,
      alpha = 0.1
    )
  # plot Dijkstra bounds
  for idx, method in enumerate(["PErf", "EErf"]):
    dijkstra_reward = dijkstra_reward_data.loc[method, :]
    color = [
      mcolors.CSS4_COLORS["dimgrey"],
      mcolors.CSS4_COLORS["black"]
    ][idx]
    ax.axhline(
      y = dijkstra_reward["total_reward_min"],
      linestyle = "dashed",
      linewidth = 1,
      color = color
    )
    ax.axhline(
      y = dijkstra_reward["total_reward_max"],
      linestyle = "dashed",
      linewidth = 1,
      color = color
    )
    ax.axhline(
      y = dijkstra_reward["total_reward_mean"],
      linestyle = "solid",
      linewidth = 2,
      color = color
    )
  ax.set_xlabel("Evaluation interval", fontsize = 14)
  ax.set_ylabel("Reward", fontsize = 14)
  plt.legend(fontsize = 14)
  plt.savefig(os.path.join(output_folder, "reward_comparison.png"), dpi=300)
  plt.close()


if __name__ == "__main__":
  args = parse_arguments()
  input_files = args.input_files
  output_folder = args.output_folder
  main(input_files, output_folder)
