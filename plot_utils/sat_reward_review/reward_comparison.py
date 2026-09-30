from matplotlib import colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd


def load_data() -> pd.DataFrame:
  data = pd.DataFrame()
  for rf in [
      "DOrf", "PErf", "EErf", "2-stages curriculum", "3-stages curriculum"
    ]:
    rf_data = pd.read_csv(f"../sat_plot_summary/summaries_csv/{rf}.csv")[[
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
  dijkstra_data_group = pd.read_csv("report_reward_jumps.csv")[[
    "function_name",
    "total_reward"
  ]].groupby("function_name")
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


def main():
  reward_data = load_data()
  dijkstra_reward_data = load_dijkstra_data()
  # plot
  _, ax = plt.subplots()
  for method, method_reward in reward_data.groupby("method"):
    # moving average
    reward_data_ma = method_reward.rolling(
      window = 50
    ).mean(numeric_only = True)
    # plot
    reward_data_ma.plot(
      x = "iteration",
      y = "mean_reward_iter",
      label = method,
      linewidth = 2,
      ax = ax,
      grid = True
    )
    ax.fill_between(
      x = reward_data_ma["iteration"].values,
      y1 = reward_data_ma["min_reward_iter"].values,
      y2 = reward_data_ma["max_reward_iter"].values,
      alpha = 0.2
    )
    # plot Dijkstra bounds
    if "curriculum" not in method:
      dijkstra_reward = dijkstra_reward_data.loc[method, :]
      color = plt.gca().lines[-1].get_color()
      ax.axhline(
        y = dijkstra_reward["total_reward_min"],
        linestyle = "dashed",
        linewidth = 2,
        color = color
      )
      ax.axhline(
        y = dijkstra_reward["total_reward_max"],
        linestyle = "dashed",
        linewidth = 2,
        color = color
      )
      ax.axhline(
        y = dijkstra_reward["total_reward_mean"],
        linestyle = "solid",
        linewidth = 3,
        color = color
      )
  ax.set_xlabel("Evaluation interval")
  ax.set_ylabel("Reward")
  plt.show()


if __name__ == "__main__":
  main()
