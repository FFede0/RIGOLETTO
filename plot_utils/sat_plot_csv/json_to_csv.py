import json
import pandas as pd
import argparse
import os

# ----- Script per convertire il file "evaluations.json" preso da cartella in results, in un csv filtrato (EVALUATION) -----


def parse_arguments() -> argparse.Namespace:
  """
  Parse input arguments
  """
  parser: argparse.ArgumentParser = argparse.ArgumentParser(
    description = "Generate per-iteration report", 
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  parser.add_argument(
    "-f", "--base_folder",
    help = "Base folder",
    type = str,
    required = True
  )
  # Parse the arguments
  args: argparse.Namespace = parser.parse_known_args()[0]
  return args


def process_json_to_csv(input_path, output_path, mapping):
  if not os.path.exists(input_path):
    print(f"Error: file '{input_path}' does not exist.")
    return

  try:
    with open(input_path, 'r') as f:
        data = json.load(f)

    rows = [] # Righe per costruire il dataframe, una riga è un'evaluation

    for eval_entry in data.get('evaluations', []):
        # Crea un dizionario per la riga corrente
        row_data = {}
        
        # Recupera fino a sampler_results e hist_stats
        sampler_results = eval_entry.get('sampler_results', {})
        hist_stats = sampler_results.get('hist_stats', {})

        # -- Valore scalare (mentre glialtri valori sono liste di elementi)
        row_data['episodes_this_iter'] = sampler_results.get('episodes_this_iter')


        # -- Valore lista di elementi (per ogni colonna desiderata, prende la lista e la converte in stringa JSON)
        for json_key, csv_column in mapping.items():
            values_list = hist_stats.get(json_key, [])
            # Converte la lista in una stringa tipo "[1, 2, 3]"
            row_data[csv_column] = json.dumps(values_list)
        
        rows.append(row_data)

    # Crea il DataFrame
    df = pd.DataFrame(rows)

    # Ordina le colonne come sono nel progress.csv filtrato
    final_columns = [
        'episodes_this_iter',
        'hist_stats/current_time',
        'hist_stats/step_reward',
        'hist_stats/hole_counter',
        'hist_stats/current_sat',
        'hist_stats/total_distance',
        'hist_stats/episode_reward',
        'hist_stats/episode_lengths',
        'hist_stats/dest_reached',
        'hist_stats/dijkstra_dist',
        'hist_stats/dijkstra_hop',
    ]
    
    # Salvataggio (usa il quoting per evitare che le virgole nelle liste rompano il CSV)
    df.to_csv(output_path, columns=final_columns, index=False)
    
    print(f"Success! Created '{output_path}' with {len(df)} rows (one per iteration).")

  except Exception as e:
    print(f"An error occurred: {e}")


def main(base_folder):

  # File input ed output
  input_json = os.path.join(base_folder, 'evaluations.json')
  output_csv = os.path.join(base_folder, 'estratto_evaluations.csv')

  # Mappatura dei valori interessati: Chiave JSON -> Nome colonna CSV
  mapping = {
      'current_time': 'hist_stats/current_time',
      'step_reward': 'hist_stats/step_reward',
      'hole_counter': 'hist_stats/hole_counter',
      'current_sat': 'hist_stats/current_sat',
      'total_distance': 'hist_stats/total_distance',
      'episode_reward': 'hist_stats/episode_reward',
      'episode_lengths': 'hist_stats/episode_lengths',
      'dest_reached': 'hist_stats/dest_reached',
      'dijkstra_dist': 'hist_stats/dijkstra_dist',
      'dijkstra_hop': 'hist_stats/dijkstra_hop', 
  }

  process_json_to_csv(input_json, output_csv, mapping)


if __name__ == "__main__":
  args = parse_arguments()
  base_folder = args.base_folder
  main(base_folder)
