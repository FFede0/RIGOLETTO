"""
Creazione della rete con random policy
"""

import json
from pathlib import Path

import src

from RL4CC.algorithms.algorithm import Algorithm


BASE_DIR = Path("/app")
CHECKPOINT_DIR = BASE_DIR / "scratch_checkpoints" / "0"

with open(BASE_DIR / "env_config.json") as f:
    env_config = json.load(f)

with open(BASE_DIR / "ray_config.json") as f:
    ray_config = json.load(f)

# RL4CC gestisce la directory dei log tramite il parametro logdir.
# Rimuoviamo la configurazione incompatibile solo dalla copia in memoria.
ray_config.get("debugging", {}).pop("logger_config", None)

# RL4CC richiede evaluation_interval come parametro separato.
# Conserviamo il valore originale definito in ray_config.json.
original_eval_interval = ray_config["evaluation"].pop(
    "evaluation_interval"
)
print("Evaluation interval originale:", original_eval_interval)

# RL4CC configura autonomamente l'ambiente tramite env_config.
# Eliminiamo le chiavi protette solo dalla copia in memoria.
for section_name, section in ray_config.items():
    if isinstance(section, dict):
        for key in ("env", "env_config"):
            if key in section:
                print(
                    "Parametro gestito da RL4CC:",
                    section_name,
                    key
                )
                section.pop(key)

for key in ("env", "env_config"):
    if key in ray_config:
        print("Parametro gestito da RL4CC: root", key)
        ray_config.pop(key)


# Protezione contro la sovrascrittura
if CHECKPOINT_DIR.exists() and any(CHECKPOINT_DIR.iterdir()):
    raise RuntimeError(
        f"Checkpoint già presente: {CHECKPOINT_DIR}"
    )

# La configurazione serializzata contiene un segnaposto,
# non una funzione Python valida per RLlib.
multi_agent = ray_config["multi_agent"]

if multi_agent.get("policy_mapping_fn") != "__not_serializable__":
    raise RuntimeError("policy_mapping_fn inattesa: verificare manualmente")

multi_agent.pop("policy_mapping_fn")

# Directory temporanea per i log di inizializzazione.
log_dir = Path("/tmp/rigoletto_init_logs")
log_dir.mkdir(parents=True, exist_ok=True)

print("Inizializzazione DQN RIGOLETTO da zero...", flush=True)

algorithm = Algorithm(
    algo_name="DQN",
    checkpoint_path=None,
    env_config=env_config,
    ray_config=ray_config,
    logdir=str(log_dir),
    eval_interval=original_eval_interval,
)

built = False

try:
    # Costruisce la DQN, senza addestrarla.
    algorithm.build()
    built = True

    iteration = algorithm.last_iteration()
    print("Iterazione iniziale:", iteration, flush=True)

    if iteration != 0:
        raise RuntimeError(
            f"Attesa iterazione 0, trovata {iteration}"
        )

    saved_path = algorithm.save_checkpoint(
        manual=True,
        path=str(CHECKPOINT_DIR)
    )

    print("CHECKPOINT INIZIALE SALVATO:", saved_path, flush=True)
    print("Nessuna iterazione di training eseguita.", flush=True)

finally:
    if built:
        algorithm.stop()
