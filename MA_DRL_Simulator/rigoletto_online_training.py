import random
from collections import deque
from rigoletto_online_buffer import RigolettoOnlineNStepBuffer
from rigoletto_rl4cc_client import RigolettoRL4CCClient
from rigoletto_validator import validate_state, validate_action
import csv
import json
import math
from pathlib import Path
from datetime import datetime

"""
decisore delle azioni da intraprendere.
Parte da una random policy, registra le transizioni in csv,
conserva le esperienze in un buffer condiviso e invia a RLCC il batch per il training.
"""
class RigolettoOnlineTrainer:
    #Tupla dei campi registrati
    CSV_FIELDS = (
        "block",
        "action",
        "state",
        "next_state",
        "reward",
        "done",
        "fail",
        "timeout",
        "truncated",
        "finish"
    )

    def __init__(self, seed=36):
        self.rng = random.Random(seed)
        self.mode = "random" #definiamo la random policy
        self._csv_file = None
        self._csv_writer = None

        # Buffer n-step e replay per il futuro training online.
        self.nstep_buffer = RigolettoOnlineNStepBuffer(
            n_step=7, gamma=0.99
        )
        self.replay = deque(maxlen=3000)
        self.nstep_generated = 0

        self.online_training_enabled = False #inizialmente false per tenere un training offline, da implementare in futuro il training online
        self.client = None
        self.batch_size = 128       #esperienze per batch
        self.train_every = 128
        self.target_update_every = 500
        self.eligible_generated = 0     #numero degli n-step validi
        self.next_training_at = 128     #delta di esperienze per il prossimo training
        self.last_target_update_at = 0
        self.train_calls = 0    #numero di training riusciti
        self.checkpoint_every = 100     #frequenza di salvataggio del checkpoint
        self.last_checkpoint_path = None    #ultimo checkpoint salvato
        self.invalid_dqn_actions = 0
        self.last_training_result = None    #risposta dell'ultimo training
    """
    Funzione che mi permette di selezionare azioni valide, cioè che corrispondono
    effettivamente a dei satelliti disponibili
    """
    def get_valid_actions(self, observation):
        observation = validate_state(observation)
        valid_actions = []

        for action in range(3):
            #individuo le coordinate dei 3 satelliti candidati
            idx = 2 + action * 2
            lat = float(observation[idx])
            lon = float(observation[idx + 1])

            if lat < -189.0 and lon < -189.0:
                continue

            valid_actions.append(action)

        return valid_actions


    """
    Seleziona l'azione in funzione alla modalità attiva
    """
    def select_action(self, observation):
        valid_actions = self.get_valid_actions(observation)

        if not valid_actions:
            return None

        #con la mod Random seleziono a caso l'azione da intraprendere
        if self.mode == "random":
            return self.rng.choice(valid_actions)

        #con la mod DQN mi baso sulla polici di RL4CC
        if self.mode == "dqn":
            if self.client is None:
                raise RuntimeError("Client RL4CC non configurato")

            action = self.client.get_action(
                observation,
                explore=self.online_training_enabled
            )
#DA MODIFICARE!!!!!!!!!!!!!
            if action not in valid_actions:
                self.invalid_dqn_actions += 1
                # Ripiego esplicito, da contabilizzare nei test.
                return self.rng.choice(valid_actions)

            return action

        raise ValueError(f"Modalità sconosciuta: {self.mode}")

    def record_transition(
        self, block_id, state, action, reward,
        next_state, terminated, truncated,
        *, fail=False, timeout=False, output_path=None
    ):

        state = validate_state(state)
        next_state = validate_state(next_state)
        action = validate_action(action)

        if not math.isfinite(float(reward)):
            raise ValueError("Reward non valida")

        if self._csv_writer is None:
            if output_path is None:
                filename = (
                    "rigoletto_online_10d_"
                    + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    + ".csv"
                )
                output_path = (
                    Path(__file__).resolve().parent
                    / "Results"
                    / filename
                )

            self.csv_path = Path(output_path)
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)

            self._csv_file = self.csv_path.open(
                "x", newline="", encoding="utf-8-sig"
            )

            self._csv_writer = csv.DictWriter(
                self._csv_file,
                fieldnames=self.CSV_FIELDS
            )
            self._csv_writer.writeheader()

        if terminated:
            finish = "DONE"
        elif fail:
            finish = "FAIL"
        elif timeout:
            finish = "TIMEOUT"
        elif truncated:
            finish = "TRUNCATED"
        else:
            finish = "CONTINUE"

        # Produce esperienze n-step dalle transizioni reali.
        ready = self.nstep_buffer.add(
            block=block_id,
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=terminated,
            truncated=truncated,
            fail=fail,
            timeout=timeout
        )

        self.replay.extend(ready)
        self.nstep_generated += len(ready)

        self._csv_writer.writerow({
            "block": str(block_id),
            "action": int(action),
            "state": json.dumps(state),
            "next_state": json.dumps(next_state),
            "reward": float(reward),
            "done": int(bool(terminated)),
            "fail": int(bool(fail)),
            "timeout": int(bool(timeout)),
            "truncated": int(bool(truncated)),
            "finish": finish
        })

        self._csv_file.flush()

        if self.online_training_enabled:
            self._maybe_train(ready)
    """
    Configuro RL4CC per il training online
    """
    def enable_online_training(
        self,
        *,
        base_url="http://127.0.0.1:8001",
        client=None,
        batch_size=128,
        train_every=128,
        target_update_every=500
    ):

        if self.nstep_generated:
            raise RuntimeError(
                "Attivare il training prima della raccolta dati"
            )

        if min(batch_size, train_every, target_update_every) < 1:
            raise ValueError("Intervalli non validi")

        self.client = (
            client if client is not None
            else RigolettoRL4CCClient(base_url=base_url)
        )
        self.batch_size = batch_size
        self.train_every = train_every
        self.target_update_every = target_update_every
        self.next_training_at = train_every
        self.online_training_enabled = True

    def _maybe_train(self, ready):
        # Le esperienze troncate sono  escluse.
        self.eligible_generated += sum(
            not exp["truncated"] for exp in ready
        )

        if self.eligible_generated < self.next_training_at:
            return

        eligible = [
            exp for exp in self.replay
            if not exp["truncated"]
        ]

        if len(eligible) < self.batch_size:
            return

        batch = self.rng.sample(eligible, self.batch_size)

        update_target = (
            self.eligible_generated - self.last_target_update_at
            >= self.target_update_every
        )

        result = self.client.train_batch(
            batch,
            gradient_steps=1,
            update_target=update_target
        )

        # Aggiorna i contatori solo dopo una risposta positiva.
        self.train_calls += 1
        self.last_training_result = result
        self.next_training_at += self.train_every

        if update_target:
            self.last_target_update_at = self.eligible_generated

        #dopo il primo update passo alla mod DQN
        self.mode = "dqn"

        if self.train_calls % self.checkpoint_every == 0:
            self._save_checkpoint_and_progress()

    def _save_checkpoint_and_progress(self):
        """
        Salva la DQN e i contatori del training online.
        """

        checkpoint_path = self.client.save_checkpoint()

        self.last_checkpoint_path = checkpoint_path
        """
        etichetta la fase del training per salvarla nel checkpoint:
        fase 1: impara a raggiungere il percorso
        fase 2: ottimizza il percorso, premiando i percorsi più corti
        fase 3: favorisce le decisioni che convergono su Dijkstra
        """

        if self.train_calls >= 3900:
            curriculum_phase = 3
        elif self.train_calls >= 750:
            curriculum_phase = 2
        else:
            curriculum_phase = 1

        progress = {
            "saved_at": datetime.now().isoformat(),
            "checkpoint_path": checkpoint_path,
            "train_calls": self.train_calls,
            "eligible_generated": self.eligible_generated,
            "nstep_generated": self.nstep_generated,
            "next_training_at": self.next_training_at,
            "last_target_update_at": self.last_target_update_at,
            "curriculum_phase": curriculum_phase,
            "mode": self.mode
        }

        output_dir = (
            Path(__file__).resolve().parent
            / "Results"
            / "online_checkpoints"
        )

        output_dir.mkdir(parents=True, exist_ok=True)

        progress_path = (
            output_dir
            / f"progress_{self.train_calls:06d}.json"
        )

        temp_path = progress_path.with_suffix(".tmp")

        temp_path.write_text(
            json.dumps(progress, indent=2),
            encoding="utf-8"
        )

        temp_path.replace(progress_path)

        print(
            f"RIGOLETTO CHECKPOINT | "
            f"updates={self.train_calls} | "
            f"fase={curriculum_phase} | "
            f"checkpoint={checkpoint_path}",
            flush=True
        )
    def close(self):
        if self._csv_file is not None:
            self._csv_file.close()
