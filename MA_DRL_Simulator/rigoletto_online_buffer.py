from collections import deque
import math
from rigoletto_validator import validate_state, validate_action

class RigolettoOnlineNStepBuffer:
    """
    costruisce incrementalmente le esperienze per il training.
    Trattiamo ogni DataBlock comme un episodio indipendente.
    """

    def __init__(self, n_step=7, gamma=0.99):

        if n_step < 1 or not (0 < gamma <= 1):
            raise ValueError("Parametri n-step non validi")

        self.n_step = n_step
        self.gamma = gamma
        self.pending = {}
        self.episode_ids = {}
        self.steps = {}
        self.last_next_state = {}
        self.closed = set()
        self.emitted = 0
        self.discarded_truncated = 0

    def _convert(self, block, window):
        """
        trasforma più transazioni in una singola esperienza, fino a n step definiti da un iperparametro
        serve per valutare la reward totale fino a n-step
        """

        last = window[-1]

        #ammettiamo una finestra piu piccola se nel frattempo arriva a destinazione, in caso contrario errore
        if len(window) < self.n_step and not last["done"]:
            raise ValueError("Finestra incompleta non terminale")

        #calcolo della reward pesata aggregata
        reward = sum((self.gamma ** k) * item["reward"]for k, item in enumerate(window))

        #restituisco l'esperienza n-step
        return {
            "block": block, #id datablock
            "episode_id": self.episode_ids[block], #id episodio
            "timestep": window[0]["timestep"],
            "observation": window[0]["state"],
            "action": window[0]["action"],
            "reward": reward,
            "next_observation": last["next_state"],
            "terminated": last["done"],
            "truncated": last["truncated"],
            "effective_n": len(window),
        }

    def add(self, block, state, action, reward, next_state, *, done=False, truncated=False,fail=False, timeout=False):
        """
        Riceve una transizione, le assegna all'episodio e aggiorna la coda
        dopodichè restituisce la lista delle esperienze n-step
        """
        #Conversione dei valori in strutture dati
        block = str(block)
        state = validate_state(state)
        next_state = validate_state(next_state)
        action = validate_action(action)
        reward = float(reward)
        done = bool(done)
        truncated = bool(truncated)
        if fail or timeout:
            raise ValueError("FAIL/TIMEOUT non ancora gestiti") #DA MODIFICARE

        if not math.isfinite(reward):
            raise ValueError("Reward non valida")

        #recupero il next state e ne verifico la consistenza con la traiettoria
        previous = self.last_next_state.get(block)

        if previous is not None and any(
            abs(a - b) > 1e-4
            for a, b in zip(previous, state)
        ):
            raise ValueError(f"Traiettoria discontinua: {block}")

        #inizializzazione strutture del prossimo episodio (nuovo datablock)
        if block not in self.pending:
            self.pending[block] = deque()
            self.episode_ids[block] = len(self.episode_ids)
            self.steps[block] = 0

        queue = self.pending[block]

        #memorizzo la transizione nella coda
        queue.append({
            "timestep": self.steps[block],
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
            "truncated": truncated,
        })

        self.steps[block] += 1
        self.last_next_state[block] = next_state

        ready = []

        # Genera una finestra non appena ci sono n passi.
        while len(queue) >= self.n_step:
            ready.append(
                self._convert(block, list(queue)[:self.n_step])
            )
            queue.popleft()

        if done:
            while queue:
                ready.append(self._convert(block, list(queue)))
                queue.popleft()
            del self.pending[block]
            self.closed.add(block)

        elif truncated:
            #scarto transizioni che non generano una struttura completa
            self.discarded_truncated += len(queue)
            queue.clear()
        #puliamo le code
            del self.pending[block]
            self.closed.add(block)

        self.emitted += len(ready)
        return ready

    def statistics(self):
        open_pending = sum(len(q) for q in self.pending.values())
        return {
            "experiences": self.emitted,
            "discarded_truncated": self.discarded_truncated,
            "open_pending": open_pending,
            "skipped_total": self.discarded_truncated + open_pending,
        }
