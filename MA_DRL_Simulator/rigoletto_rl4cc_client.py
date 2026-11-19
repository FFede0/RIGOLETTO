import json
import urllib.request
import urllib.error
from rigoletto_validator import validate_state, validate_action

class RigolettoRL4CCClient:
    """
    Componente intermediario tra RIGOLETTO e RL4CC.
    Permette di inoltrare e ricevere richieste HTTP dal server
    """

    def __init__(
        self,
        base_url="http://127.0.0.1:8001",
        timeout=20
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout


    """
    Richiede un'azione sulla base dello stato corrente
    """
    def get_action(self, observation, explore=True):
        observation = validate_state(observation)

        payload = {
            "observation": {
                "obs": observation
            },
            "agent_parameters": {
                "explore": bool(explore)
            }
        }

        request = urllib.request.Request(
            self.base_url + "/action",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(
                request, timeout=self.timeout
            ) as response:
                result = json.load(response)

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(
                f"RL4CC HTTP {exc.code}: {detail}"
            ) from exc

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"Servizio RL4CC non raggiungibile: {exc}"
            ) from exc

        return validate_action(result)

    """
    funzione che organizza i dati per l'inoltro al server
    """
    def build_train_payload(
        self, experiences, *, gradient_steps=1,
        update_target=False
    ):
        if not experiences:
            raise ValueError("Batch di training vuoto")

        if gradient_steps < 1:
            raise ValueError("gradient_steps non valido")

        for exp in experiences:
            validate_state(exp["observation"])
            validate_state(exp["next_observation"])
            validate_action(exp["action"])
            if (exp["effective_n"] < 7 and not exp["terminated"]):
                raise ValueError("Finestra n-step incompleta non terminale")

        return {
            "observations": [
                e["observation"] for e in experiences
            ],
            "actions": [
                e["action"] for e in experiences
            ],
            "rewards": [
                e["reward"] for e in experiences
            ],
            "next_observations": [
                e["next_observation"] for e in experiences
            ],
            "terminateds": [
                e["terminated"] for e in experiences
            ],
            "truncateds": [
                e["truncated"] for e in experiences
            ],
            "episode_ids": [
                e["episode_id"] for e in experiences
            ],
            "timesteps": [
                e["timestep"] for e in experiences
            ],
            "gradient_steps": gradient_steps,
            "update_target": bool(update_target),
        }


    """
    Funzione che richiede l'aggiornamento dei pesi al server, sulla base delle esperienze selezionate nel minibatch
    """
    def train_batch(
        self, experiences, *, gradient_steps=1,
        update_target=False
    ):
        payload = self.build_train_payload(
            experiences,
            gradient_steps=gradient_steps,
            update_target=update_target
        )

        request = urllib.request.Request(
            self.base_url + "/train",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=max(self.timeout, 180)
            ) as response:
                result = json.load(response)

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(
                f"RL4CC train HTTP {exc.code}: {detail}"
            ) from exc

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"RL4CC non raggiungibile: {exc}"
            ) from exc

        # Verifico che RL4CC confermi il completamento del training.
        if not isinstance(result, dict) or result.get("status") != "Training completed.":
            raise RuntimeError(
                f"Risposta training inattesa: {result}"
            )

        return result

    def save_checkpoint(self):
        """Salva sul server la DQN attualmente addestrata."""

        payload = {"manual": False}

        request = urllib.request.Request(
            self.base_url + "/checkpoint",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            with urllib.request.urlopen(
                request, timeout=max(self.timeout, 180)
            ) as response:
                result = json.load(response)

        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(
                "utf-8", errors="replace"
            )
            raise RuntimeError(
                f"RL4CC checkpoint HTTP {exc.code}: {detail}"
            ) from exc

        except urllib.error.URLError as exc:
            raise ConnectionError(
                f"RL4CC non raggiungibile: {exc}"
            ) from exc

        if not isinstance(result, dict) or not result.get("checkpoint_path"):
            raise RuntimeError(
                f"Risposta checkpoint inattesa: {result}"
            )

        return result["checkpoint_path"]
