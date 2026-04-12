from typing import Any
from openenv.core.env_client import EnvClient
from models import AMLAction, AMLObservation


class AMLEnv(EnvClient):
    action_class = AMLAction
    observation_class = AMLObservation

    def _parse_result(self, data: dict) -> Any:
        obs_data = data.get("observation", data)
        return type("StepResult", (), {
            "observation": AMLObservation(**obs_data),
            "reward": data.get("reward", 0.0),
            "done": data.get("done", False),
            "info": data.get("info", {}),
        })()

    def _parse_state(self, data: dict) -> Any:
        return data

    def _step_payload(self, action: AMLAction) -> dict:
        return action.model_dump()