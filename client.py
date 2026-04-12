from openenv.core.env_client import EnvClient
from models import AMLAction, AMLObservation


class AMLEnv(EnvClient):
    """
    Client for the AML Compliance Officer environment.
    Usage:
        async with AMLEnv(base_url="https://shiva13-ai.hf.space") as env:
            result = await env.reset()
            result = await env.step(AMLAction(
                transaction_id=result.observation.transaction_id,
                decision="block",
                reasoning="High-risk country + PEP involved"
            ))
    """

    action_class = AMLAction
    observation_class = AMLObservation

    async def reset(self, task_name: str = "triage_basic"):
        return await super().reset(task_name=task_name)