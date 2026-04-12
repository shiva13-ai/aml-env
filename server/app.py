import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from openenv.core.env_server import create_fastapi_app
from models import AMLAction, AMLObservation
from aml_environment import AMLEnvironment

app = create_fastapi_app(AMLEnvironment, AMLAction, AMLObservation)