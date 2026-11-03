"""Share the app fixtures defined in test_routes.py / test_ingest.py with other API test modules."""
from tests.test_api.test_ingest import app_env  # noqa: F401
from tests.test_api.test_routes import client  # noqa: F401
