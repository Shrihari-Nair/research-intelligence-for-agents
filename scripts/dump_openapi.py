"""Write the FastAPI OpenAPI schema for the frontend type generator: python -m scripts.dump_openapi"""
import json
from pathlib import Path

from research_os.api.app import create_app

if __name__ == "__main__":
    schema = create_app(service=object()).openapi()  # routes only; no model loading
    Path("frontend/openapi.json").write_text(json.dumps(schema, indent=2))
    print("wrote frontend/openapi.json", len(schema["paths"]), "paths")
