"""Check account capabilities without inference or access to provider billing settings."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def status(url, headers, select):
    try:
        with urlopen(Request(url, headers=headers), timeout=15) as response:
            payload = json.load(response)
        return {"status": "verified", **select(payload)}
    except HTTPError as error:
        return {"status": "unavailable", "http": error.code}
    except (KeyError, TypeError, ValueError, URLError, TimeoutError):
        return {"status": "unavailable"}


def main():
    snapshot = {"observed_at": datetime.now(timezone.utc).isoformat()}
    google = os.getenv("GEMINI_API_KEY")
    router = os.getenv("OPENROUTER_API_KEY")
    snapshot["google_models"] = status(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=1000",
        {"x-goog-api-key": google},
        lambda p: {"gemma_models_visible": sorted(x["name"] for x in p["models"]
                 if x["name"] in {"models/gemma-4-31b-it", "models/gemma-4-26b-a4b-it"}),
                   "billing_tier": "not_reported_by_models_api"}
    ) if google else {"status": "unconfigured"}
    snapshot["openrouter_key"] = status(
        "https://openrouter.ai/api/v1/key", {"Authorization": "Bearer " + router},
        lambda p: {"free_tier": p["data"].get("is_free_tier"),
                   "daily_free_requests_remaining": "not_reported_by_key_api"}
    ) if router else {"status": "unconfigured"}
    snapshot["asknews_credential_configured"] = bool(os.getenv("ASKNEWS_API_KEY") or
        (os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET")))
    snapshot["asknews_entitlement"] = "not_verified"
    Path("provider-status.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps(snapshot))


if __name__ == "__main__":
    main()
