"""Reproduce the ALFRED real-time bulk export; it uses no API key or inference."""
from datetime import date
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

URL = "https://alfred.stlouisfed.org/series/downloaddata?seid=UNRATE"
HERE = Path(__file__).resolve().parent
DATA = HERE / "unrate_realtime.csv"


def fetch(cutoff=date(2025, 12, 31)):
    with urlopen(URL, timeout=60) as response:
        form = response.read().decode()
    dates = re.findall(r'<option value="(\d{4}-\d{2}-\d{2})"', form)
    dates = [d for d in dates if d <= cutoff.isoformat()]
    if len(dates) < 500:
        raise RuntimeError("Missing archive release dates")
    fields = {
        "form[units]": "lin", "form[obs_start_date]": "1970-01-01",
        "form[obs_end_date]": cutoff.replace(day=1).isoformat(), "form[selected_vintage_dates][]": dates,
        "form[entered_vintage_dates]": "", "form[file_type]": "1",
        "form[file_format]": "csv", "form[download_data]": "",
    }
    with urlopen(Request(URL, data=urlencode(fields, doseq=True).encode()), timeout=60) as response:
        with zipfile.ZipFile(io.BytesIO(response.read())) as archive:
            return archive.read("obs._by_real-time_period.csv")


if __name__ == "__main__":
    raw = fetch()
    print("source", URL, "sha256", hashlib.sha256(raw).hexdigest())
    if raw != DATA.read_bytes():
        sys.exit("Upstream archive changed. Preserve the frozen CSV for historical reproduction.")
