"""Fetch the official NeurIPS 2025 style and verify the report's expected file."""

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile
import hashlib


URL = "https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip"
EXPECTED_SHA256 = "67a960f35da6fecb5a0ce0202fa400459f4c8bb6c294f1a3fdd82fbbd39244c2"


def main():
    destination = Path(__file__).resolve().parents[1] / "report/manuscript/neurips_2025.sty"
    if destination.exists():
        data = destination.read_bytes()
        if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
            raise ValueError("Existing style differs from the report's expected version")
        print("Official report style already verified")
        return
    request = Request(URL, headers={"User-Agent": "COMP6704-report-style/0.1"})
    with urlopen(request, timeout=60) as response:
        archive = response.read()
    with ZipFile(BytesIO(archive)) as bundle:
        names = [name for name in bundle.namelist() if name.split("/")[-1] == "neurips_2025.sty"]
        if len(names) != 1:
            raise ValueError("Official archive must contain exactly one expected style file")
        data = bundle.read(names[0])
    if hashlib.sha256(data).hexdigest() != EXPECTED_SHA256:
        raise ValueError("Downloaded style checksum differs from the report's expected version")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(data)
    print("Official report style downloaded and verified")


if __name__ == "__main__":
    main()
