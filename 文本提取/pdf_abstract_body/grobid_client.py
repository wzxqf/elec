"""Small REST client for GROBID full-text PDF processing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class GrobidResult:
    ok: bool
    tei_xml: str = ""
    error: str = ""


class GrobidClient:
    def __init__(self, base_url: str = "http://localhost:8070", timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def is_available(self) -> bool:
        try:
            import requests

            response = requests.get(f"{self.base_url}/api/isalive", timeout=10)
            return response.ok and response.text.strip().lower() in {"true", "ok", "alive"}
        except Exception:
            return False

    def process_fulltext_document(self, pdf_path: Path) -> GrobidResult:
        try:
            import requests
        except Exception as exc:
            return GrobidResult(ok=False, error=f"requests unavailable: {exc}")

        endpoint = f"{self.base_url}/api/processFulltextDocument"
        data = {
            "consolidateHeader": "0",
            "consolidateCitations": "0",
            "includeRawCitations": "0",
            "includeRawAffiliations": "0",
        }
        try:
            with pdf_path.open("rb") as handle:
                files = {"input": (pdf_path.name, handle, "application/pdf")}
                response = requests.post(endpoint, data=data, files=files, timeout=self.timeout)
            if response.status_code != 200:
                return GrobidResult(
                    ok=False,
                    error=f"grobid HTTP {response.status_code}: {response.text[:300]}",
                )
            if not response.text.strip():
                return GrobidResult(ok=False, error="grobid returned empty TEI")
            return GrobidResult(ok=True, tei_xml=response.text)
        except Exception as exc:
            return GrobidResult(ok=False, error=f"grobid request failed: {exc}")
