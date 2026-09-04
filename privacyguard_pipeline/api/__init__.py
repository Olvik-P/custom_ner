"""HTTP API subpackage for PrivacyGuard Pipeline.

Exposes PrivacyGuardPipeline.process() (and, when the "pdf" extra is
installed, PDFAnonymizer) over an authenticated HTTP interface, as an
alternative entry point to the existing CLI/programmatic API.

Requires the optional "api" dependency group (fastapi, uvicorn,
python-multipart). This subpackage is never imported by the rest of
privacyguard_pipeline's top-level surface, so a missing "api" extra
only breaks running the server itself, not the base import.
"""

from __future__ import annotations
