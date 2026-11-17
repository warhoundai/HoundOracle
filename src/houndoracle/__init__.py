"""HoundOracle: agent-facing search and evidence verification."""

from .engine import Oracle
from .models import Claim, Decision, DocumentInput, Verification
from .store import Store

__version__ = "0.1.0.dev1"
__all__ = ["Claim", "Decision", "DocumentInput", "Oracle", "Store", "Verification"]
