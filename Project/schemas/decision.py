from pydantic import BaseModel
from typing import List

class FinalDecision(BaseModel):
    verdict: str
    confidence: float
    support_score: int
    reasoning: str
    missing_evidence: List[str]