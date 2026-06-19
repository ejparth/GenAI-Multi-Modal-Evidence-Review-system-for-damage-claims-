from pydantic import BaseModel
from typing import List

class Claim(BaseModel):
    object_type: str
    claimed_damage: str
    damage_location: str
    severity_claimed: str
    incident_description: str
    required_evidence: List[str]