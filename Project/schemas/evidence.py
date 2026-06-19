from pydantic import BaseModel
from typing import List

class Damage(BaseModel):
    type: str
    location: str
    severity: str

class ImageEvidence(BaseModel):
    object_type: str
    visible_damages: List[Damage]
    image_quality: str
    confidence: float