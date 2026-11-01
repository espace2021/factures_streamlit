from pydantic import BaseModel, Field
from typing import Optional

class Product(BaseModel):
    ref: str = Field(..., description="Référence unique du produit")
    name: str = Field(..., description="Nom du produit")
    unit_price: Optional[float] = Field(None, description="Prix catalogue HT")
    category: Optional[str] = None