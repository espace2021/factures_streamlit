from pydantic import BaseModel, Field
from typing import Optional, List

class InvoiceLine(BaseModel):
    product_ref: Optional[str] = Field(None, description="Référence du produit")
    description: Optional[str] = Field(None, description="Libellé sur la facture")
    quantity: Optional[float] = None
    unit_price: Optional[float] = Field(None, description="Prix facturé")

class InvoiceData(BaseModel):
    invoice_number: Optional[str] = None
    invoice_date: Optional[str] = None
    supplier_name: Optional[str] = None
    client_name: Optional[str] = None
    total_amount: Optional[float] = None
    currency: Optional[str] = None
    source_file: Optional[str] = None
    lines: List[InvoiceLine] = Field(default_factory=list)