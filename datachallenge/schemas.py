from typing import TypedDict


class TripletRow(TypedDict):
    MONTH: str    # 'YYYY-MM'
    PEAKID: int   # 0=OFF-Peak, 1=ON-Peak
    EID: str


class CostRow(TypedDict):
    MONTH: str
    PEAKID: int
    EID: str
    C: float      # always positive after abs()


class PriceRow(TypedDict):
    DATETIME: str
    PEAKID: int
    EID: str
    PRICEREALIZED: float  # always positive after abs()


class ProfitRow(TypedDict):
    MONTH: str
    PEAKID: int
    EID: str
    COST: float
    PRICE: float
    PROFIT: float


class PredictionRow(TypedDict):
    MONTH: str
    PEAKID: int
    EID: str
    PREDICTED_PROFIT: float
