"""Validation rules for raw FRED downloads.

The four FRED series: Brent crude, the UK 10-year gilt yield, and the
OECD business and consumer confidence indices sit on very different
scales, so one shared rulebook can't sensibly range-check them all. These
rules therefore only check the basics: the date is a date, the value is a
number, and neither is missing. The per-series range checks happen later,
in processed_quarterly.py.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

RAW_FRED_SCHEMA = DataFrameSchema(
    {
        "date": Column(
            pa.DateTime,
            nullable=False,
            description="Observation date.",
        ),
        "value": Column(
            float,
            nullable=False,
            description=(
                "Observation value. Per-series range checks happen at the "
                "processed stage; this schema enforces type and non-null only."
            ),
        ),
    },
    strict=False,
    coerce=False,
)
