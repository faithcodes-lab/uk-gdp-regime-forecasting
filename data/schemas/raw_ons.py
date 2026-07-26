"""Validation rules for raw ONS downloads (every series except GDP).

GDP has its own rules in raw_ons_gdp.py. Every other ONS series arrives
as just a date and a value, so these rules only check the basics: the
date is a date, the value is a number, and neither is missing. They do
not check whether the values are sensible which happens later, in
processed_quarterly.py.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema

RAW_ONS_SCHEMA = DataFrameSchema(
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
