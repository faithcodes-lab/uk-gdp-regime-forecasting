"""Validation rules for raw ONS GDP data.

GDP arrives as a date and a growth value. These rules check the basics that
the date is a date, the value is a number, and neither is missing and
also that the growth rate falls between -25% and +25%. That range is
deliberately wide so the COVID crash of Q2 2020 (about -19%) passes as a
real observation rather than being rejected as an impossible value.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

ONS_GDP_RAW_SCHEMA = DataFrameSchema(
    {
        "date": Column(
            pa.DateTime,
            nullable=False,
            description="Observation date.",
        ),
        "gdp_growth": Column(
            float,
            checks=Check.in_range(-25.0, 25.0),
            nullable=False,
            description=(
                "UK GDP quarter-on-quarter growth rate (%). "
                "Bound of -25 accommodates the COVID Q2 2020 observation."
            ),
        ),
    },
    strict=False,
    coerce=False,
)
