"""Validation rules for raw Bank of England Bank Rate data.

The Bank Rate arrives as a date and a rate value. These rules check the
basics, the date is a date, the value is a number, and neither is
missing and also that the rate falls between 0% and 25%. That range is
a generous sanity bound: the highest UK Bank Rate on record is 17%
(November 1979) and the rate has never gone negative, so a value outside
0-25 almost certainly signals a data problem rather than a real rate.

Source: Bank of England, Official Bank Rate history (1694-present).
https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy/baserate.xls
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column, DataFrameSchema

BOE_RATE_RAW_SCHEMA = DataFrameSchema(
    {
        "date": Column(
            pa.DateTime,
            nullable=False,
            description="Observation date.",
        ),
        "bank_rate": Column(
            float,
            checks=Check.in_range(0.0, 25.0),
            nullable=False,
            description=(
                "Bank of England Bank Rate (%). Bound of 25 is a generous "
                "upper limit relative to historical UK policy rates."
            ),
        ),
    },
    strict=False,
    coerce=False,
)
