from collections import OrderedDict

import pandas as pd
import pandera as pa
from pandera.typing import Series


class LibgenRecords(pa.DataFrameModel):
    authors: Series[str] = pa.Field(nullable=True, alias="author(s)")
    series: Series[str] = pa.Field(nullable=True)
    title: Series[str] = pa.Field()
    language: Series[str] = pa.Field()
    file: Series[str] = pa.Field()
    mirror1: Series[str] = pa.Field()
    mirror2: Series[str] = pa.Field()
    md5: Series[str] = pa.Field()
    topic: Series[str] = pa.Field()
    extension: Series[str] = pa.Field()
    size: Series[str] = pa.Field()

    class Config:
        coerce = True


# Usage
# df = pd.read_csv('path_to_your_file.csv')
# BookDataSchema.validate(df)


class LibgenDataFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return LibgenDataFrame

    def validate(self):
        return LibgenRecords.validate(self)

    def to_ordered_dict(self):
        """Converts a pandas DataFrame into an ordered dictionary."""
        ordered_dict = OrderedDict()
        for index, row in self.iterrows():
            ordered_dict[index] = row.to_dict()
        return ordered_dict

    def to_polars(self) -> "PolarsDataFrame":
        """Convert results field to polars dataframe."""
        try:
            from polars import (
                from_pandas,  # pylint: disable=import-outside-toplevel
            )
        except ImportError as exc:
            raise ImportError(
                "Please install polars: `poetry add polars` to use this function."
            ) from exc

        return from_pandas(self.reset_index())
