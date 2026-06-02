from typing import Dict, Optional

import pandas as pd
from rich.table import Table


def dataframe_to_rich_table(
    pandas_dataframe: pd.DataFrame,
    show_index: bool = True,
    index_name: Optional[str] = None,
    formatters: Optional[Dict[str, str]] = {},
    column_titles: dict = {},
    **kwargs,
) -> Table:
    """Convert a pandas DataFrame to a rich Table."""
    rich_table = Table(**kwargs)
    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for row in pandas_dataframe.itertuples():
        t_row = [str(row[0])] if show_index else []
        r = row._asdict()
        for c in pandas_dataframe.columns:
            fmt = formatters.get(c, None)
            if callable(fmt):
                t_row.append(fmt(r[c]))
            elif fmt is not None:
                t_row.append(fmt.format(r[c]))
            else:
                t_row.append(str(r[c]))
        rich_table.add_row(*t_row)

    for c in rich_table.columns:
        if c.header in column_titles:
            c.header = column_titles[c.header]
            


    return rich_table
