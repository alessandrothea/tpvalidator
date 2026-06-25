#!/usr/bin/env python


from rich import print
import awkward as ak
import pandas as pd
import uproot


def rebuild_entry_index(df: pd.DataFrame, keys: list[str]):
    # combine columns into a single tuple-key, rank densely to get 0-based entry index
    entry_keys = df[keys].apply(tuple, axis=1)

    df.index = pd.MultiIndex.from_arrays(
        [
            entry_keys.rank(method="dense").astype(int) - 1,  # entry
            df.groupby(entry_keys).cumcount(),                 # subentry
        ],
        names=["entry", "subentry"],
    )

    return df


with uproot.open('radbkg_10.root') as f:
    arr = f['taFinder/ta_win_stats'].arrays(library='ak')
    print(arr)

    df = ak.to_dataframe(arr)

    print(df)

    keys = ['event_uid', 'run', 'subrun', 'event']


    # entry_cols = ['event_uid', 'run', 'subrun', 'event']   # define entry by this combination of columns

    # # combine columns into a single tuple-key, rank densely to get 0-based entry index
    # entry_keys = df[entry_cols].apply(tuple, axis=1)

    # print(entry_keys)

    # df.index = pd.MultiIndex.from_arrays(
    #     [
    #         entry_keys.rank(method="dense").astype(int) - 1,  # entry
    #         df.groupby(entry_keys).cumcount(),                 # subentry
    #     ],
    #     names=["entry", "subentry"],
    # )

    print("-------")
    df = rebuild_entry_index(df, keys)

    print(df)