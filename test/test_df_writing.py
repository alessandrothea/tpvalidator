#!/usr/bin/env python


from rich import print

import uproot, awkward as ak, numpy as np, pandas as pd

# Flat dataframe: one row per hit, multiple rows per event
df_flat = pd.DataFrame({
    "run": [100]*10,
    "subrun": [5]*10,
    "event": [0, 0, 0, 1, 1, 2, 2, 2, 2, 3],
    "hit_x": np.random.uniform(0, 10, 10).astype(np.float32),
    "hit_e": np.random.uniform(0,  1, 10).astype(np.float32),
})

keys = ['run', 'subrun', 'event']

print(df_flat)


schema = { c:(t if c in keys else f'var * {t.name}') for c, t in df_flat.dtypes.items()}

print(schema)

grouped     = df_flat.groupby("event", sort=False)
counts      = grouped.size().to_numpy()
offsets     = np.zeros(len(counts) + 1, dtype=np.int64)
np.cumsum(counts, out=offsets[1:])

print(grouped)
print(counts)
print(offsets)

def make_jagged(series):
    flat = series.to_numpy()   # view into existing memory if possible
    return ak.Array(
        ak.contents.ListOffsetArray(
            ak.index.Index64(offsets),
            ak.contents.NumpyArray(flat)
        )
    )

# branch_data = {
#     "event":  grouped["event"].first().to_numpy(dtype=np.int32),
#     "hit_x":  make_jagged(df_flat["hit_x"]),
#     "hit_e":  make_jagged(df_flat["hit_e"]),
# }


branch_data = {}

for c, t in schema.items():
    if c in keys:
        d = grouped[c].first().to_numpy(dtype=t)
    else:
        continue
        d = make_jagged(df_flat[c])

    branch_data[c] = d

print(branch_data)

# --- write ---
with uproot.recreate("output.root") as f:
    # f.mkrntuple("ntuple", schema)
    # f["ntuple"].extend(branch_data)
    f["ntuple"] = branch_data
    f["ntuple"].extend(branch_data)

