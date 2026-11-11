# OSM point data (not shipped — too large for the repository)

`osm250mbinary32`: 250M 2D points, binary format —
little-endian uint32 count header, then `count` pairs of little-endian
uint32 coordinates (x, y). Loaded via `np.memmap` in `src/mvp/workloads.py`.

Source: OpenStreetMap (ODbL). Rebuild from a 2-column CSV of uint32 x,y
pairs with `./data/prepare.sh osm <csv>` (the format written there matches
`load_points_real` in `src/mvp/workloads.py`).

`queries_osm250m.csv` (shipped): 240 window queries in the same coordinate
space; the first 120 are the training split, the rest are test.
