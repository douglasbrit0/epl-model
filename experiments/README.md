# Experiment scripts

These scripts reproduce the experiments logged in `reports/experiments.md`. Run them from this
folder, after `python run_backtest.py` has written `reports/tuning.json` (E3 production tuning).

| Script | Purpose |
|---|---|
| `exp.py NAME OPTION OFF v1,v2,...` | E4–E6: one `DixonColesModel` option on a grid, on top of the per-season E3 config |
| `exp_dyn.py` | E7: dynamic random-walk ratings grid (slow: ~80 min on 8 cores) |
| `exp_dep.py` | E8: bivariate Poisson and Frank copula |
| `nested_opt.py GRID OFF OUT.csv` | Nested selection: season S uses the best value on seasons before S |
| `compare.py A.csv B.csv LABEL` | Paired per-season comparison with a block bootstrap by matchweek |

Examples:

```sh
python exp.py e4 home_shrink None None,3,10,30,100,300
python nested_opt.py e4_grid.parquet None e4.csv
python compare.py ../reports/walk_forward.csv e4.csv e4
```

Outputs (`*.parquet`, `*.csv`) are written here and are not committed.
