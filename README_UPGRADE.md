# Football Predictor model upgrade

This overlay replaces `src/model.py`, `src/backtest.py`, and `src/tune_model.py`, and adds `tests/test_model.py`.

## What changes

- Adds optional xG blending. `xg_weight=0.0` is the goals-only baseline; `0.25`, `0.5`, etc. blend xG with goals. If a particular xG value is missing, the model falls back to actual goals for that side.
- Uses 365 days as the default recency half-life, because earlier validation selected it over 180 days. The new tuner revisits this choice alongside xG weight.
- Extends the score grid to 0–12 goals per side, reducing probability mass lost from the previous 0–8 truncation.
- Joins `match_features` into the backtest and tuning loaders.
- Adds `--league` to both scripts, so the same workflow can be reused for the other leagues later.
- Keeps all matches from the same calendar date out of training until that date has been evaluated.
- Adds small unit tests for the model and its probability invariants.

## Install the overlay

From the project root, extract this ZIP into your repository root, allowing it to replace the three matching files and add `tests/test_model.py`.

## Validate before running a full tuning job

```bash
python -m py_compile src/model.py src/backtest.py src/tune_model.py tests/test_model.py
python -m unittest discover -s tests -v
```

## Tune Premier League parameters

```bash
python -m src.tune_model --league "Premier League"
```

The tuner chooses half-life and xG blend weight using the earlier validation period, then compares the selected model to a goals-only control on the later period. Lower Brier score and lower log loss are better. Adopt xG only if the results support it.

The later period has already been examined in earlier experiments, so treat it as a comparative holdout rather than a perfectly untouched research holdout. For a stronger final check, record the chosen settings and evaluate them prospectively on future 2026/27 results without retuning.

## Backtest one chosen configuration

```bash
python -m src.backtest --league "Premier League" --half-life-days 365 --xg-weight 0.5
```

Replace `365` and `0.5` with settings selected by validation. Until the tuner reports a useful xG gain, the backtest default remains goals-only (`--xg-weight 0`).

## Important integration note

This overlay makes `backtest.py` and `tune_model.py` load xG from `match_features`. Your current `predict.py` also needs to join `match_features` into the historical training query and instantiate `FootballModel` with the settings selected by the tuner before production predictions use xG. Do that after we have the tuning results, rather than enabling an unvalidated xG weight now.

## Next: the five major leagues

Once the Premier League comparison is complete, sync the other leagues, verify each exact `league_name` stored by the provider, then run the tuner separately for each league. Keep evaluations within each league for now; a cross-league team-strength model would need additional design.
