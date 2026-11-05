# Packing Lab - Football Analytics Dashboard

Everything (code + processed data) is in this folder.

    py -3.13 -m venv venv313
    venv313\Scripts\activate
    pip install -r requirements.txt
    streamlit run app.py

Optional (faster loading): python convert_to_parquet.py

## Light / dark theme
The app follows your device setting. To switch: three dots (top right) > Settings > Theme > Light / Dark / Use system setting.
Charts use transparent backgrounds and grey gridlines, so they read correctly on both; the football pitch keeps its green background in both themes.

## Pages (27: Home + 26)
Home (app.py): KPIs, navigation, "packing explained" drawing, data coverage, glossary

Look at the pitch      1 Pass & Carry Map | 2 Freeze-Frame Viewer | 3 Team Shape & Network
Players & teams        4 Player Leaderboards (+ events-only scouting tab) | 5 Team Analysis | 6 Game State
Machine learning       7 xPack | 8 Pass Value (xPass) | 9 Pass Types (KMeans) | 10 Expected Threat (xT) | 11 Player Archetypes (PCA + KMeans)
Results & context      12 Packing -> Shots & Goals | 13 Match Outcome Model | 14 Competitions & Leagues (freeze-frame + events-only tabs)
Space, shape, pressure 15 Pitch Control & Space (Voronoi) | 16 Defensive Lines | 20 Receivers & Targets | 21 Pressure Resistance
Sequences & timing     17 Possession Sequences | 18 Set Pieces | 19 Match Momentum & Tempo
Defending & game mgmt  22 Pressing & Counter-pressing | 23 Shooting & Finishing (goal-mouth, post-shot xG) | 24 Expected League Tables (simulator) | 25 Chain Value | 26 Substitutions & Tactical Shifts

Pitch pages draw a full pitch (stripes, boxes, arcs, goals with nets and posts), real pass arrows,
and label players with their position (GK, CB, RB, CM, CAM, LW, CF ...).

## Files
- pitch.py: pitch drawing, arrows, labelled players, position abbreviations.   ui.py: theme, header, cards.
- packing.py: packing / line-breaking maths.   utils.py: data loading and sidebar filters.   views/: tab content for merged pages.
- models.py: shared cached xPass and post-shot xG models.   chains.py: shot-to-chain linking.   data/processed: actions (with xT and opponent-shape columns), possessions, shots, minutes, matches, game states, frames_top, xt_grid,
  team_match_metrics, player_metrics, league_shots.   See DATA_SOURCES.md for licences and attribution.

## Models used
xPack and xPass (gradient boosting, match-grouped cross-validation) | Expected Threat (Markov chain on a 16x12 grid) |
Pass types (MiniBatchKMeans) | Player archetypes (PCA + KMeans + cosine similarity) | Packing -> shots (logistic/linear regression, bootstrap CIs) |
Match outcome (logistic regression vs gradient boosting, grouped CV) | Wyscout xG (gradient boosting, held-out matches) |
Pitch control (nearest-player Voronoi via KD-tree) | Pressure resistance (xPass over expected) | Opponent-shape features from freeze frames (defensive-line height, compactness).

## Limits
Packing needs StatsBomb 360 freeze frames (418 matches). AFCON 2023 has none. Wyscout and the 2015/16 StatsBomb leagues are analysed with event proxies only.

## Extra data files (added for pages 22-26)
defensive_events (348k pressures, recoveries, duels ...), counterpress_losses (102k losses with counter-press flags), regains (57k with what happened next),
pressing_team_match, shots_detail (51k shots incl. goal-mouth end location), league_xpoints (per-match win/draw/loss probabilities from shot-level xG),
lineup_stints (substitutions, positional switches), cards, tactical_shifts.
