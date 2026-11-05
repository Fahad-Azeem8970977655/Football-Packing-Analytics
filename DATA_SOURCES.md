# Data sources, licences and attribution

**Please keep this attribution in any report, slide deck or app that uses this data.**

| Data | Source | Licence / terms |
|---|---|---|
| StatsBomb events, lineups, matches, 360 freeze frames (Bundesliga 2023/24, La Liga 2020/21, Ligue 1 2021/22 and 2022/23, Euro 2020/2024, World Cup 2022, Women's Euro 2022/2025, Women's World Cup 2023, AFCON 2023, MLS 2023) | StatsBomb Open Data, https://github.com/statsbomb/open-data | Research / non-commercial use; attribution to StatsBomb required (see the repository licence). |
| StatsBomb events 2015/16 Premier League, La Liga, Serie A, Ligue 1, Bundesliga (Leverkusen) and Copa America 2024 | StatsBomb Open Data (same repository). No freeze frames exist for these. | Same as above. |
| Wyscout events and metadata (2017/18 Premier League, La Liga, Serie A, Bundesliga, Ligue 1; Euro 2016; World Cup 2018) | Pappalardo & Massucco, Soccer match event dataset, https://doi.org/10.6084/m9.figshare.c.4415000.v5 | CC BY 4.0 - attribution required. |

## What was derived by this project (not provider data)
- **Packing, line-breaking, lane/pressure features** - computed from StatsBomb 360 freeze frames (packing.py).
- **Estimated xG for Wyscout shots** - a gradient-boosting model trained on the Wyscout shots themselves (shot location, body part, free kick), scored out-of-fold by match. StatsBomb xG is the provider's own.
- **Wyscout minutes played** - estimated from lineups and substitutions assuming 90-minute matches.
- **Role (GK/DF/MD/FW) for StatsBomb players** - derived from the most-played position.

## Known limits
- Wyscout has no player positions, so true packing is not available for Wyscout competitions.
- The 2015/16 StatsBomb leagues and Copa America 2024 have no freeze frames.
- AFCON 2023 has no usable freeze frames (51 of 52 matches are marked "unscheduled" by StatsBomb).
- Bundesliga 2015/16 and the La Liga 2020/21, Ligue 1 2021/22 and 2022/23 and Bundesliga 2023/24 freeze-frame samples cover a single club's season.
