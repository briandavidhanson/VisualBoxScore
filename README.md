# 🏀 Visual Boxscores

**Every possession tells a story.**

Visual Boxscores transforms NBA games into detailed, interactive visualizations that reveal the flow and rhythm of basketball in ways traditional box scores never could.

## What is Visual Boxscores?

Traditional box scores reduce a 48-minute game to a handful of counting stats. Visual Boxscores takes a different approach—tracking every possession to show *how* a game unfolded, not just the final tallies.

Each visualization includes:

- **Efficiency Chart** — Points minus possessions (pMp) for each team, showing offensive efficiency in real-time across the game
- **Margin Flow** — Game margin visualized possession-by-possession, with clutch time highlighted
- **Player Timelines** — Gantt-style charts showing when each player was on the court, color-coded by plus/minus during each stint
- **Shot Tracking** — Every shot attempt mapped to the possession it occurred on, with markers for makes, misses, and assisted baskets
- **Enhanced Box Scores** — Traditional stats plus advanced metrics like points per possession, shooting efficiency by zone, assist points, turnover points, and modified GameScore

## Navigating the Site

Games can be browsed two ways:
- **By Date** — Games organized chronologically by game date
- **By Team** — All available games for each NBA team

Click any game to open its full interactive visualization.

## Reading the Visualizations

### Efficiency Chart (Top)
- **Red line**: Away team's cumulative points minus possessions
- **Blue line**: Home team's cumulative points minus possessions  
- **Green line**: Home team's margin (in possessions)
- **Orange segments**: Clutch time (final 5 minutes, margin within 5 points)
- **Shot markers**: Circles (close), squares (mid-range), triangles (3PT)

### Player Timelines (Middle)
- **Gray bars**: Player on bench
- **Colored bars**: Player on court (green = positive plus/minus, red = negative)
- **Shot symbols**: Same as efficiency chart, showing individual player's shots
- **Other markers**: Assists (○), rebounds (□), turnovers (✗), steals (✗ filled), blocks (|)

### Box Scores (Bottom)
Each stat column shows multiple values:
- **PTS/PPP**: Points and points per possession
- **CLS/MID/3PT/FT**: Made-attempted and points per shot
- **TOV/AST/ORB/STL**: Count, percentage, and points generated/lost
- **GmSc/mGmSc**: Traditional GameScore and modified GameScore

## About

Visual Boxscores is a personal project exploring new ways to understand basketball through data visualization. The visualizations are built with Python and Plotly, using play-by-play data to reconstruct each game possession by possession.

---

*Beyond the box score.*
