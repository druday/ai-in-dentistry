# Flow Figure Caption Drafts

## Option 1 (Main figure candidate)

**Figure X. Institutional role flow across period cohorts in AI-dentistry collaboration networks.**  
All institutions are classified in each period as Core, Semi-Peripheral, Peripheral, Isolate, or Absent based on period-specific degree centrality thresholds. Stream widths represent the number of institutions transitioning between roles across adjacent periods, including entries from and exits to the Absent state.

## Option 2 (Companion / supplementary)

**Figure Y. Field-specific institutional role flows across period cohorts.**  
The alluvial flow analysis is stratified by institution field category (Dental, Medical, Technical, Other). Within each panel, stream widths indicate role transitions across adjacent periods, enabling comparison of structural consolidation and peripheral turnover by field.

## Methods note (for text consistency)

Role assignment for these visualizations uses period-specific degree centrality:

- Core: top 10% of non-zero degree centrality values in that period
- Semi-Peripheral: 50th to 90th percentile of non-zero degree centrality
- Peripheral: non-zero degree centrality below the 50th percentile
- Isolate: degree centrality of 0
- Absent: institution not observed in that period

## Interactive Sankey note

The interactive Sankey version uses institution-level nodes `(period, institution)` and links each institution to its next observed period. Visual encoding:

- Node fill: Newcomer (blue) vs Consistent (orange)
- Node border: institution field (Dental, Medical, Technical, Other)
- Link color: institution field (semi-transparent)
