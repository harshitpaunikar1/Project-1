# Project Buildup History: X Education Lead Prioritization

- Repository: `x-education-lead-prioritization`
- Category: `data_science`
- Subtype: `optimization`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2023-10-02 - Day 3: Logistic regression baseline

- Task summary: Started the main modeling work for X Education Lead Prioritization. The goal is to score each lead with a probability of conversion so the sales team can prioritize their calls. Built the logistic regression baseline today with proper train/test splits and class weight adjustment for the imbalanced classes. The baseline AUC was around 0.87 which is a decent starting point. Also looked at the calibration — the predicted probabilities were a bit overconfident at the extremes so added a Platt scaling step.
- Deliverable: Logistic regression baseline at 0.87 AUC. Platt scaling added for calibration.
