# Project Buildup History: X Education Lead Prioritization

- Repository: `x-education-lead-prioritization`
- Category: `data_science`
- Subtype: `optimization`
- Source: `project_buildup_2021_2025_daily_plan_extra.csv`
## 2023-10-02 - Day 3: Logistic regression baseline

- Task summary: Started the main modeling work for X Education Lead Prioritization. The goal is to score each lead with a probability of conversion so the sales team can prioritize their calls. Built the logistic regression baseline today with proper train/test splits and class weight adjustment for the imbalanced classes. The baseline AUC was around 0.87 which is a decent starting point. Also looked at the calibration - the predicted probabilities were a bit overconfident at the extremes so added a Platt scaling step.
- Deliverable: Logistic regression baseline at 0.87 AUC. Platt scaling added for calibration.
## 2023-10-02 - Day 3: Logistic regression baseline

- Task summary: Realized the lead source feature had 14 unique values and was being one-hot encoded, creating a very wide matrix. Applied target encoding instead to reduce dimensionality while preserving the signal.
- Deliverable: Lead source target-encoded instead of one-hot. Matrix width reduced significantly.
## 2023-10-09 - Day 4: Lead scoring refinement

- Task summary: Refined the lead scoring model today by adding interaction terms between the most predictive pairs of features. The interaction between website visits count and time spent on the website was particularly useful - high visits with low time suggests browse-and-leave behavior which correlates with non-conversion. Also built the final scoring table output that ranks leads by predicted conversion probability with a suggested tier label.
- Deliverable: Interaction features added. Final lead scoring output table built.
## 2023-11-27 - Day 5: Business framing

- Task summary: Worked on the business framing section of the X Education Lead Prioritization project. The model output is a score but what the sales team needs is an actionable decision rule. Wrote a section explaining how to set the score threshold based on team capacity - if the team can make 200 calls per week and there are 1000 leads, set the threshold at the 80th percentile. Also built a simple calculator cell that takes call capacity as input and returns the recommended threshold.
- Deliverable: Business framing written. Threshold calculator based on call capacity added.
## 2023-11-27 - Day 5: Business framing

- Task summary: Added a section on model monitoring - what metrics to track over time to detect model degradation, and when to retrain.
- Deliverable: Model monitoring guidance section added.
## 2023-11-27 - Day 5: Business framing

- Task summary: Cleaned up the notebook flow and moved all the configuration parameters to a single cell at the top so they're easy to find and change.
- Deliverable: All config parameters moved to top cell.
