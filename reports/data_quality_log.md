
# Data Quality Log

Dataset: freMTPL2freq + freMTPL2sev (merged)
Initial shape: 678,013 rows x 13 columns

---

### Driver age outside [18, 100]

- **Rows affected**: 0
- **Action**: Rows excluded from cleaned dataset
- **Rationale**: Ages below 18 are legally impossible for a licensed driver. Ages above 100 are almost certainly data entry errors. This pattern is documented in actuarial tutorials for this dataset.


### Exposure <= 0

- **Rows affected**: 0
- **Action**: Rows excluded (zero or negative exposure has no actuarial meaning)
- **Rationale**: Exposure represents the fraction of a year a policy was active. A value of zero means no time at risk; such rows cannot contribute meaningful frequency information and are dropped.


### Exposure > 1

- **Rows affected**: 1,224
- **Action**: Exposure capped at 1.0
- **Rationale**: Exposure cannot exceed 1 year for a single policy period. Values slightly above 1.0 are rounding artefacts and are capped rather than dropped to retain the policy record.


### Negative vehicle age

- **Rows affected**: 0
- **Action**: Rows excluded
- **Rationale**: Negative vehicle age is a data entry error with no valid interpretation.


### ClaimNb > 4 with Exposure < 0.1

- **Rows affected**: 4
- **Action**: ClaimNb capped at 4 for affected rows
- **Rationale**: Several rows with very short exposure (< 5 weeks) carry an implausibly high claim count. These are documented outlier rows in actuarial literature for the freMTPL2 dataset. Capping at 4 retains the policies while limiting the distortion they cause in the Poisson model.


### BonusMalus > 150

- **Rows affected**: 209
- **Action**: Rows retained, flagged with is_high_risk=True
- **Rationale**: BonusMalus above 150 indicates a very high-risk driver. These are genuine policies and should be included in the model; they provide important signal for the high end of the risk spectrum. A flag column is added so the segment can be isolated in EDA.


---

**Final cleaned shape**: 678,013 rows x 14 columns

