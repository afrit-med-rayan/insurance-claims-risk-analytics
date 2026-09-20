# Key Findings from Exploratory Data Analysis

### Claim Frequency by Region

Region R11 stands out with one of the highest claim frequencies despite having a large exposure volume, 
suggesting significant urban concentration (like the Paris area) or regional risk factors that 
warrant a territorial pricing multiplier. Conversely, regions like R73 and R53 
exhibit much lower risk profiles.

### Claim Frequency by Vehicle Power

There is a notable non-linear relationship between vehicle power and claim frequency. 
Mid-to-high power vehicles (e.g., categories 9 and above) generally show elevated 
risk compared to standard commuter vehicles (categories 4-6). This confirms vehicle 
performance is a relevant rating factor.

### Claim Frequency by Driver Age Band

Drivers in the 18-25 age band exhibit a substantially higher claim frequency than 
any other group, indicating the classic "young driver" risk premium is strongly 
justified here. Risk drops off sharply for the 26-40 and 41-60 bands, before 
flattening out or slightly rising for the 61+ cohort.

### Claim Frequency by BonusMalus

BonusMalus is highly predictive of future claim frequency. There is a steep, 
monotonic increase in claims per year as the BonusMalus score worsens (higher deciles). 
This validates the French bonus-malus system as an effective mechanism for tracking 
unobserved driver risk over time.

### Severity Distribution

Claim amounts follow a heavy-tailed distribution, heavily skewed to the right. 
When viewed on a log-scale, the distribution roughly resembles a normal curve, 
confirming that severity modeling requires techniques suited for strictly positive, 
right-skewed data (such as Gamma or Tweedie GLMs).

### Numeric Feature Correlations

Most numeric features show low collinearity, which is ideal for regression modeling. 
The highest correlations are expected mechanical relationships, such as between the 
simulated Premium and its components (BonusMalus, VehPower, Density). DrivAge has a 
moderate negative correlation with BonusMalus, reflecting that older drivers tend 
to have accumulated better risk histories.

### Portfolio Composition

The portfolio is geographically concentrated in a few top regions (like R24). 
The vehicle mix is split fairly evenly between Regular and Diesel fuel types. 
Vehicle brands B1 and B2 dominate the book. These volumetric imbalances suggest 
that while some segments have deep data for pricing, rarer segments (e.g., brand B14) 
may suffer from high variance in claims experience.

