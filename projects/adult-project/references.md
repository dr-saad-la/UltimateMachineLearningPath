## Understanding Data

When replicating this project, I had to look up online so I can understand the data in more depth. The most difficult feature was `fnlwgt` (which is the `final weight`). Here are the details of this variable based on the notes given in metadata file: 

1.	**CPS (Current Population Survey)**: CPS is a survey conducted by the U.S. Census Bureau, which provides data on the labor force status, demographics, and economic characteristics of the population. The survey represents a sample of households across the U.S. and uses weights like `fnlwgt` to adjust for differences in sampling probabilities to better represent the entire population.

2.	**Purpose of fnlwgt (Final Weight)**: The `fnlwgt` variable is used to “weight” individual records in the dataset so that they represent the population accurately. This is necessary because the CPS sample is not a simple random sample. Different households and individuals may have different probabilities of being included in the sample based on location, demographics, etc. `fnlwgt` adjusts for these differences.

3.	**Controlled to Population Estimates**: The weights are calibrated to reflect independent estimates of the U.S. population. These estimates are based on three specific control factors:
    1. Single cell estimate of the population 16+ for each state: Adjusting the data to match population estimates of people aged 16 and older in each state.
    2. Hispanic Origin by age and sex: Ensuring the data accurately represents the distribution of Hispanic people by age and sex.
    3. Race, age, and sex: Making sure the data reflects the correct proportions of racial groups, ages, and sexes.

4.	**Raking Method**: The term “raking” refers to an iterative process of adjusting the weights (fnlwgt) to match multiple control totals (e.g., age, sex, race) for the population. The dataset is “raked” six times through these control totals, ensuring the weighted data fits the population estimates as closely as possible.

5.	**Weighted Tallies**: By applying fnlwgt to the survey data, the Census Bureau generates “weighted tallies” (or estimates) of various socio-economic characteristics of the U.S. population. For example, they can estimate how many people in a certain state, age group, or demographic category are employed, unemployed, or not in the labor force.

6.	**Within-State Weights**: The caveat mentioned here is that the weights are designed to ensure that people with similar demographic characteristics (like age, sex, race) have similar weights within the same state. However, the statement does not apply across different states because each state has its own probability of being included in the sample.