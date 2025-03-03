# import some basic libraries
import os
import pandas as pd

# import utility function for sample data path
from pymer4.utils import get_resource_path


# Load and checkout sample data
df = pd.read_csv(os.path.join(get_resource_path(), "sample_data.csv"))
print(df.head())

# Import the linear regression model class
from pymer4.models import Lm

# print(df)

# Initialize model using 2 predictors and sample data
model = Lm("DV ~ IV1 + IV2 + (1|Group)", data=df)

# Fit it
print(model.fit())

from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

# random = {"a": 'C(Group)'}
# model = BinomialBayesMixedGLM.from_formula('DV ~ IV1 + IV2 + (1|Group)', random, df)
# result = model.fit()