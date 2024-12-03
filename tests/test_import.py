import nideconv as nd
from lazyplot import plotting
import pandas as pd
import os
opj = os.path.join
opd = os.path.dirname

beta_file = opj(opd(opd(plotting.__file__)), "data", "betas.csv")
profile_file = opj(opd(opd(plotting.__file__)), "data", "betas.csv")
df_beta = pd.read_csv(beta_file)
df_prof = pd.read_csv(profile_file)
