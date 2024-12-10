import os
import matplotlib.pyplot as plt
import pandas as pd

this_filepath = os.path.abspath(__file__)

# Load data
datapath = os.path.join(os.path.dirname(this_filepath), "..", "data", "results.xls")

references = [10, 50, 80]

data = dict()
for ref in references:
    data[ref] = pd.read_excel(
        datapath,
        sheet_name=f"{ref}",
    )

# TODO: Plot the errobar
