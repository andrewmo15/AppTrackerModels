import pandas as pd

# read in data
df = pd.read_csv("data generation/data.csv")

# drop irrelevant features
unrelevant_features = ["company"]
relevant_features = ["from", "subject", "body"]
df.drop(unrelevant_features,inplace=True,axis=1)

# separate dataframe by status
submitted = df.loc[df["status"] == "SUBMITTED"]
inprogress = df.loc[df["status"] == "INPROGRESS"]
rejected = df.loc[df["status"] == "REJECTED"]
notinterview = df[df["status"].isnull()]

# Convert labels into integers
#   SUBMITTED = 1
#   INPROGRESS = 2
#   REJECTED = 3
#   NA = 4
submitted["status"] = 1
inprogress["status"] = 2
rejected["status"] = 3
notinterview["status"] = 4

# reconcatenate data
data = pd.concat([submitted, inprogress, rejected, notinterview],axis=0)
data.reset_index(inplace=True)