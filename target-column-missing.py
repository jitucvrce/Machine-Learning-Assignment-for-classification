import pandas as pd

# Load the raw dataset from the web link
url = "https://uci.edu"
df = pd.read_csv('C:\\Users\\bbich\\Downloads\\wine+quality\\winequality-white.csv', sep=';')

# Create the binary target column required by the app
df['target'] = (df['quality'] >= 7).astype(int)

# Drop the old quality column so it matches our feature count
df = df.drop(columns=['quality'])

# Take a test slice (just the last 500 rows to keep it light for Streamlit free tier)
test_slice = df.tail(500)

# Save it locally
test_slice.to_csv("fixed_test_data.csv", index=False)
print("Successfully generated fixed_test_data.csv with the 'target' column!")
