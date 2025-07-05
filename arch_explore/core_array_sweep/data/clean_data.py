import pandas as pd
import os

input_file = 'sweep_data_100M.csv'
output_file = 'sweep_data_100M_clean.csv'

# Read the CSV file
df = pd.read_csv(input_file)
# Filter out rows where 'best_val_loss' is missing (NaN or empty)
cleaned_df = df[df['best_val_loss'].notnull() & (df['best_val_loss'] != '')]
# Save the cleaned data to a new CSV file
cleaned_df.to_csv(output_file, index=False)

# Save the cleaned data to the same folder as the input file
output_path = os.path.join(os.path.dirname(input_file), output_file)
cleaned_df.to_csv(output_path, index=False)

# filter out the models with sizes under 95M
filtered_df = cleaned_df[cleaned_df['num_params'] >= 95_000_000]
filtered_df = filtered_df[filtered_df['num_params'] <= 105_000_000]

# Save the filtered data to a new CSV file
filtered_output_file = 'sweep_data_100M_filtered.csv'
filtered_df.to_csv(filtered_output_file, index=False)


