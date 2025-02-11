import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
import matplotlib.pyplot as plt

np.random.seed(42)

# # Generate x_data: 300 samples, 136 features (mimics UCLA distribution - healty)
# x_data = np.random.normal(loc=0, scale=1, size=(300, 136))
#
# # Generate y_data: 20 samples, 136 features (mimics pain distribution - healty)
# y_data = np.random.normal(loc=0.2, scale=1, size=(20, 136))
#
# # Generate y_new_data: 20 samples, 136 features (mimics pain distribution - pain)
# y_pain_data = np.random.normal(loc=0.2, scale=1, size=(20, 136))
# # mofify some features to look like pain data
# shift_feature_indices = [10, 11, 12, 13]
# y_pain_data[:, shift_feature_indices] += 0.3

#  Align y_data to x_data

# Combine x_data and y_data along the sample (row) dimension.
# The combined data has shape (320, 136): 320 samples, 136 features.
combined_data = np.concatenate([x_data, y_data], axis=0)

# Create batch labels: 'ref' for x_data samples and 'new' for y_data samples
batch_labels = ['ref'] * x_data.shape[0] + ['new'] * y_data.shape[0]
covars = pd.DataFrame({'batch': batch_labels})

# Transpose combined_data to shape (n_features, n_samples) as required by neuroCombat.
combined_data_T = combined_data.T  # shape becomes (136, 320)

# Run neuroCombat to harmonize the data.
combat_results = neuroCombat(dat=combined_data_T,
                             covars=covars,
                             batch_col='batch')


aligned_data_T = combat_results['data']
aligned_data = aligned_data_T.T  # devine (320, 136)

# Store y_data after alignement
y_data_aligned = aligned_data[x_data.shape[0]:, :]

# Print available keys from neuroCombat output and the estimates sub-dictionary.
print("neuroCombat output keys:", combat_results.keys())
estimates = combat_results['estimates']
print("neuroCombat estimates keys:", estimates.keys())

# Retrieve gamma and delta from the estimates dictionary.
# Based on your output, they are stored under 'gamma.star' and 'delta.star'
if 'gamma.star' in estimates:
    gamma = estimates['gamma.star']
elif 'gamma_star' in estimates:
    gamma = estimates['gamma_star']
else:
    raise KeyError("Could not find gamma in estimates: available keys: " + str(estimates.keys()))

if 'delta.star' in estimates:
    delta = estimates['delta.star']
elif 'delta_star' in estimates:
    delta = estimates['delta_star']
else:
    raise KeyError("Could not find delta in estimates: available keys: " + str(estimates.keys()))



gamma_new = gamma[0, :]   # shape becomes (136,)
delta_new = delta[0, :]   # shape becomes (136,)

# Reshape gamma_new and delta_new to (136, 1) for proper broadcasting.
gamma_new = gamma_new[:, np.newaxis]  # now shape: (136, 1)
delta_new = delta_new[:, np.newaxis]  # now shape: (136, 1)

# Compute the overall means (alpha) per feature from the combined (transposed) data.
alpha = combined_data_T.mean(axis=1)  # shape: (136,)

# Process y_new_data: transpose it to have shape (n_features, n_samples) as required by the transformation.
y_pain_data_T = y_pain_data.T  # shape becomes (136, 20)

# Center the new data using alpha.
centered = y_pain_data_T - alpha[:, np.newaxis]  # shape: (136, 20)

# Apply the transformation to obtain the adjusted (harmonized) new data.
pain_adjusted_data_T = (centered - gamma_new) / delta_new  # shape: (136, 20)

# Transpose back to original orientation: (n_samples, n_features)
pain_adjusted_data = pain_adjusted_data_T.T  # now shape: (20, 136)
# pain_adjusted_data coresponds to pain class

# Print shapes of all datasets to ensure consistency.

x_means = x_data.mean(axis=1)           # shape: (300,)
y_means = y_data.mean(axis=1)           # shape: (20,)
y_data_aligned_means = y_data_aligned.mean(axis=1)
y_pain_means = y_pain_data.mean(axis=1)     # shape: (20,)
adjusted_pain_means = pain_adjusted_data.mean(axis=1)  # shape: (20,)

# Create a figure with two subplots.
plt.figure(figsize=(14, 6))

# Left subplot: Histogram of sample means.
plt.subplot(1, 2, 1)
plt.hist(x_means, bins=30, alpha=0.6, label='x_data (ref)')
plt.hist(y_means, bins=30, alpha=0.6, label='y_data (old)')
plt.hist(y_data_aligned_means, bins=30, alpha=0.6, label='y_data (aligned)')
plt.hist(y_pain_means, bins=30, alpha=0.6, label='y_pain_data (pre-align)')
plt.hist(adjusted_pain_means, bins=30, alpha=0.6, label='y_pain_data (post-align)')
plt.xlabel('Mean across features')
plt.ylabel('Frequency')
plt.title('Histogram of Sample Means')
plt.legend()

# Right subplot: Boxplot for a side-by-side comparison.
plt.subplot(1, 2, 2)
data_to_plot = [x_means, y_means, y_data_aligned_means, y_pain_means, adjusted_pain_means]
plt.boxplot(data_to_plot, labels=['x_data (ref)', 'y (pre-align)','y (aligned)', 'pain (pre-align)', 'pain (aligned)'])
plt.title('Boxplot of Sample Means')

plt.tight_layout()
plt.show()