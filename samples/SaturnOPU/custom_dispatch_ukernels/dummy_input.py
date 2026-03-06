import numpy as np

# Shape (batch_size, input_size)
shape = (16, 1024)

# Data type is float32, as shown in your IR
dtype = np.float32

# Generate random f32 data
# np.random.randn() produces data with a standard normal distribution (mean 0, var 1)
data = np.random.randn(*shape).astype(dtype)

# Define the output filename
filename = 'input_16x1024_f32.npy'

# Save the array to a .npy file
np.save(filename, data)

print(f"Successfully saved .npy file to: {filename}")
print(f"Shape: {data.shape}")
print(f"Dtype: {data.dtype}")