### Write Python scripts to implement basic operations of TensorFlow and tensors

This experiment focuses on getting familiar with the fundamental building blocks of TensorFlow: Tensors. Tensors are multi-dimensional arrays, and they are the central data structure in TensorFlow. We'll cover how to create, manipulate, and perform basic operations on them.

#### What and Why for each cell:

#### Cell 1: Setup and Creating Tensors

**What:**

```python
# Import TensorFlow library
import tensorflow as tf
import numpy as np

# Print the TensorFlow version to confirm installation
print(f"TensorFlow version: {tf.__version__}")

# Create a constant tensor (a 0-D tensor or scalar)
scalar = tf.constant(7)
print(f"Scalar: {scalar}")
print(f"Scalar shape: {scalar.shape}")

# Create a 1-D tensor (vector)
vector = tf.constant([10, 7])
print(f"\nVector: {vector}")
print(f"Vector shape: {vector.shape}")

# Create a 2-D tensor (matrix)
matrix = tf.constant([[1, 2, 3],
                      [4, 5, 6]])
print(f"\nMatrix: {matrix}")
print(f"Matrix shape: {matrix.shape}")

# Create a tensor with a specified data type
float_tensor = tf.constant([[1., 2., 3.],
                           [4., 5., 6.]], dtype=tf.float32)
print(f"\nFloat Tensor: {float_tensor}")
print(f"Float Tensor data type: {float_tensor.dtype}")
```

**Why:**
This cell is for the basic setup and demonstrating how to create tensors.

  - We import TensorFlow and NumPy (often used alongside TensorFlow).
  - We use `tf.constant()` to create tensors with fixed values. This is the most common way to create a basic tensor.
  - We show examples of a scalar (0-D tensor), a vector (1-D tensor), and a matrix (2-D tensor) to illustrate the concept of dimensionality.
  - We also demonstrate how to explicitly set the data type using the `dtype` parameter. This is important for memory management and for ensuring compatibility with different TensorFlow operations.

#### Cell 2: Tensor Manipulation and Attributes

**What:**

```python
# Create a tensor with random values
random_tensor = tf.random.uniform(shape=(3, 3), minval=0., maxval=1.)
print(f"Random Tensor:\n{random_tensor}")

# Get the rank of a tensor (number of dimensions)
print(f"\nRank of random_tensor: {tf.rank(random_tensor)}")

# Get the size of a tensor
print(f"Size of random_tensor: {tf.size(random_tensor)}")

# Get the shape of a tensor
print(f"Shape of random_tensor: {random_tensor.shape}")

# Reshaping a tensor
reshaped_tensor = tf.reshape(random_tensor, shape=(9, 1))
print(f"\nReshaped Tensor (9x1):\n{reshaped_tensor}")
print(f"Reshaped Tensor shape: {reshaped_tensor.shape}")
```

**Why:**
This cell covers essential tensor properties and manipulation.

  - We create a tensor with random values using `tf.random.uniform()`. Random tensors are often used to initialize weights in a neural network.
  - We introduce key tensor attributes: `tf.rank()` (the number of dimensions), `tf.size()` (total number of elements), and `tensor.shape` (the size of each dimension). Understanding these attributes is crucial for working with neural networks.
  - We demonstrate how to reshape a tensor using `tf.reshape()`. This operation changes the dimensions of a tensor without changing its total number of elements. Reshaping is a fundamental operation for preparing data for different layers in a neural network (e.g., flattening an image into a vector).

#### Cell 3: Basic Tensor Operations

**What:**

```python
# Create a new tensor for operations
tensor = tf.constant([[1., 2., 3.],
                      [4., 5., 6.]])
print(f"Original Tensor:\n{tensor}")

# Element-wise addition
added_tensor = tensor + 10
print(f"\nTensor + 10:\n{added_tensor}")

# Element-wise multiplication
multiplied_tensor = tensor * 2
print(f"\nTensor * 2:\n{multiplied_tensor}")

# Matrix multiplication (Dot product)
# This requires a compatible shape. Transpose the tensor.
transposed_tensor = tf.transpose(tensor)
print(f"\nTransposed Tensor:\n{transposed_tensor}")

# Perform matrix multiplication
dot_product = tf.matmul(tensor, transposed_tensor)
print(f"\nDot Product (tf.matmul):\n{dot_product}")

# Another way to perform matrix multiplication using the '@' operator
dot_product_op = tensor @ transposed_tensor
print(f"\nDot Product ('@' operator):\n{dot_product_op}")
```

**Why:**
This cell focuses on arithmetic operations on tensors.

  - We show element-wise operations like addition and multiplication, which apply the operation to every element of the tensor. This is a common pattern in machine learning for scaling or shifting data.
  - We introduce the concept of matrix multiplication, which is the cornerstone of neural network computations. We first show how to transpose a tensor using `tf.transpose()` to make its dimensions compatible for multiplication.
  - We then use `tf.matmul()` and the more concise `@` operator to perform the matrix multiplication. The output shows the result of multiplying the original matrix by its transpose.

#### Cell 4: Slicing and Indexing Tensors

**What:**

```python
# Create a 3-D tensor
three_d_tensor = tf.constant([[[1, 2, 3],
                               [4, 5, 6]],
                              [[7, 8, 9],
                               [10, 11, 12]]])
print(f"3-D Tensor:\n{three_d_tensor}")

# Index a single element
print(f"\nElement at [0, 1, 2]: {three_d_tensor[0, 1, 2]}")

# Slice a part of the tensor (from the first dimension)
slice_1 = three_d_tensor[0]
print(f"\nSlice of the first dimension:\n{slice_1}")

# Slice with a specific range
slice_2 = three_d_tensor[:, :, 1]
print(f"\nSlice of the second column of each matrix:\n{slice_2}")

# Access the first element of each row
slice_3 = three_d_tensor[..., 0]
print(f"\nSlice of the first element of each innermost vector:\n{slice_3}")
```

**Why:**
This cell demonstrates how to access specific parts of a tensor, which is vital for data manipulation and model debugging.

  - We create a 3-D tensor to show indexing in multiple dimensions.
  - We show how to index a single element using bracket notation (e.g., `[0, 1, 2]`).
  - We explain slicing with `:` and `...` (ellipsis) notation. Slicing allows you to extract sub-tensors (e.g., a specific row, column, or a sub-matrix). This is a powerful and flexible way to prepare data for different parts of a model or to inspect model outputs. The `...` operator is a convenient way to select across all preceding dimensions.
