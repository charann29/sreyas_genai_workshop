# Compact GAN Implementation for Google Colab

## Cell 1: Setup and Data Preparation
**What:** Install dependencies, load MNIST dataset, and preprocess images
**Why:** We need TensorFlow for neural networks, MNIST for training data, and normalization to [-1,1] range for stable GAN training

```python
!pip install tensorflow matplotlib numpy

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output

# Load and preprocess MNIST
(train_images, _), (_, _) = tf.keras.datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1] for tanh activation

# Create dataset pipeline
BATCH_SIZE = 256
dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(60000).batch(BATCH_SIZE)

print(f"TensorFlow: {tf.__version__}, GPU: {len(tf.config.list_physical_devices('GPU')) > 0}")
print(f"Data shape: {train_images.shape}, Range: [{train_images.min():.1f}, {train_images.max():.1f}]")
```

## Cell 2: Define Generator and Discriminator Networks
**What:** Create both neural networks - generator creates fake images, discriminator classifies real vs fake
**Why:** Generator uses transposed convolutions to upsample noise into images, discriminator uses regular convolutions to classify

```python
def make_generator():
    """
    Generator: Noise (100D) → Image (28×28×1)
    Dense: Fully connected layer that transforms 100D noise to 7×7×256 feature map
    BatchNormalization: Normalizes layer inputs to stabilize training
    LeakyReLU: Activation function that allows small negative values (prevents dying neurons)
    Conv2DTranspose: Upsampling convolution that increases spatial dimensions
    Tanh: Output activation that maps to [-1,1] range (matches our data normalization)
    """
    model = tf.keras.Sequential([
        layers.Dense(7*7*256, input_shape=(100,)),           # Transform noise vector
        layers.BatchNormalization(),                          # Stabilize training
        layers.LeakyReLU(0.2),                               # Non-linear activation
        layers.Reshape((7, 7, 256)),                         # Reshape for convolution
        
        layers.Conv2DTranspose(128, 5, strides=2, padding='same'),  # 7×7 → 14×14
        layers.BatchNormalization(),
        layers.LeakyReLU(0.2),
        
        layers.Conv2DTranspose(1, 5, strides=2, padding='same', activation='tanh')  # 14×14 → 28×28
    ])
    return model

def make_discriminator():
    """
    Discriminator: Image (28×28×1) → Probability (Real/Fake)
    Conv2D: Regular convolution that reduces spatial dimensions while extracting features
    Dropout: Randomly sets some neurons to 0 during training (prevents overfitting)
    Flatten: Converts 2D feature maps to 1D vector for final classification
    Dense: Final layer outputs single value (logit) for binary classification
    """
    model = tf.keras.Sequential([
        layers.Conv2D(64, 5, strides=2, padding='same', input_shape=[28, 28, 1]),  # 28×28 → 14×14
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),                                 # Prevent overfitting
        
        layers.Conv2D(128, 5, strides=2, padding='same'),    # 14×14 → 7×7
        layers.LeakyReLU(0.2),
        layers.Dropout(0.3),
        
        layers.Flatten(),                                     # Flatten to 1D
        layers.Dense(1)                                       # Binary classification output
    ])
    return model

# Create networks
generator = make_generator()
discriminator = make_discriminator()

print("Networks created!")
generator.summary()
discriminator.summary()
```

## Cell 3: Define Loss Functions and Optimizers
**What:** Set up loss functions for adversarial training and Adam optimizers
**Why:** Binary crossentropy measures classification accuracy, Adam optimizer adapts learning rates automatically

```python
# Loss function
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)  # from_logits=True means no sigmoid in model

def discriminator_loss(real_output, fake_output):
    """
    Discriminator wants: Real images → 1, Fake images → 0
    real_loss: How well it classifies real images as real
    fake_loss: How well it classifies fake images as fake
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)    # Real should be 1
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)   # Fake should be 0
    return real_loss + fake_loss

def generator_loss(fake_output):
    """
    Generator wants: Fake images → 1 (fool discriminator)
    Tries to make discriminator classify fake images as real
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)         # Want fake to be classified as 1

# Optimizers - Adam adapts learning rate automatically
gen_optimizer = tf.keras.optimizers.Adam(1e-4)    # Learning rate for generator
disc_optimizer = tf.keras.optimizers.Adam(1e-4)   # Learning rate for discriminator

print("Loss functions and optimizers ready!")
```

## Cell 4: Training Loop with Visualization
**What:** Complete training loop that alternates between training generator and discriminator
**Why:** GradientTape records operations for automatic differentiation, @tf.function compiles for speed

```python
noise_dim = 100  # Dimension of input noise vector
seed = tf.random.normal([16, noise_dim])  # Fixed noise for consistent visualization

@tf.function  # Compiles function to TensorFlow graph for faster execution
def train_step(images):
    """
    One training step:
    1. Generate fake images from noise
    2. Get discriminator predictions on real and fake images
    3. Calculate losses for both networks
    4. Update weights using gradients
    """
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    # GradientTape records operations for automatic differentiation
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Calculate gradients (derivatives of loss w.r.t. weights)
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients to update weights
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

def show_images(epoch):
    """Generate and display images using fixed seed for comparison"""
    predictions = generator(seed, training=False)
    
    plt.figure(figsize=(8, 8))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')  # Denormalize for display
        plt.axis('off')
    plt.suptitle(f'Epoch {epoch}')
    plt.show()

# Training loop
EPOCHS = 30
print("Starting training...")

for epoch in range(EPOCHS):
    gen_losses, disc_losses = [], []
    
    for image_batch in dataset:
        g_loss, d_loss = train_step(image_batch)
        gen_losses.append(g_loss)
        disc_losses.append(d_loss)
    
    # Show progress every 5 epochs
    if (epoch + 1) % 5 == 0:
        clear_output(wait=True)
        avg_g_loss = tf.reduce_mean(gen_losses)
        avg_d_loss = tf.reduce_mean(disc_losses)
        print(f'Epoch {epoch+1}/{EPOCHS} - Gen Loss: {avg_g_loss:.4f}, Disc Loss: {avg_d_loss:.4f}')
        show_images(epoch + 1)

print("Training complete!")
```

## Cell 5: Generate Final Results and Save Model
**What:** Generate final images with different noise inputs and save the trained generator
**Why:** Demonstrates final model capability and preserves the trained model for future use

```python
# Generate final results with new random noise
print("Final Results:")
test_noise = tf.random.normal([25, noise_dim])
final_images = generator(test_noise, training=False)

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(final_images[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')
plt.suptitle('Generated MNIST Digits - Final Results')
plt.tight_layout()
plt.show()

# Save the trained generator
generator.save('mnist_generator.h5')
print("Model saved! Use tf.keras.models.load_model('mnist_generator.h5') to load it later.")

# Quick test of saved model
print("\nTesting saved model:")
loaded_gen = tf.keras.models.load_model('mnist_generator.h5')
test_image = loaded_gen(tf.random.normal([1, noise_dim]), training=False)
plt.figure(figsize=(3, 3))
plt.imshow(test_image[0, :, :, 0] * 127.5 + 127.5, cmap='gray')
plt.title('Generated by Loaded Model')
plt.axis('off')
plt.show()
```

## Key Terms Explained:

**Neural Network Components:**
- **Dense**: Fully connected layer where each input connects to each output
- **Conv2D**: 2D convolution that detects patterns/features in images
- **Conv2DTranspose**: "Deconvolution" that upsamples (increases image size)
- **BatchNormalization**: Normalizes inputs to each layer for stable training
- **Dropout**: Randomly ignores some neurons during training to prevent overfitting

**Activation Functions:**
- **LeakyReLU**: Like ReLU but allows small negative values (prevents "dead neurons")
- **Tanh**: Maps values to [-1, 1] range (matches our normalized data)

**Training Components:**
- **GradientTape**: Records operations to compute gradients automatically
- **Adam Optimizer**: Adaptive optimizer that adjusts learning rates
- **Binary Crossentropy**: Loss function for binary classification (real vs fake)
- **@tf.function**: Compiles Python function to optimized TensorFlow graph

**GAN Concepts:**
- **Generator**: Creates fake data from random noise
- **Discriminator**: Classifies data as real or fake
- **Adversarial Training**: Both networks compete - generator tries to fool discriminator, discriminator tries to detect fakes

This compact version reduces the original 13 cells to just 5 cells while maintaining all essential functionality and detailed explanations!
