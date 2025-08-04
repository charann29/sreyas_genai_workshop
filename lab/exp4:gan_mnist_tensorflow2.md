# GAN Implementation for Google Colab 

## Cell 1: Install and Import Dependencies
**What:** Install required packages and import necessary libraries
**Why:** We need TensorFlow 2.x, matplotlib for visualization, and other standard libraries for our GAN implementation

```python
# Install required packages
!pip install tensorflow matplotlib numpy

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from IPython.display import clear_output
import time

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

## Cell 2: Load and Preprocess MNIST Dataset
**What:** Load MNIST dataset and preprocess it for GAN training
**Why:** We need to normalize pixel values to [-1, 1] range and prepare batches for efficient training

```python
# Load and preprocess MNIST dataset
(train_images, train_labels), (_, _) = keras.datasets.mnist.load_data()

# Normalize images to [-1, 1] range (important for GAN training)
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Create dataset pipeline
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

print(f"Dataset shape: {train_images.shape}")
print(f"Pixel value range: {train_images.min()} to {train_images.max()}")
```

## Cell 3: Define Generator Network
**What:** Create the generator network that transforms random noise into fake images
**Why:** The generator learns to create realistic images from random noise vectors

```python
def make_generator_model():
    model = tf.keras.Sequential()
    
    # Start with a dense layer and reshape
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Reshape to start the convolutional stack
    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)
    
    # Upsample to 14x14
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Upsample to 14x14
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    
    # Upsample to 28x28 (final image size)
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)
    
    return model

# Create generator
generator = make_generator_model()

# Test generator with random noise
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
plt.title('Generated Image (Untrained)')
plt.axis('off')
plt.show()

print("Generator created successfully!")
generator.summary()
```

## Cell 4: Define Discriminator Network
**What:** Create the discriminator network that classifies images as real or fake
**Why:** The discriminator learns to distinguish between real and generated images, providing feedback to the generator

```python
def make_discriminator_model():
    model = tf.keras.Sequential()
    
    # Input layer
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))
    
    # Flatten and output
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    
    return model

# Create discriminator
discriminator = make_discriminator_model()

# Test discriminator
decision = discriminator(generated_image)
print(f"Discriminator decision on generated image: {decision}")

discriminator.summary()
```

## Cell 5: Define Loss Functions
**What:** Define loss functions for both generator and discriminator
**Why:** We use binary crossentropy loss to train the adversarial networks

```python
# Binary crossentropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    """
    Discriminator loss: wants to classify real images as 1 and fake images as 0
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    """
    Generator loss: wants discriminator to classify fake images as real (1)
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)

print("Loss functions defined successfully!")
```

## Cell 6: Define Optimizers
**What:** Create Adam optimizers for both networks
**Why:** Adam optimizer works well for GAN training with appropriate learning rates

```python
# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

print("Optimizers created successfully!")
print(f"Generator optimizer: {generator_optimizer}")
print(f"Discriminator optimizer: {discriminator_optimizer}")
```

## Cell 7: Define Training Step
**What:** Define the core training step that updates both networks
**Why:** This implements the adversarial training process where both networks compete

```python
# Training parameters
noise_dim = 100
num_examples_to_generate = 16

# Create seed for consistent image generation during training
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    # Calculate gradients
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    # Apply gradients
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    
    return gen_loss, disc_loss

print("Training step function defined successfully!")
```

## Cell 8: Image Generation and Visualization Function
**What:** Create function to generate and display images during training
**Why:** Visual feedback helps monitor training progress and quality

```python
def generate_and_save_images(model, epoch, test_input):
    """Generate images and display them"""
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(4, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.suptitle(f'Epoch {epoch}')
    plt.tight_layout()
    plt.show()

# Test the function
print("Testing image generation function...")
generate_and_save_images(generator, 0, seed)
```

## Cell 9: Main Training Loop
**What:** Implement the main training loop for the GAN
**Why:** This orchestrates the entire training process with progress monitoring

```python
def train(dataset, epochs):
    """Main training function"""
    gen_losses = []
    disc_losses = []
    
    for epoch in range(epochs):
        start = time.time()
        
        # Train on each batch
        epoch_gen_loss = []
        epoch_disc_loss = []
        
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            epoch_gen_loss.append(gen_loss)
            epoch_disc_loss.append(disc_loss)
        
        # Record average losses
        avg_gen_loss = tf.reduce_mean(epoch_gen_loss)
        avg_disc_loss = tf.reduce_mean(epoch_disc_loss)
        gen_losses.append(avg_gen_loss)
        disc_losses.append(avg_disc_loss)
        
        # Generate images every 5 epochs
        if (epoch + 1) % 5 == 0:
            clear_output(wait=True)
            generate_and_save_images(generator, epoch + 1, seed)
        
        # Print progress
        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Generator Loss: {avg_gen_loss:.4f}')
        print(f'Discriminator Loss: {avg_disc_loss:.4f}')
        print(f'Time for epoch: {time.time()-start:.2f} sec')
        print('-' * 50)
    
    return gen_losses, disc_losses

print("Training function defined. Ready to start training!")
```

## Cell 10: Start Training
**What:** Execute the training process
**Why:** Train the GAN for a specified number of epochs

```python
# Training parameters
EPOCHS = 50

print("Starting GAN training...")
print(f"Training for {EPOCHS} epochs")
print(f"Batch size: {BATCH_SIZE}")
print(f"Dataset size: {len(train_images)}")

# Start training
gen_losses, disc_losses = train(train_dataset, EPOCHS)

print("Training completed!")
```

## Cell 11: Plot Training Losses
**What:** Visualize the training progress through loss curves
**Why:** Loss curves help understand training dynamics and stability

```python
# Plot training losses
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(gen_losses, label='Generator Loss', color='blue')
plt.plot(disc_losses, label='Discriminator Loss', color='red')
plt.title('Training Losses')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(gen_losses, label='Generator Loss', color='blue')
plt.title('Generator Loss Over Time')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"Final Generator Loss: {gen_losses[-1]:.4f}")
print(f"Final Discriminator Loss: {disc_losses[-1]:.4f}")
```

## Cell 12: Generate Final Results
**What:** Generate a grid of final images and test with different noise inputs
**Why:** Demonstrate the final capability of the trained generator

```python
# Generate final results
print("Generating final results...")

# Generate with the seed we've been using
generate_and_save_images(generator, EPOCHS, seed)

# Generate completely new images
print("\nGenerating new random images...")
random_noise = tf.random.normal([16, noise_dim])
generate_and_save_images(generator, "Final", random_noise)

# Generate a large batch for variety
print("\nGenerating more samples...")
large_noise = tf.random.normal([25, noise_dim])
predictions = generator(large_noise, training=False)

plt.figure(figsize=(8, 8))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
    plt.axis('off')

plt.suptitle('Generated MNIST Digits - Final Results')
plt.tight_layout()
plt.show()

print("GAN training and evaluation completed successfully!")
```

## Cell 13: Save the Model (Optional)
**What:** Save the trained generator model
**Why:** Preserve the trained model for future use

```python
# Save the trained generator
generator.save('mnist_generator.h5')
print("Generator model saved as 'mnist_generator.h5'")

# You can also save the discriminator if needed
# discriminator.save('mnist_discriminator.h5')

# To load the model later, use:
# loaded_generator = tf.keras.models.load_model('mnist_generator.h5')
```

## Training Tips and Notes:

1. **Training Time**: Each epoch takes about 30-60 seconds on GPU, 2-3 minutes on CPU
2. **Loss Behavior**: Generator and discriminator losses should fluctuate but remain relatively balanced
3. **Quality Improvement**: Image quality typically improves significantly after 20-30 epochs
4. **Hyperparameter Tuning**: You can experiment with learning rates, batch sizes, and network architectures
5. **Mode Collapse**: If all generated images look similar, try reducing learning rates or adjusting network architecture

This implementation provides a complete, working GAN that you can run cell by cell in Google Colab!
