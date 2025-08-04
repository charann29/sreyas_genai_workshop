# GAN Image Generation - Experiment 5 
## Cell 1: Install and Import Libraries
**What:** Install TensorFlow and import necessary libraries  
**Why:** Required dependencies for GAN implementation

```python
!pip install tensorflow matplotlib

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

tf.random.set_seed(42)
np.random.seed(42)
```

---

## Cell 2: Load and Preprocess Dataset
**What:** Load CIFAR-10, filter one class, and normalize data  
**Why:** Custom dataset preparation for GAN training

```python
# Load CIFAR-10 and filter automobiles (class 1)
(x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()
x_train = x_train[y_train.flatten() == 1]  # Filter automobiles only
x_train = (x_train.astype('float32') - 127.5) / 127.5  # Normalize to [-1,1]

# Create dataset
BATCH_SIZE = 64
dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(1000).batch(BATCH_SIZE, drop_remainder=True)

print(f"Training samples: {len(x_train)}, Batches: {len(x_train)//BATCH_SIZE}")
```

---

## Cell 3: Define Generator
**What:** Create generator network that produces images from noise  
**Why:** Core component that learns to generate realistic images

```python
def make_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(4*4*512, input_shape=(100,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Reshape((4, 4, 512)),
        
        tf.keras.layers.Conv2DTranspose(256, 4, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        
        tf.keras.layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        
        tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same', activation='tanh')
    ])
    return model

generator = make_generator()
```

---

## Cell 4: Define Discriminator
**What:** Create discriminator network that classifies real vs fake images  
**Why:** Provides adversarial feedback to improve generator

```python
def make_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 4, strides=2, padding='same', input_shape=[32, 32, 3]),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(128, 4, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv2D(256, 4, strides=2, padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(0.2),
        
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1)
    ])
    return model

discriminator = make_discriminator()
```

---

## Cell 5: Define Loss Functions and Optimizers
**What:** Set up binary crossentropy loss and Adam optimizers  
**Why:** Essential components for GAN training with different learning rates

```python
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# Experiment with different learning rates
gen_optimizer = tf.keras.optimizers.Adam(0.0001, beta_1=0.5)  # Slower generator
disc_optimizer = tf.keras.optimizers.Adam(0.0004, beta_1=0.5)  # Faster discriminator
```

---

## Cell 6: Training Step
**What:** Define single training step with gradient updates  
**Why:** Core training logic for adversarial learning

```python
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, 100])
    
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    
    gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    
    gen_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    disc_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
    
    return gen_loss, disc_loss
```

---

## Cell 7: Training Loop
**What:** Main training loop with progress tracking  
**Why:** Execute GAN training and monitor loss progression

```python
def train_gan(epochs):
    # Fixed noise for consistent visualization
    seed = tf.random.normal([16, 100])
    
    for epoch in range(epochs):
        gen_loss_avg = tf.keras.metrics.Mean()
        disc_loss_avg = tf.keras.metrics.Mean()
        
        for image_batch in tqdm(dataset, desc=f'Epoch {epoch+1}/{epochs}'):
            gen_loss, disc_loss = train_step(image_batch)
            gen_loss_avg.update_state(gen_loss)
            disc_loss_avg.update_state(disc_loss)
        
        print(f'Epoch {epoch+1}: Gen Loss = {gen_loss_avg.result():.4f}, Disc Loss = {disc_loss_avg.result():.4f}')
        
        # Show generated images every 10 epochs
        if (epoch + 1) % 10 == 0:
            generated_images = generator(seed, training=False)
            
            plt.figure(figsize=(10, 10))
            for i in range(16):
                plt.subplot(4, 4, i+1)
                img = (generated_images[i] * 127.5 + 127.5).numpy().astype('uint8')
                plt.imshow(img)
                plt.axis('off')
            plt.suptitle(f'Generated Images - Epoch {epoch+1}')
            plt.show()

# Train the GAN
train_gan(50)
```

---

## Cell 8: Hyperparameter Experiments
**What:** Test different loss functions and learning rates  
**Why:** Compare performance with different configurations

```python
# Experiment 1: Standard vs LSGAN loss
def lsgan_disc_loss(real_output, fake_output):
    real_loss = tf.reduce_mean(tf.square(real_output - 1))
    fake_loss = tf.reduce_mean(tf.square(fake_output))
    return 0.5 * (real_loss + fake_loss)

def lsgan_gen_loss(fake_output):
    return 0.5 * tf.reduce_mean(tf.square(fake_output - 1))

# Reset models and try LSGAN
generator = make_generator()
discriminator = make_discriminator()

# Update loss functions for experiment
discriminator_loss = lsgan_disc_loss
generator_loss = lsgan_gen_loss

print("Training with LSGAN loss...")
train_gan(30)
```

---

## Cell 9: Final Results
**What:** Generate final samples and compare configurations  
**Why:** Evaluate final model performance

```python
# Generate diverse samples
noise = tf.random.normal([25, 100])
generated_images = generator(noise, training=False)

plt.figure(figsize=(12, 12))
for i in range(25):
    plt.subplot(5, 5, i+1)
    img = (generated_images[i] * 127.5 + 127.5).numpy().astype('uint8')
    plt.imshow(img)
    plt.axis('off')
plt.suptitle('Final Generated Samples')
plt.show()

print("Experiment 5 completed! Key findings:")
print("- Different learning rates help balance generator/discriminator")
print("- LSGAN often provides more stable training than standard GAN")
print("- Batch normalization and dropout improve training stability")
```
