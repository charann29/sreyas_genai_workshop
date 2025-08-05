# Short GAN Code for Lab Exam

## Cell 1: Quick Setup
**What:** Import libraries and load data
**Why:** TensorFlow for neural networks, matplotlib for plotting, MNIST for digit images

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST data
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train - 127.5) / 127.5  # Normalize to [-1, 1]
x_train = x_train[..., np.newaxis]   # Add channel dimension

print("Data loaded:", x_train.shape)
```

## Cell 2: Build Generator (Fake Image Maker)
**What:** Creates fake images from random noise
**Why:** Dense = fully connected layer, Reshape = change shape, Conv2DTranspose = make image bigger

```python
def make_generator():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7*7*128, input_shape=(100,)),  # 100 noise → 7×7×128
        tf.keras.layers.Reshape((7, 7, 128)),                # Make it 2D image
        tf.keras.layers.Conv2DTranspose(64, 4, 2, 'same'),   # 7×7 → 14×14
        tf.keras.layers.Conv2DTranspose(1, 4, 2, 'same', activation='tanh')  # 14×14 → 28×28
    ])
    return model

generator = make_generator()
print("Generator built!")
```

## Cell 3: Build Discriminator (Real/Fake Detector)
**What:** Tells if image is real or fake
**Why:** Conv2D = find patterns, Flatten = make 1D, Dense = final decision

```python
def make_discriminator():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(64, 4, 2, 'same', input_shape=(28, 28, 1)),  # 28×28 → 14×14
        tf.keras.layers.Conv2D(128, 4, 2, 'same'),           # 14×14 → 7×7
        tf.keras.layers.Flatten(),                           # Make 1D
        tf.keras.layers.Dense(1)                             # Real=1, Fake=0
    ])
    return model

discriminator = make_discriminator()
print("Discriminator built!")
```

## Cell 4: Simple Training (Key Concepts)
**What:** Train both networks to compete
**Why:** Generator tries to fool discriminator, discriminator tries to catch fakes

```python
# Loss functions
bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def d_loss(real_out, fake_out):
    real_loss = bce(tf.ones_like(real_out), real_out)    # Real should be 1
    fake_loss = bce(tf.zeros_like(fake_out), fake_out)   # Fake should be 0
    return real_loss + fake_loss

def g_loss(fake_out):
    return bce(tf.ones_like(fake_out), fake_out)         # Want fake to look real

# Optimizers
g_opt = tf.keras.optimizers.Adam(0.0002)
d_opt = tf.keras.optimizers.Adam(0.0002)

print("Training setup complete!")
```

## Cell 5: Quick Training Loop
**What:** Train for few epochs with basic loop
**Why:** GradientTape records gradients, apply_gradients updates weights

```python
@tf.function
def train_step(real_images):
    noise = tf.random.normal([len(real_images), 100])
    
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        fake_images = generator(noise)
        real_out = discriminator(real_images)
        fake_out = discriminator(fake_images)
        
        g_loss_val = g_loss(fake_out)
        d_loss_val = d_loss(real_out, fake_out)
    
    g_grads = g_tape.gradient(g_loss_val, generator.trainable_variables)
    d_grads = d_tape.gradient(d_loss_val, discriminator.trainable_variables)
    
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

# Quick training (reduce epochs for exam)
dataset = tf.data.Dataset.from_tensor_slices(x_train).batch(128)

for epoch in range(10):  # Just 10 epochs for exam
    for batch in dataset:
        train_step(batch)
    print(f"Epoch {epoch+1} done")

print("Training complete!")
```

## Cell 6: Generate and Show Results (EXAM FOCUS)
**What:** Create fake images and display them
**Why:** This is what examiners want to see - visual results!

```python
# Generate diverse samples (MAIN EXAM CODE)
noise = tf.random.normal([25, 100])                    # 25 random noise vectors
generated_images = generator(noise, training=False)    # Generate fake images

# Display results
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    # Convert from [-1,1] to [0,255] for display
    img = (generated_images[i] * 127.5 + 127.5).numpy().astype('uint8')
    plt.imshow(img.squeeze(), cmap='gray')  # squeeze removes channel dimension
    plt.axis('off')

plt.suptitle('Generated MNIST Digits', fontsize=16)
plt.tight_layout()
plt.show()

# Print key findings (memorize this!)
print("Key GAN Concepts:")
print("✓ Generator: Noise → Fake Images")
print("✓ Discriminator: Image → Real/Fake")
print("✓ Adversarial Training: They compete!")
print("✓ Loss balancing is crucial")
```

## Memory Tips for Exam:

### **Quick Formulas to Remember:**
```python
# Data normalization
data = (data - 127.5) / 127.5    # [-1, 1] range

# Image denormalization for display  
display_img = (img * 127.5 + 127.5)  # [0, 255] range

# Generator loss (wants fake to be real)
g_loss = bce(ones, fake_predictions)

# Discriminator loss (real=1, fake=0)
d_loss = bce(ones, real_pred) + bce(zeros, fake_pred)
```

### **Key Architecture Pattern:**
- **Generator:** Dense → Reshape → Conv2DTranspose (upsample)
- **Discriminator:** Conv2D → Conv2D → Flatten → Dense

### **Essential Steps:**
1. Load data and normalize
2. Build Generator (noise → image)
3. Build Discriminator (image → real/fake)
4. Define losses and optimizers
5. Train with GradientTape
6. Generate and display results

### **Common Exam Questions:**
- "What does Conv2DTranspose do?" → **Upsamples/enlarges images**
- "Why normalize to [-1,1]?" → **Matches tanh activation range**
- "What is adversarial training?" → **Two networks competing**
- "How to display generated images?" → **Denormalize and use plt.imshow()**

This shortened version focuses on the core concepts and the specific code you showed, making it perfect for lab exam memorization!
