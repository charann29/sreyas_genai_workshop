# Minimal LSTM Text Generation Tutorial - Google Colab


## Cell 1: Setup and Data Preparation

**Purpose**: Install dependencies and prepare text data for LSTM training  
**Why**: LSTM needs clean, tokenized text data in numerical format

```python
# Install and import everything we need
!pip install tensorflow requests matplotlib

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import requests
import re
import matplotlib.pyplot as plt

# Set seeds for reproducibility (important for consistent results)
tf.random.set_seed(42)
np.random.seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"GPU available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# Download text data (Alice in Wonderland - good for learning patterns)
url = "https://www.gutenberg.org/files/11/11-0.txt"
try:
    response = requests.get(url)
    text = response.text
    
    # Clean the text - keep only the story part
    start_idx = text.find("CHAPTER I. Down the Rabbit-Hole")
    end_idx = text.find("End of the Project Gutenberg EBook")
    if start_idx != -1 and end_idx != -1:
        text = text[start_idx:end_idx]
    
    print("✓ Successfully downloaded Alice in Wonderland")
except:
    # Fallback text if download fails
    text = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.""" * 100
    print("⚠ Using fallback text (download failed)")

# Basic text cleaning (essential preprocessing)
text = re.sub(r'\r\n', ' ', text)  # Remove line breaks
text = re.sub(r'\s+', ' ', text)   # Remove extra spaces
text = text.strip()

print(f"Text length: {len(text)} characters")
print(f"Sample: {text[:200]}...")
print(f"Unique characters: {len(set(text))}")
```

**Expected Output:**
- TensorFlow version information
- GPU availability status
- Text download confirmation
- Text statistics (length, sample, unique characters)

---

## Cell 2: Create Training Data

**Purpose**: Convert text to numerical sequences for LSTM training  
**Why**: Neural networks only understand numbers, need input-output pairs

```python
# Create character mappings (vocabulary)
chars = sorted(list(set(text)))  # Get unique characters
char_to_idx = {char: idx for idx, char in enumerate(chars)}  # char → number
idx_to_char = {idx: char for idx, char in enumerate(chars)}  # number → char
vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")
print(f"Characters: {chars[:20]}...")  # Show first 20 characters

# Convert entire text to numbers
text_as_int = [char_to_idx[c] for c in text]
print(f"Text converted to {len(text_as_int)} integers")

# Create training sequences (sliding window approach)
sequence_length = 100  # Look back 100 characters to predict next one
step_size = 3         # Skip 3 chars between sequences (creates overlap)

sequences = []
next_chars = []

# Create input-output pairs
for i in range(0, len(text_as_int) - sequence_length, step_size):
    sequences.append(text_as_int[i:i + sequence_length])      # Input: 100 chars
    next_chars.append(text_as_int[i + sequence_length])       # Output: next char

# Convert to NumPy arrays
X = np.array(sequences)
y = tf.keras.utils.to_categorical(next_chars, num_classes=vocab_size)  # One-hot encode

# Split data (80% train, 20% validation)
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Training sequences: {len(X_train)}")
print(f"Validation sequences: {len(X_val)}")
print(f"Input shape: {X.shape}, Output shape: {y.shape}")

# Show example sequence
print(f"\nExample input sequence: {''.join([idx_to_char[i] for i in X[0][:50]])}...")
print(f"Target character: '{idx_to_char[next_chars[0]]}')")
```

**Key Concepts:**
- **Character mapping**: Converting text to numerical indices
- **Sliding window**: Creating overlapping sequences for training
- **One-hot encoding**: Converting target characters to probability vectors
- **Train/validation split**: Preventing overfitting

---

## Cell 3: Build and Train Model

**Purpose**: Create LSTM architecture and train on text sequences  
**Why**: LSTM can remember long-term patterns in text for better generation

```python
# Build LSTM model (Sequential architecture)
model = models.Sequential([
    # Embedding: Convert character indices to dense vectors (256-dim)
    layers.Embedding(vocab_size, 256, input_length=sequence_length),
    
    # LSTM layer 1: 512 units, return sequences for stacking
    layers.LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
    
    # LSTM layer 2: 512 units, final output
    layers.LSTM(512, dropout=0.3, recurrent_dropout=0.3),
    
    # Dense output: Probability distribution over vocabulary
    layers.Dense(vocab_size, activation='softmax')
])

# Compile model with optimizer and loss function
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),  # Adaptive learning rate
    loss='categorical_crossentropy',                 # Multi-class classification
    metrics=['accuracy']                             # Track prediction accuracy
)

print("Model Architecture:")
model.summary()

# Set up training callbacks for better training
callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
]

# Train the model
print("\n🚀 Starting LSTM training...")
print("This will take 10-20 minutes depending on your hardware")

history = model.fit(
    X_train, y_train,
    batch_size=128,                    # Process 128 sequences at once
    epochs=25,                         # Maximum training iterations
    validation_data=(X_val, y_val),    # Monitor overfitting
    callbacks=callbacks_list,          # Smart training control
    verbose=1
)

print("✅ Training completed!")

# Plot training history
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Final model performance
final_loss = history.history['val_loss'][-1]
final_acc = history.history['val_accuracy'][-1]
print(f"Final validation loss: {final_loss:.4f}")
print(f"Final validation accuracy: {final_acc:.4f} ({final_acc*100:.2f}%)")
```

**Model Architecture Explained:**
- **Embedding Layer**: Converts character indices to dense vector representations
- **LSTM Layers**: Two stacked LSTM layers with 512 units each for pattern learning
- **Dropout**: Prevents overfitting by randomly setting inputs to 0
- **Dense Output**: Final layer with softmax activation for character probability distribution

**Training Features:**
- **Early Stopping**: Stops training when validation loss stops improving
- **Learning Rate Reduction**: Automatically reduces learning rate when stuck
- **Validation Monitoring**: Tracks performance on unseen data

---

## Cell 4: Generate Text

**Purpose**: Use trained LSTM to generate new text sequences  
**Why**: Demonstrates practical application of the trained model

```python
def generate_text(seed_text, length=500, temperature=1.0):
    """
    Generate text using the trained LSTM model
    
    Args:
        seed_text: Starting text to generate from
        length: Number of characters to generate  
        temperature: Controls randomness (0.5=safe, 1.0=balanced, 1.5=creative)
    
    Returns:
        Generated text string
    """
    
    # Prepare seed text (pad or truncate to sequence_length)
    if len(seed_text) < sequence_length:
        seed_text = ' ' * (sequence_length - len(seed_text)) + seed_text
    else:
        seed_text = seed_text[-sequence_length:]
    
    # Convert to numerical sequence
    current_sequence = [char_to_idx.get(char, 0) for char in seed_text]
    generated_text = seed_text
    
    print(f"🎯 Generating {length} characters with temperature {temperature}")
    print(f"📝 Seed: '{seed_text[-50:]}'")
    print("📖 Generated text:")
    print("-" * 50)
    
    # Generate character by character
    for i in range(length):
        # Prepare input for model
        x_input = np.array(current_sequence).reshape(1, sequence_length)
        
        # Get prediction probabilities
        predictions = model.predict(x_input, verbose=0)[0]
        
        # Apply temperature sampling
        if temperature == 0:
            # Deterministic: always pick most likely
            next_char_idx = np.argmax(predictions)
        else:
            # Stochastic: sample based on probabilities
            predictions = np.log(predictions + 1e-8) / temperature
            exp_predictions = np.exp(predictions)
            predictions = exp_predictions / np.sum(exp_predictions)
            next_char_idx = np.random.choice(len(predictions), p=predictions)
        
        # Convert back to character
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char
        
        # Update sequence (sliding window)
        current_sequence = current_sequence[1:] + [next_char_idx]
        
        # Print character in real-time
        print(next_char, end='', flush=True)
    
    print("\n" + "-" * 50)
    return generated_text

# Test text generation with different settings
test_seeds = [
    "Alice was beginning to get very tired",
    "The Queen of Hearts said",
    "Down the rabbit hole Alice fell"
]

temperatures = [0.5, 1.0, 1.5]

print("🔥 TEXT GENERATION EXPERIMENTS")
print("=" * 60)

# Test different temperature settings
print("\n1️⃣ TEMPERATURE COMPARISON:")
seed = test_seeds[0]
for temp in temperatures:
    print(f"\n🌡️  Temperature: {temp}")
    generated = generate_text(seed, length=200, temperature=temp)
    
# Test different seed texts  
print(f"\n2️⃣ DIFFERENT SEED TEXTS (Temperature=1.0):")
for i, seed in enumerate(test_seeds, 1):
    print(f"\n🌱 Seed {i}: '{seed}'")
    generated = generate_text(seed, length=200, temperature=1.0)

# Interactive generation function
def interactive_generation():
    """Allow user to input custom seeds and parameters"""
    print("\n🎮 INTERACTIVE MODE")
    print("Enter your own seed text and parameters!")
    
    while True:
        try:
            seed = input("\n📝 Enter seed text (or 'quit' to exit): ")
            if seed.lower() == 'quit':
                break
                
            length = int(input("📏 Enter length (default 300): ") or "300")
            temp = float(input("🌡️ Enter temperature 0.5-2.0 (default 1.0): ") or "1.0")
            
            generated = generate_text(seed, length, temp)
            print(f"\n✨ Complete text:\n{generated}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Error: {e}")

print("\n🎉 LSTM Text Generator Complete!")
print("✅ Model successfully trained on text sequences")
print("✅ Can generate coherent text continuations")
print("✅ Temperature controls creativity vs coherence")

# Save the model for later use
model.save('lstm_text_generator.h5')
print("\n💾 Model saved as 'lstm_text_generator.h5'")

# Uncomment to run interactive mode
# interactive_generation()
```

**Temperature Parameter Explained:**
- **0.5 (Conservative)**: More predictable, safer text generation
- **1.0 (Balanced)**: Good balance between creativity and coherence
- **1.5 (Creative)**: More random and creative, potentially less coherent
- **0.0 (Deterministic)**: Always picks the most likely next character

---

## Key Concepts Summary

### LSTM Architecture
- **Long Short-Term Memory**: Can remember patterns over long sequences
- **Embedding Layer**: Learns dense representations of characters
- **Stacked LSTMs**: Multiple layers for complex pattern recognition
- **Dropout**: Regularization technique to prevent overfitting

### Training Process
- **Sequence Learning**: Model learns to predict next character given previous characters
- **Categorical Cross-entropy**: Loss function for multi-class classification
- **Adam Optimizer**: Adaptive learning rate optimization
- **Callbacks**: Smart training control (early stopping, learning rate scheduling)

### Text Generation
- **Autoregressive Generation**: Use model's own predictions as input for next prediction
- **Temperature Sampling**: Control randomness in text generation
- **Sliding Window**: Maintain context window during generation

---


## Expected Results

After training, the model should achieve:
- **Validation accuracy**: 50-70% (character-level prediction)
- **Generated text**: Coherent sentences following Alice in Wonderland style
- **Pattern recognition**: Proper capitalization, punctuation, and word structure

## Troubleshooting

- **Low GPU memory**: Reduce batch_size or LSTM units
- **Poor text quality**: Increase training epochs or use larger dataset
- **Overfitting**: Increase dropout rates or reduce model complexity
- **Slow training**: Ensure GPU is available and properly configured

## Extensions

- **Word-level generation**: Use word tokenization instead of characters
- **Different datasets**: Train on different text genres
- **Beam search**: Implement more sophisticated text generation
- **Fine-tuning**: Continue training on specific text styles
