# LSTM Network for Text Generation - Complete Google Colab Tutorial

## Cell 1: Install and Import Dependencies

**What:** Set up the environment with all necessary libraries for LSTM text generation
**Why:** We need TensorFlow for deep learning, NumPy for numerical operations, requests for data download, and other utilities for text processing

```python
# Install required packages
!pip install tensorflow matplotlib numpy requests

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import numpy as np
import matplotlib.pyplot as plt
import requests
import re
import random
from collections import Counter
import pickle

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
```

**Technical Explanation:**
- **tensorflow.keras.layers**: Provides neural network building blocks (LSTM, Dense, Embedding)
- **tensorflow.keras.models**: Sequential model container for stacking layers
- **tensorflow.keras.optimizers**: Training algorithms (Adam optimizer)
- **tensorflow.keras.callbacks**: Training monitoring tools (EarlyStopping, ModelCheckpoint)
- **requests**: HTTP library for downloading text data from Project Gutenberg
- **re**: Regular expressions for text cleaning and preprocessing
- **pickle**: Python serialization for saving/loading character mappings
- **Random seeds**: Ensure reproducible results across multiple runs

---

## Cell 2: Download and Prepare Text Dataset

**What:** Download Alice in Wonderland from Project Gutenberg and preprocess it for training
**Why:** We need a substantial, varied text corpus to train the LSTM. Alice in Wonderland provides rich vocabulary and diverse sentence structures

```python
def download_text_data():
    """
    Download Alice in Wonderland from Project Gutenberg
    This is a good dataset because:
    - Public domain (no copyright issues)
    - Rich vocabulary and varied sentence structures
    - Sufficient length for training (~150,000 characters)
    """
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    
    try:
        response = requests.get(url)
        text = response.text
        
        # Remove Project Gutenberg header and footer
        start_marker = "CHAPTER I. Down the Rabbit-Hole"
        end_marker = "End of the Project Gutenberg EBook"
        
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            text = text[start_idx:end_idx]
        
        # Clean the text
        text = re.sub(r'\r\n', ' ', text)  # Replace line breaks with spaces
        text = re.sub(r'\s+', ' ', text)   # Replace multiple spaces with single space
        text = text.strip()
        
        print(f"Downloaded text length: {len(text)} characters")
        print(f"First 200 characters: {text[:200]}")
        
        return text
        
    except Exception as e:
        print(f"Error downloading text: {e}")
        # Fallback to a smaller sample text
        return """
        The quick brown fox jumps over the lazy dog. This is a sample text for training
        our LSTM model. Machine learning is fascinating and powerful. Deep learning
        enables us to create models that can understand and generate human-like text.
        Natural language processing has many applications in real world scenarios.
        Artificial intelligence is revolutionizing how we interact with technology.
        Neural networks can learn complex patterns from data and make predictions.
        """ * 50  # Repeat to make it longer

# Download and prepare the dataset
raw_text = download_text_data()
```

**Technical Explanation:**
- **requests.get()**: HTTP GET request to download raw text file
- **text.find()**: Locates specific markers to extract only the story content
- **re.sub(r'\r\n', ' ', text)**: Regular expression replaces Windows line endings with spaces
- **re.sub(r'\s+', ' ', text)**: Collapses multiple whitespace characters into single spaces
- **text.strip()**: Removes leading/trailing whitespace
- **Fallback mechanism**: Provides backup text if download fails, ensuring code always works

---

## Cell 3: Create Character-Level Tokenization

**What:** Convert text to numerical format and create bidirectional character mappings
**Why:** Neural networks only process numbers, so we need to convert characters to integers and back

```python
def create_char_mappings(text):
    """
    Create character-to-index and index-to-character mappings
    Why character-level?
    - Smaller vocabulary (typically 50-100 unique characters)
    - Can generate new words and handle typos
    - Good for learning fundamental language patterns
    """
    # Get unique characters and sort them for consistency
    chars = sorted(list(set(text)))
    
    # Create bidirectional mappings
    char_to_idx = {char: idx for idx, char in enumerate(chars)}
    idx_to_char = {idx: char for idx, char in enumerate(chars)}
    
    vocab_size = len(chars)
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Characters: {chars}")
    
    return char_to_idx, idx_to_char, vocab_size

# Create character mappings
char_to_idx, idx_to_char, vocab_size = create_char_mappings(raw_text)

# Convert text to numerical sequences
text_as_int = [char_to_idx[c] for c in raw_text]
print(f"Text converted to {len(text_as_int)} integers")
```

**Technical Explanation:**
- **set(text)**: Creates set of unique characters, removing duplicates
- **sorted()**: Ensures consistent ordering across different runs
- **Dictionary comprehension**: {char: idx for idx, char in enumerate(chars)} creates mapping
- **enumerate()**: Provides both index and character in the loop
- **Bidirectional mappings**: char_to_idx for encoding, idx_to_char for decoding
- **List comprehension**: [char_to_idx[c] for c in raw_text] converts entire text to integers
- **Character-level advantages**: Handles unknown words, learns spelling patterns, smaller vocabulary

---

## Cell 4: Create Training Sequences

**What:** Create input-output pairs where input is sequence of characters, output is next character
**Why:** LSTM needs fixed-length sequences for training, and we train it to predict next character given context

```python
def create_training_sequences(text_as_int, sequence_length=100, step_size=3):
    """
    Create training sequences for the LSTM
    
    Args:
    - sequence_length: How many characters to look back (context window)
    - step_size: How many characters to skip between sequences (for efficiency)
    
    Why these parameters?
    - sequence_length=100: Gives model enough context to learn patterns
    - step_size=3: Creates overlapping sequences for better training coverage
    """
    sequences = []
    next_chars = []
    
    # Create sequences by sliding a window across the text
    for i in range(0, len(text_as_int) - sequence_length, step_size):
        sequences.append(text_as_int[i:i + sequence_length])
        next_chars.append(text_as_int[i + sequence_length])
    
    print(f"Created {len(sequences)} training sequences")
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = tf.keras.utils.to_categorical(next_chars, num_classes=vocab_size)
    
    print(f"Input shape: {X.shape}")
    print(f"Output shape: {y.shape}")
    
    return X, y

# Create training data
sequence_length = 100  # Look back 100 characters
X, y = create_training_sequences(text_as_int, sequence_length)

# Split into train and validation sets
split_idx = int(0.8 * len(X))
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")
```

**Technical Explanation:**
- **Sliding window**: for i in range(0, len(text_as_int) - sequence_length, step_size) creates overlapping sequences
- **sequence_length=100**: Context window size - how many previous characters model sees
- **step_size=3**: Skip size between sequences - creates overlapping training examples
- **text_as_int[i:i + sequence_length]**: Input sequence (100 characters)
- **text_as_int[i + sequence_length]**: Target character (the next character to predict)
- **to_categorical()**: Converts integer labels to one-hot encoded vectors for classification
- **One-hot encoding**: [0,0,1,0,0...] where 1 indicates the correct character class
- **80/20 split**: Standard train/validation split for monitoring overfitting

---

## Cell 5: Build LSTM Architecture

**What:** Create a sequential model with embedding, LSTM layers, and output layer for character prediction
**Why:** LSTM can learn long-term dependencies in text sequences, embedding helps with character relationships

```python
def build_lstm_model(vocab_size, sequence_length, embedding_dim=256, lstm_units=512):
    """
    Build LSTM model for text generation
    
    Architecture explanation:
    1. Embedding Layer: Converts character indices to dense vectors
       - Why: Helps model learn relationships between characters
       - embedding_dim=256: Good balance between capacity and efficiency
    
    2. LSTM Layers: Learn sequential patterns
       - lstm_units=512: Large enough to capture complex patterns
       - return_sequences=True: Needed for stacking LSTM layers
       - dropout=0.3: Prevents overfitting
    
    3. Dense Output: Predicts probability distribution over vocabulary
       - vocab_size neurons: One for each possible character
       - softmax activation: Outputs probabilities that sum to 1
    """
    
    model = models.Sequential([
        # Embedding layer: maps character indices to dense vectors
        layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=sequence_length,
            name='embedding'
        ),
        
        # First LSTM layer with return_sequences=True for stacking
        layers.LSTM(
            units=lstm_units,
            return_sequences=True,
            dropout=0.3,
            recurrent_dropout=0.3,
            name='lstm_1'
        ),
        
        # Second LSTM layer (final layer doesn't return sequences)
        layers.LSTM(
            units=lstm_units,
            dropout=0.3,
            recurrent_dropout=0.3,
            name='lstm_2'
        ),
        
        # Dense layer for classification
        layers.Dense(vocab_size, activation='softmax', name='output')
    ])
    
    return model

# Build the model
model = build_lstm_model(vocab_size, sequence_length)

# Display model architecture
model.summary()

# Compile model with appropriate loss and optimizer
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),  # Adam is generally good for RNNs
    loss='categorical_crossentropy',  # Multi-class classification
    metrics=['accuracy']
)
```

**Technical Explanation:**
- **Embedding Layer**: Converts sparse integer indices to dense vector representations
  - **input_dim=vocab_size**: Number of unique characters in vocabulary
  - **output_dim=256**: Dimensionality of dense embedding vectors
  - **input_length=sequence_length**: Fixed input sequence length
- **LSTM Layer 1**: 
  - **units=512**: Number of LSTM cells (memory units)
  - **return_sequences=True**: Returns full sequence (needed for stacking)
  - **dropout=0.3**: Randomly sets 30% of inputs to 0 during training
  - **recurrent_dropout=0.3**: Applies dropout to recurrent connections
- **LSTM Layer 2**: Final LSTM layer without return_sequences (outputs single vector)
- **Dense Layer**: 
  - **vocab_size units**: One output neuron per character
  - **softmax activation**: Converts logits to probability distribution
- **Adam optimizer**: Adaptive learning rate optimization algorithm
- **categorical_crossentropy**: Loss function for multi-class classification
- **Learning rate=0.001**: Step size for gradient updates

---

## Cell 6: Set Up Training Callbacks

**What:** Configure callbacks for monitoring training progress and preventing overfitting
**Why:** Callbacks help optimize training by saving best models, adjusting learning rate, and stopping early

```python
# Create callbacks for better training
callbacks_list = [
    # Save best model based on validation loss
    callbacks.ModelCheckpoint(
        filepath='best_lstm_model.h5',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    ),
    
    # Reduce learning rate when validation loss plateaus
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # Reduce LR by half
        patience=3,  # Wait 3 epochs before reducing
        min_lr=1e-7,
        verbose=1
    ),
    
    # Stop training if no improvement for 8 epochs
    callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]

print("Callbacks configured:")
print("- Model checkpointing: Saves best model")
print("- Learning rate reduction: Adapts learning rate")
print("- Early stopping: Prevents overfitting")
```

**Technical Explanation:**
- **ModelCheckpoint**:
  - **filepath**: Where to save the best model
  - **monitor='val_loss'**: Metric to track for "best" model
  - **save_best_only=True**: Only saves when validation loss improves
  - **save_weights_only=False**: Saves entire model architecture + weights
- **ReduceLROnPlateau**:
  - **factor=0.5**: Multiply learning rate by 0.5 when triggered
  - **patience=3**: Wait 3 epochs without improvement before reducing
  - **min_lr=1e-7**: Minimum learning rate threshold
- **EarlyStopping**:
  - **patience=8**: Stop training after 8 epochs without improvement
  - **restore_best_weights=True**: Revert to best model weights when stopping
- **Callback benefits**: Automatic hyperparameter adjustment, prevents overfitting, saves computational resources

---

## Cell 7: Train the LSTM Model

**What:** Execute the training process with monitoring and callbacks
**Why:** This is where the model learns to predict next characters from sequential patterns

```python
print("Starting LSTM training...")
print("This may take 15-30 minutes depending on your hardware")
print("Watch the loss decrease and accuracy increase over epochs")

# Train the model
history = model.fit(
    X_train, y_train,
    batch_size=128,  # Batch size affects memory usage and training stability
    epochs=50,       # Maximum epochs (early stopping may end training sooner)
    validation_data=(X_val, y_val),
    callbacks=callbacks_list,
    verbose=1
)

print("Training completed!")

# Save the final model and mappings for later use
model.save('lstm_text_generator.h5')

# Save character mappings
with open('char_mappings.pkl', 'wb') as f:
    pickle.dump({
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size
    }, f)

print("Model and mappings saved!")
```

**Technical Explanation:**
- **model.fit()**: Main training method that performs backpropagation
- **batch_size=128**: Number of sequences processed simultaneously
  - **Larger batches**: More stable gradients, better GPU utilization
  - **Smaller batches**: More frequent updates, better generalization
- **epochs=50**: Maximum number of complete passes through training data
- **validation_data**: Separate data for monitoring overfitting
- **verbose=1**: Print progress during training
- **Backpropagation process**:
  1. Forward pass: Compute predictions and loss
  2. Backward pass: Calculate gradients
  3. Weight update: Adjust parameters using optimizer
- **model.save()**: Saves complete model architecture and trained weights
- **pickle.dump()**: Serializes Python objects for later loading

---

## Cell 8: Visualize Training Progress

**What:** Plot training/validation loss and accuracy curves to analyze learning
**Why:** Visual analysis helps identify overfitting, underfitting, and optimal stopping points

```python
def plot_training_history(history):
    """
    Plot training history to visualize learning progress
    
    What to look for:
    - Loss should decrease over time
    - Training and validation loss should be close (no overfitting)
    - Accuracy should increase over time
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training and validation loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss', color='blue')
    axes[0, 0].plot(history.history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Model Loss Over Time')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot training and validation accuracy
    axes[0, 1].plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    axes[0, 1].plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    axes[0, 1].set_title('Model Accuracy Over Time')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot learning rate (if it was reduced)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], label='Learning Rate', color='green')
        axes[1, 0].set_title('Learning Rate Over Time')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Summary statistics
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    axes[1, 1].text(0.1, 0.8, f'Final Training Loss: {final_train_loss:.4f}', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.7, f'Final Validation Loss: {final_val_loss:.4f}', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.6, f'Final Training Accuracy: {final_train_acc:.4f}', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.5, f'Final Validation Accuracy: {final_val_acc:.4f}', transform=axes[1, 1].transAxes)
    axes[1, 1].text(0.1, 0.4, f'Total Epochs: {len(history.history["loss"])}', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title('Training Summary')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('lstm_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot the training history
plot_training_history(history)
```

**Technical Explanation:**
- **matplotlib.pyplot.subplots()**: Creates grid of subplots for multiple visualizations
- **Loss curves**: Show how well model fits training data over time
  - **Decreasing loss**: Model is learning
  - **Gap between train/val**: Indicates overfitting degree
- **Accuracy curves**: Show percentage of correct next-character predictions
- **Learning rate plot**: Shows ReduceLROnPlateau callback effects
- **plt.yscale('log')**: Logarithmic scale for learning rate visualization
- **Training diagnostics**:
  - **Good training**: Both losses decrease, small train/val gap
  - **Overfitting**: Training loss much lower than validation loss
  - **Underfitting**: Both losses remain high and plateau early

---

## Cell 9: Implement Text Generation Function

**What:** Create function that uses trained model to generate new text character by character
**Why:** This demonstrates the practical application of the trained LSTM for creative text generation

```python
def generate_text(model, seed_text, length=500, temperature=1.0):
    """
    Generate text using the trained LSTM model
    
    Args:
    - seed_text: Starting text to generate from
    - length: Number of characters to generate
    - temperature: Controls randomness (0.5=conservative, 1.0=balanced, 1.5=creative)
    
    How it works:
    1. Convert seed text to numerical sequence
    2. Pad or truncate to match model's expected input length
    3. For each character to generate:
       - Predict probability distribution over vocabulary
       - Apply temperature to control randomness
       - Sample next character from distribution
       - Update input sequence with new character
    """
    
    # Prepare seed text
    if len(seed_text) < sequence_length:
        # Pad with spaces if too short
        seed_text = ' ' * (sequence_length - len(seed_text)) + seed_text
    else:
        # Use last sequence_length characters if too long
        seed_text = seed_text[-sequence_length:]
    
    # Convert to numerical format
    generated_text = seed_text
    current_sequence = [char_to_idx.get(char, 0) for char in seed_text]
    
    print(f"Generating {length} characters with temperature {temperature}")
    print(f"Seed text: '{seed_text[-50:]}'")  # Show last 50 chars of seed
    print("Generated text:")
    print("-" * 80)
    
    for i in range(length):
        # Reshape for model input
        x_input = np.array(current_sequence).reshape(1, sequence_length)
        
        # Get predictions
        predictions = model.predict(x_input, verbose=0)[0]
        
        # Apply temperature
        if temperature == 0:
            # Deterministic: choose most likely character
            next_char_idx = np.argmax(predictions)
        else:
            # Stochastic: sample based on probability distribution
            predictions = np.asarray(predictions).astype('float64')
            predictions = np.log(predictions + 1e-8) / temperature
            exp_predictions = np.exp(predictions)
            predictions = exp_predictions / np.sum(exp_predictions)
            
            # Sample from the distribution
            next_char_idx = np.random.choice(len(predictions), p=predictions)
        
        # Convert back to character
        next_char = idx_to_char[next_char_idx]
        generated_text += next_char
        
        # Update sequence for next iteration
        current_sequence = current_sequence[1:] + [next_char_idx]
        
        # Print character without newline for real-time generation display
        print(next_char, end='', flush=True)
    
    print("\n" + "-" * 80)
    return generated_text

# Test the generation function with different parameters
print("Testing text generation with different temperature settings:")
```

**Technical Explanation:**
- **Seed text preparation**: Ensures input matches model's expected sequence length
- **char_to_idx.get(char, 0)**: Maps characters to indices, using 0 for unknown characters
- **model.predict()**: Forward pass through network to get probability distribution
- **Temperature scaling**: Controls randomness in character selection
  - **Temperature = 0**: Deterministic (always pick most likely)
  - **Temperature < 1**: Conservative (favor likely characters)
  - **Temperature = 1**: Balanced (use model's natural probabilities)
  - **Temperature > 1**: Creative (more random, less predictable)
- **Temperature formula**: predictions = np.log(predictions + 1e-8) / temperature
  - **np.log()**: Convert probabilities to log space
  - **1e-8**: Small constant to avoid log(0)
  - **Division by temperature**: Scales the distribution
- **np.random.choice()**: Samples character based on probability distribution
- **Sliding window**: current_sequence[1:] + [next_char_idx] maintains fixed length

---

## Cell 10: Generate Sample Texts

**What:** Demonstrate model capabilities with different temperature settings and seed texts
**Why:** Shows how temperature affects creativity and coherence in generated text

```python
# Generate with different temperature settings
test_seeds = [
    "Alice was beginning to get very tired",
    "The Queen of Hearts",
    "Down the rabbit hole"
]

temperatures = [0.5, 1.0, 1.5]

for seed in test_seeds:
    print(f"\n{'='*100}")
    print(f"SEED: '{seed}'")
    print(f"{'='*100}")
    
    for temp in temperatures:
        print(f"\n--- Temperature: {temp} ---")
        generated = generate_text(model, seed, length=300, temperature=temp)
        print(f"\nComplete generated text with temperature {temp}:")
        print(f"'{generated[len(seed):][:200]}...'")  # Show first 200 chars of generated text
```

**Technical Explanation:**
- **Multiple seeds**: Tests model's ability to continue different starting contexts
- **Temperature comparison**: Demonstrates effect of randomness parameter
  - **0.5**: Conservative, predictable, grammatically correct
  - **1.0**: Balanced creativity and coherence
  - **1.5**: More creative, potentially less coherent
- **generated[len(seed):]**: Extracts only the newly generated portion
- **[:200]**: Limits display to first 200 characters for readability
- **Evaluation criteria**:
  - **Coherence**: Does text make grammatical sense?
  - **Creativity**: Are there novel word combinations?
  - **Context consistency**: Does it follow from the seed?

---

## Cell 11: Advanced Text Generation with Interactive Input

**What:** Create interactive interface for custom text generation with user-specified parameters
**Why:** Allows experimentation with different seeds and settings for practical use

```python
def interactive_generation():
    """
    Interactive text generation interface
    Allows users to input custom seed text and parameters
    """
    
    print("=== Interactive LSTM Text Generator ===")
    print("Enter your own seed text and parameters!")
    
    while True:
        try:
            # Get user input
            seed = input("\nEnter seed text (or 'quit' to exit): ")
            
            if seed.lower() == 'quit':
                break
            
            # Get generation parameters
            try:
                length = int(input("Enter length to generate (default 300): ") or "300")
                temperature = float(input("Enter temperature 0.5-2.0 (default 1.0): ") or "1.0")
            except ValueError:
                print("Using default values: length=300, temperature=1.0")
                length = 300
                temperature = 1.0
            
            # Generate text
            print(f"\nGenerating {length} characters...")
            generated_text = generate_text(model, seed, length, temperature)
            
            # Display result
            print(f"\nFull generated text:")
            print(f"'{generated_text}'")
            
            # Ask if user wants to save
            save_choice = input("\nSave this generation to file? (y/n): ")
            if save_choice.lower() == 'y':
                filename = f"generated_text_{len(generated_text)}chars.txt"
                with open(filename, 'w') as f:
                    f.write(generated_text)
                print(f"Saved to {filename}")
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

# Run interactive generator
# Uncomment the next line to run interactively
# interactive_generation()
```

**Technical Explanation:**
- **input()**: Captures user input from console
- **Default values**: or "300" provides fallback if user presses enter
- **try/except ValueError**: Handles invalid input gracefully
- **while True loop**: Continues until user types 'quit'
- **KeyboardInterrupt**: Handles Ctrl+C gracefully
- **File saving**: Writes generated text to .txt file for later use
- **Dynamic filename**: Includes character count in filename for organization

---

## Cell 12: Model Evaluation and Analysis

**What:** Comprehensive evaluation of model performance using multiple metrics
**Why:** Quantify model quality beyond just visual inspection of generated text

```python
def evaluate_model_performance(model, X_val, y_val):
    """
    Comprehensive model evaluation
    
    Metrics explained:
    - Loss: How well model predicts next character
    - Accuracy: Percentage of correct next-character predictions
    - Perplexity: Measure of how "surprised" the model is by test data
    """
    
    print("=== Model Performance Evaluation ===")
    
    # Evaluate on validation set
    val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
    
    # Calculate perplexity (lower is better)
    perplexity = np.exp(val_loss)
    
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
    print(f"Perplexity: {perplexity:.2f}")
    
    # Analyze prediction confidence
    predictions = model.predict(X_val[:100], verbose=0)
    avg_confidence = np.mean(np.max(predictions, axis=1))
    
    print(f"Average Prediction Confidence: {avg_confidence:.4f}")
    
    # Generate sample texts for diversity analysis
    print("\n=== Diversity Analysis ===")
    seed = "The quick brown fox"
    
    generations = []
    for i in range(5):
        gen_text = generate_text(model, seed, length=200, temperature=1.0)
        generations.append(gen_text[len(seed):])  # Remove seed part
    
    # Calculate unique character sequences
    unique_bigrams = set()
    unique_trigrams = set()
    
    for gen in generations:
        # Extract bigrams and trigrams
        for i in range(len(gen)-1):
            unique_bigrams.add(gen[i:i+2])
        for i in range(len(gen)-2):
            unique_trigrams.add(gen[i:i+3])
    
    print(f"Unique bigrams in 5 generations: {len(unique_bigrams)}")
    print(f"Unique trigrams in 5 generations: {len(unique_trigrams)}")
    
    return {
        'val_loss': val_loss,
        'val_accuracy': val_accuracy,
        'perplexity': perplexity,
        'avg_confidence': avg_confidence,
        'unique_bigrams': len(unique_bigrams),
        'unique_trigrams': len(unique_trigrams)
    }

# Run comprehensive evaluation
evaluation_results = evaluate_model_performance(model, X_val, y_val)
```

**Technical Explanation:**
- **model.evaluate()**: Computes loss and accuracy on validation set
- **Perplexity**: exp(loss) - measures how well model predicts the sequence
  - **Lower perplexity**: Better predictions, less "surprised" by data
  - **Higher perplexity**: More uncertain predictions
- **Prediction confidence**: Average of highest probability for each prediction
- **Diversity metrics**: Measure variety in generated text
  - **Bigrams**: Two-character sequences (like "th", "he", "at")
  - **Trigrams**: Three-character sequences (like "the", "and", "ing")
- **Higher diversity**: More unique n-grams indicate creative, varied output

---

## Cell 13: Model Visualization and Architecture Analysis

**What:** Visualize model weights and analyze learned patterns
**Why:** Understanding what the model learned helps improve future iterations

```python
def visualize_model_insights(model, char_to_idx, idx_to_char):
    """
    Analyze and visualize what the LSTM model has learned
    """
    
    print("=== Model Architecture Analysis ===")
    
    # Get model layers
    embedding_layer = model.get_layer('embedding')
    lstm1_layer = model.get_layer('lstm_1')
    lstm2_layer = model.get_layer('lstm_2')
    output_layer = model.get_layer('output')
    
    # Print layer information
    print(f"Embedding Layer: {embedding_layer.input_dim} → {embedding_layer.output_dim}")
    print(f"LSTM Layer 1: {lstm1_layer.units} units")
    print(f"LSTM Layer 2: {lstm2_layer.units} units")
    print(f"Output Layer: {output_layer.units} units")
    
    # Analyze embedding weights
    embedding_weights = embedding_layer.get_weights()[0]
    
    print(f"\nEmbedding matrix shape: {embedding_weights.shape}")
    
    # Find similar characters based on embedding similarity
    print("\n=== Character Similarity Analysis ===")
    
    def find_similar_chars(target_char, top_k=5):
        if target_char not in char_to_idx:
            return []
        
        target_idx = char_to_idx[target_char]
        target_embedding = embedding_weights[target_idx]
        
        # Calculate cosine similarity with all other characters
        similarities = []
        for char, idx in char_to_idx.items():
            if char != target_char:
                char_embedding = embedding_weights[idx]
                # Cosine similarity
                similarity = np.dot(target_embedding, char_embedding) / (
                    np.linalg.norm(target_embedding) * np.linalg.norm(char_embedding)
                )
                similarities.append((char, similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    # Analyze some interesting characters
    interesting_chars = ['a', 'e', 'i', 'o', 'u', ' ', '.', ',']
    
    for char in interesting_chars:
        if char in char_to_idx:
            similar = find_similar_chars(char)
            print(f"Characters similar to '{char}': {similar}")
    
    # Plot embedding visualization (2D projection using PCA)
    from sklearn.decomposition import PCA
    
    try:
        # Reduce embeddings to 2D for visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(embedding_weights)
        
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.6)
        
        # Annotate some characters
        for i, char in enumerate(list(char_to_idx.keys())[:20]):  # Show first 20 chars
            plt.annotate(repr(char), (embeddings_2d[i, 0], embeddings_2d[i, 1]))
        
        plt.title('Character Embeddings (2D PCA Projection)')
        plt.xlabel('First Principal Component')
        plt.ylabel('Second Principal Component')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('character_embeddings.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    except ImportError:
        print("sklearn not available - skipping embedding visualization")

# Run model analysis
visualize_model_insights(model, char_to_idx, idx_to_char)
```

**Technical Explanation:**
- **model.get_layer()**: Retrieves specific layers by name for analysis
- **get_weights()**: Returns trainable parameters of each layer
- **Embedding analysis**: Studies learned character representations
- **Cosine similarity**: Measures angle between embedding vectors
  - **np.dot()**: Dot product of vectors
  - **np.linalg.norm()**: Vector magnitude (Euclidean norm)
- **PCA (Principal Component Analysis)**: Reduces high-dimensional embeddings to 2D
- **Character clustering**: Similar characters should have similar embeddings
- **Visualization insights**: Vowels might cluster together, punctuation separate from letters

---

## Cell 14: Save and Load Model Functions

**What:** Create utility functions for saving and loading trained models
**Why:** Essential for model persistence and deployment in production

```python
def save_complete_model(model, char_to_idx, idx_to_char, vocab_size, sequence_length, filepath_base="lstm_model"):
    """
    Save complete model and all necessary components for later use
    """
    
    print("=== Saving Complete Model ===")
    
    # Save model architecture and weights
    model.save(f"{filepath_base}.h5")
    print(f"✓ Model saved to {filepath_base}.h5")
    
    # Save all mappings and parameters
    model_config = {
        'char_to_idx': char_to_idx,
        'idx_to_char': idx_to_char,
        'vocab_size': vocab_size,
        'sequence_length': sequence_length,
        'model_architecture': {
            'embedding_dim': model.get_layer('embedding').output_dim,
            'lstm_units': model.get_layer('lstm_1').units,
        }
    }
    
    with open(f"{filepath_base}_config.pkl", 'wb') as f:
        pickle.dump(model_config, f)
    print(f"✓ Configuration saved to {filepath_base}_config.pkl")
    
    # Save training history if available
    if 'history' in globals():
        with open(f"{filepath_base}_history.pkl", 'wb') as f:
            pickle.dump(history.history, f)
        print(f"✓ Training history saved to {filepath_base}_history.pkl")
    
    print("Model saving completed!")
    return f"{filepath_base}.h5", f"{filepath_base}_config.pkl"

def load_complete_model(model_filepath, config_filepath):
    """
    Load complete model and all necessary components
    
    Returns:
    - loaded_model: Keras model ready for generation
    - char_to_idx: Character to index mapping
    - idx_to_char: Index to character mapping
    - vocab_size: Vocabulary size
    - sequence_length: Sequence length for generation
    """
    
    print("=== Loading Complete Model ===")
    
    # Load model
    loaded_model = tf.keras.models.load_model(model_filepath)
    print(f"✓ Model loaded from {model_filepath}")
    
    # Load configuration
    with open(config_filepath, 'rb') as f:
        config = pickle.load(f)
    
    char_to_idx = config['char_to_idx']
    idx_to_char = config['idx_to_char']
    vocab_size = config['vocab_size']
    sequence_length = config['sequence_length']
    
    print(f"✓ Configuration loaded from {config_filepath}")
    print(f"  - Vocabulary size: {vocab_size}")
    print(f"  - Sequence length: {sequence_length}")
    
    return loaded_model, char_to_idx, idx_to_char, vocab_size, sequence_length

# Save the current model
model_files = save_complete_model(model, char_to_idx, idx_to_char, vocab_size, sequence_length)

# Demonstrate loading (you can use this in a new session)
print("\n" + "="*50)
print("DEMONSTRATION: Loading saved model")
print("="*50)

# Load the model we just saved
loaded_model, loaded_char_to_idx, loaded_idx_to_char, loaded_vocab_size, loaded_sequence_length = load_complete_model(
    model_files[0], model_files[1]
)

# Test that loaded model works
print("\nTesting loaded model with generation:")
test_generation = generate_text(loaded_model, "Alice was", length=100, temperature=1.0)
```

**Technical Explanation:**
- **model.save()**: Saves complete Keras model (architecture + weights)
- **pickle.dump()**: Serializes Python objects for persistent storage
- **Model configuration**: Stores all parameters needed for text generation
- **tf.keras.models.load_model()**: Loads complete Keras model
- **Persistence benefits**: Models can be used across different Python sessions
- **Production deployment**: Saved models can be deployed to web services or apps
- **Version control**: Different model versions can be saved and compared

---

## Cell 15: Performance Optimization and Tips

**What:** Advanced techniques for improving model performance and efficiency
**Why:** Real-world applications require optimized, efficient models

```python
def optimize_model_performance():
    """
    Advanced optimization techniques for LSTM text generation
    """
    
    print("=== Performance Optimization Techniques ===")
    
    print("\n1. MODEL ARCHITECTURE OPTIMIZATIONS:")
    print("   • Use GRU instead of LSTM for faster training (fewer parameters)")
    print("   • Implement attention mechanism for longer sequences")
    print("   • Use residual connections for deeper networks")
    print("   • Apply layer normalization for training stability")
    
    print("\n2. TRAINING OPTIMIZATIONS:")
    print("   • Gradient clipping to prevent exploding gradients")
    print("   • Learning rate scheduling (cosine annealing, warm restarts)")
    print("   • Mixed precision training for faster computation")
    print("   • Data loading optimization with tf.data API")
    
    print("\n3. INFERENCE OPTIMIZATIONS:")
    print("   • Model quantization to reduce size")
    print("   • TensorRT optimization for GPU inference")
    print("   • Batch prediction for multiple sequences")
    print("   • Caching for repeated generations")

def create_optimized_model(vocab_size, sequence_length, use_gru=False):
    """
    Create an optimized version of the text generation model
    """
    
    print("Building optimized model...")
    
    model = models.Sequential([
        # Embedding with mask_zero for variable length sequences
        layers.Embedding(
            vocab_size, 256, 
            input_length=sequence_length,
            mask_zero=True  # Handle padding tokens
        ),
        
        # Use GRU for faster training (optional)
        layers.GRU(512, return_sequences=True, dropout=0.3) if use_gru else
        layers.LSTM(512, return_sequences=True, dropout=0.3, recurrent_dropout=0.3),
        
        # Layer normalization for training stability
        layers.LayerNormalization(),
        
        layers.GRU(512, dropout=0.3) if use_gru else
        layers.LSTM(512, dropout=0.3, recurrent_dropout=0.3),
        
        layers.LayerNormalization(),
        
        # Dense layer with larger intermediate dimension
        layers.Dense(1024, activation='relu'),
        layers.Dropout(0.5),
        
        layers.Dense(vocab_size, activation='softmax')
    ])
    
    # Compile with gradient clipping
    optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def benchmark_generation_speed(model, seed_text, lengths=[100, 500, 1000]):
    """
    Benchmark text generation speed for different lengths
    """
    import time
    
    print("\n=== Generation Speed Benchmark ===")
    
    for length in lengths:
        start_time = time.time()
        
        # Generate text (suppress output)
        generated = generate_text(model, seed_text, length=length, temperature=1.0)
        
        end_time = time.time()
        generation_time = end_time - start_time
        chars_per_second = length / generation_time
        
        print(f"Length {length}: {generation_time:.2f}s ({chars_per_second:.1f} chars/sec)")

# Run optimization examples
optimize_model_performance()

# Benchmark current model
print("\nBenchmarking current model:")
benchmark_generation_speed(model, "Alice was beginning to", lengths=[100, 300, 500])
```

**Technical Explanation:**
- **GRU vs LSTM**: GRU has fewer parameters (faster training, less memory)
- **Layer Normalization**: Normalizes inputs to each layer for stable training
- **Gradient clipping**: clipnorm=1.0 prevents exploding gradients
- **mask_zero=True**: Handles variable-length sequences with padding
- **Mixed precision**: Uses float16 for faster computation (not shown, requires special setup)
- **Benchmarking**: Measures characters generated per second
- **Production considerations**: Model size, inference speed, memory usage

---

## Cell 16: Conclusion and Next Steps

**What:** Summary of what was accomplished and suggestions for further development
**Why:** Provides roadmap for extending and improving the LSTM text generator

```python
def print_tutorial_summary():
    """
    Comprehensive summary of the LSTM text generation tutorial
    """
    
    print("="*80)
    print("🎉 LSTM TEXT GENERATION TUTORIAL COMPLETED! 🎉")
    print("="*80)
    
    print("\n📚 WHAT YOU'VE LEARNED:")
    print("✓ Character-level text tokenization and preprocessing")
    print("✓ LSTM architecture design for sequence prediction")
    print("✓ Training deep learning models with callbacks")
    print("✓ Text generation with temperature-controlled sampling")
    print("✓ Model evaluation and performance analysis")
    print("✓ Model saving, loading, and deployment preparation")
    print("✓ Performance optimization techniques")
    
    print("\n🔧 WHAT YOU'VE BUILT:")
    print("• Complete LSTM text generation pipeline")
    print("• Interactive text generation interface")
    print("• Model evaluation and visualization tools")
    print("• Production-ready save/load functionality")
    
    print("\n🚀 NEXT STEPS FOR IMPROVEMENT:")
    print("\n1. ADVANCED ARCHITECTURES:")
    print("   • Transformer models (GPT-style)")
    print("   • Attention mechanisms")
    print("   • Bidirectional LSTMs")
    print("   • Variational autoencoders for text")
    
    print("\n2. BETTER DATASETS:")
    print("   • Larger corpora (Wikipedia, BookCorpus)")
    print("   • Domain-specific datasets")
    print("   • Multiple languages")
    print("   • Code generation datasets")
    
    print("\n3. ADVANCED TECHNIQUES:")
    print("   • Beam search for better generation")
    print("   • Nucleus (top-p) sampling")
    print("   • Fine-tuning on specific styles")
    print("   • Transfer learning from pre-trained models")
    
    print("\n4. DEPLOYMENT OPTIONS:")
    print("   • Flask/FastAPI web service")
    print("   • Streamlit interactive app")
    print("   • Mobile app with TensorFlow Lite")
    print("   • Cloud deployment (AWS, GCP, Azure)")
    
    print("\n5. EVALUATION IMPROVEMENTS:")
    print("   • BLEU scores for quality measurement")
    print("   • Human evaluation studies")
    print("   • Diversity metrics (distinct n-grams)")
    print("   • Coherence scoring")
    
    print("\n📖 RESOURCES FOR FURTHER LEARNING:")
    print("• 'Deep Learning' by Ian Goodfellow")
    print("• 'Natural Language Processing with Python' by Steven Bird")
    print("• Hugging Face Transformers library")
    print("• Papers: 'Attention Is All You Need', 'BERT', 'GPT' series")
    print("• Online courses: fast.ai, CS224N (Stanford NLP)")
    
    print("\n💡 PROJECT IDEAS:")
    print("• Poetry generator trained on specific poets")
    print("• Code completion system")
    print("• Chatbot with personality")
    print("• Story continuation tool")
    print("• Song lyrics generator")
    print("• Technical documentation assistant")
    
    print("\n" + "="*80)
    print("Thank you for completing this tutorial!")
    print("Share your generated text creations! 🎨")
    print("="*80)

# Print the comprehensive summary
print_tutorial_summary()

# Final demonstration with user's choice
print("\n🎯 FINAL DEMONSTRATION:")
print("Let's generate one final creative text sample!")

final_seeds = [
    "Once upon a time in a land far away",
    "The future of artificial intelligence",
    "In the depths of the ocean",
    "A mysterious letter arrived"
]

print("\nChoose a seed for final generation:")
for i, seed in enumerate(final_seeds, 1):
    print(f"{i}. {seed}")

print("\nGenerating with seed 1...")
final_generation = generate_text(model, final_seeds[0], length=400, temperature=1.2)

print(f"\n🌟 FINAL CREATIVE OUTPUT:")
print(f"Seed: '{final_seeds[0]}'")
print(f"Generated continuation:\n'{final_generation[len(final_seeds[0]):]}'\n")

print("🎊 Congratulations! You've built a complete LSTM text generator! 🎊")
```

**Technical Explanation:**
- **Tutorial recap**: Summarizes all major concepts covered
- **Next steps**: Provides clear progression path for learning
- **Resource recommendations**: Points to authoritative learning materials
- **Project ideas**: Suggests practical applications to build
- **Final demonstration**: Shows completed system in action
- **Deployment guidance**: Outlines path to production systems

---

## 📝 Tutorial Notes and Tips

### Key Concepts Mastered:
1. **Sequential Data Processing**: Understanding how LSTMs handle time-series data
2. **Character-Level Modeling**: Benefits and trade-offs vs word-level approaches
3. **Temperature Sampling**: Controlling creativity vs coherence in generation
4. **Model Architecture**: Embedding → LSTM → Dense pipeline design
5. **Training Strategies**: Callbacks, early stopping, learning rate scheduling

### Common Issues and Solutions:
- **Memory errors**: Reduce batch_size or sequence_length
- **Slow training**: Use GPU, reduce model size, or implement gradient accumulation
- **Poor generation quality**: Train longer, use larger dataset, tune temperature
- **Overfitting**: Add more dropout, use early stopping, regularization
- **Vanishing gradients**: Use gradient clipping, layer normalization

### Performance Benchmarks:
- **Training time**: 15-30 minutes on GPU for 50 epochs
- **Generation speed**: ~50-200 characters/second depending on hardware
- **Model size**: ~10-50MB depending on architecture
- **Memory usage**: ~2-8GB during training

This completes the comprehensive LSTM text generation tutorial! The code is production-ready and includes all necessary components for building, training, evaluating, and deploying a character-level text generation system.
