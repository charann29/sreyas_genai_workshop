# ============================================================================
# EXPERIMENT 6: LSTM NETWORK FOR TEXT GENERATION - GOOGLE COLAB VERSION
# ============================================================================

# CELL 1: Install and Import Dependencies
# Why: Set up the environment with all necessary libraries
# What: Import TensorFlow, NumPy, Matplotlib, and other essential libraries
# Execute this cell first to ensure all dependencies are available

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

print("TensorFlow version:", tf..__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ============================================================================

# CELL 2: Download and Prepare Text Dataset
# Why: We need a substantial text corpus for training the LSTM
# What: Download a classic text (Alice in Wonderland) and preprocess it
# This gives us a rich, varied dataset for learning language patterns

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

# ============================================================================

# CELL 3: Create Character-Level Tokenization
# Why: Convert text to numerical format that the neural network can process
# What: Create mappings between characters and integers, build vocabulary
# Character-level allows the model to learn spelling and word formation

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

# ============================================================================

# CELL 4: Create Training Sequences
# Why: LSTM needs sequences of fixed length for training
# What: Create input-output pairs where input is sequence of chars, output is next char
# This teaches the model to predict the next character given previous context

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

# ============================================================================

# CELL 5: Build LSTM Architecture
# Why: LSTM can learn long-term dependencies in text sequences
# What: Create a model with embedding, LSTM layers, and output layer
# Architecture choices explained below

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

# ============================================================================

# CELL 6: Set Up Training Callbacks
# Why: Monitor training progress and prevent overfitting
# What: Configure callbacks for model checkpointing, early stopping, and learning rate reduction

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

# ============================================================================

# CELL 7: Train the LSTM Model
# Why: Train the model to learn text generation patterns
# What: Fit the model on training data with validation monitoring
# This cell will take the longest time to execute (15-30 minutes depending on hardware)

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

# ============================================================================

# CELL 8: Visualize Training Progress
# Why: Understand how well the model learned and if there's overfitting
# What: Plot training/validation loss and accuracy curves

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

# ============================================================================

# CELL 9: Implement Text Generation Function
# Why: Use the trained model to generate new text
# What: Create a function that takes a seed text and generates continuations
# Temperature controls creativity vs coherence

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

# ============================================================================

# CELL 10: Generate Sample Texts
# Why: Demonstrate the model's capabilities with different creativity levels
# What: Generate text with various temperatures and seed texts

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

# ============================================================================

# CELL 11: Advanced Text Generation with Interactive Input
# Why: Allow for interactive experimentation with the model
# What: Create an interactive interface for custom text generation

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

# ============================================================================

# CELL 12: Model Evaluation and Analysis
# Why: Assess model performance and understand its capabilities
# What: Calculate perplexity, analyze character-level accuracy, and evaluate diversity

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

# Evaluate the model
performance_metrics = evaluate_model_performance(model, X_val, y_val)

# Save performance metrics
with open('model_performance.pkl', 'wb') as f:
    pickle.dump(performance_metrics, f)

print("\nModel evaluation completed!")
print("All files saved:")
print("- lstm_text_generator.h5 (trained model)")
print("- char_mappings.pkl (character mappings)")
print("- model_performance.pkl (performance metrics)")
print("- lstm_training_history.png (training plots)")

# ============================================================================

# CELL 13: Final Summary and Next Steps
# Why: Summarize what was accomplished and suggest improvements
# What: Display final results and recommendations for further experimentation

print("="*80)
print("EXPERIMENT 6: LSTM TEXT GENERATION - SUMMARY")
print("="*80)

print(f"""
WHAT WE ACCOMPLISHED:
✓ Built character-level LSTM text generator
✓ Trained on {len(raw_text)} characters of text data
✓ Achieved {performance_metrics['val_accuracy']*100:.1f}% character prediction accuracy
✓ Model perplexity: {performance_metrics['perplexity']:.1f}
✓ Generated coherent text samples with controllable creativity

ARCHITECTURE USED:
- Embedding layer: {256} dimensions
- LSTM layers: 2 layers with {512} units each
- Vocabulary size: {vocab_size} characters
- Sequence length: {sequence_length} characters
- Total parameters: {model.count_params():,}

SUGGESTIONS FOR IMPROVEMENT:
1. Use larger dataset (e.g., complete works of Shakespeare, Wikipedia)
2. Experiment with word-level tokenization for better semantic understanding
3. Try different architectures (GRU, Bidirectional LSTM)
4. Implement beam search for better text generation
5. Add attention mechanism for longer context understanding
6. Fine-tune on domain-specific text for specialized generation

HOW TO USE YOUR TRAINED MODEL:
1. Load model: model = tf.keras.models.load_model('lstm_text_generator.h5')
2. Load mappings: with open('char_mappings.pkl', 'rb') as f: mappings = pickle.load(f)
3. Generate text using the generate_text() function

EXPERIMENT COMPLETED SUCCESSFULLY! 🎉
""")
