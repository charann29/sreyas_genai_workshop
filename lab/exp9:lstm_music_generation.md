### Implement a Long Short-Term Memory (LSTM) network using TensorFlow 2 for music generation

This experiment focuses on building a generative model that can learn patterns from a sequence of music notes and then generate new, original musical compositions. We'll use a Long Short-Term Memory (LSTM) network, which is a type of recurrent neural network (RNN) well-suited for processing sequential data like music.

We'll use a simplified approach where music is represented as a sequence of MIDI note numbers.

#### What and Why for each cell:

#### Cell 1: Setup and Data Loading

**What:**

```python
# Install and import necessary libraries
!pip install tensorflow music21

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation, Dropout
import numpy as np
from music21 import converter, instrument, note, stream, chord

# We'll need a dataset of MIDI files. For this example, we assume you have a folder
# named 'midi_files' containing your MIDI data.
# This part is a placeholder. You need to download a MIDI dataset.
# E.g., 'A classical music MIDI dataset' from Kaggle or similar sources.

# Load and parse MIDI files to extract notes and chords
notes = []
for file in glob.glob("midi_files/*.mid"):
    try:
        midi = converter.parse(file)
        print(f"Parsing {file}")
        
        parts = instrument.partitionByInstrument(midi)
        if parts: # file has instrument parts
            notes_to_parse = parts.parts[0].recurse()
        else: # file has no instrument parts
            notes_to_parse = midi.flat.notes
            
        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
                
    except Exception as e:
        print(f"Could not parse {file}: {e}")

# Create a set of unique notes and chords
unique_notes = sorted(list(set(notes)))
print(f"Found {len(unique_notes)} unique notes and chords.")
```

**Why:**
This cell is for setting up our environment and preparing the raw music data.

  - We install `tensorflow` for building the neural network and `music21` for parsing MIDI files.
  - `music21` is a powerful toolkit for manipulating musical data. We use it to read MIDI files, extract individual notes and chords, and represent them as strings. Chords are represented as a string of their note numbers separated by a dot.
  - The `glob` module is used to find all `.mid` files in a specified directory.
  - We iterate through each file, parse it, and extract a sequence of notes and chords, storing them in the `notes` list.
  - We then create `unique_notes` to build a vocabulary of all possible notes and chords in our dataset. This is essential for converting the notes into a numerical format that the model can process.

#### Cell 2: Data Preprocessing and Sequence Creation

**What:**

```python
# Create a dictionary to map notes to integers and vice versa
note_to_int = {note: i for i, note in enumerate(unique_notes)}
int_to_note = {i: note for i, note in enumerate(unique_notes)}

# Define the sequence length for the LSTM input
sequence_length = 100

# Create input and output sequences for the model
network_input = []
network_output = []

for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[note] for note in sequence_in])
    network_output.append(note_to_int[sequence_out])

# Reshape the input into a format compatible with LSTMs
n_patterns = len(network_input)
network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))

# Normalize the input data
n_vocab = len(unique_notes)
network_input = network_input / float(n_vocab)

# One-hot encode the output data
network_output = tf.keras.utils.to_categorical(network_output, num_classes=n_vocab)
```

**Why:**
This cell is where we prepare the data for the neural network.

  - **Mapping:** We create two dictionaries, `note_to_int` and `int_to_note`, to convert between our string-based notes/chords and numerical indices. This is a crucial step as neural networks operate on numbers, not text.
  - **Sequencing:** We define a `sequence_length` (e.g., 100 notes). The core idea is to train the model to predict the next note given the previous `sequence_length` notes. We slide a window of this size over our entire `notes` list to create our training samples. `network_input` contains the sequences, and `network_output` contains the note that follows each sequence.
  - **Reshaping:** LSTM layers in Keras expect a 3D input tensor of shape `(number of samples, time steps, features)`. We reshape our `network_input` to match this format. `time steps` is our `sequence_length`, and `features` is `1` since each step is a single integer representing a note.
  - **Normalization:** We normalize the input data by dividing by the total number of unique notes (`n_vocab`). This helps the model train more effectively by keeping the input values in a smaller, more stable range.
  - **One-hot Encoding:** The output of our model will be a probability distribution over all possible notes. We convert our integer-based `network_output` into a one-hot encoded vector, which is a standard format for classification tasks where the model predicts one out of `n_vocab` classes.

#### Cell 3: Building the LSTM Model

**What:**

```python
# Build the LSTM model
model = Sequential()
model.add(LSTM(
    512,
    input_shape=(network_input.shape[1], network_input.shape[2]),
    return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Dropout(0.3))
model.add(Dense(n_vocab))
model.add(Activation('softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam')

# Display the model summary
model.summary()
```

**Why:**
This is the heart of the experiment, where we define the neural network architecture.

  - **Sequential Model:** We use a `Sequential` model, which is a linear stack of layers.
  - **LSTM Layers:** We add several LSTM layers. These layers are designed to remember information over long sequences. `return_sequences=True` means the layer will output a sequence of its hidden states, which is necessary when stacking multiple LSTM layers. The final LSTM layer has `return_sequences=False` because we only need the final hidden state to make our prediction.
  - **Dropout:** Dropout layers are added to prevent overfitting. They randomly "drop out" a fraction of the neurons during training, forcing the network to learn more robust features.
  - **Dense Layers:** We add `Dense` (fully connected) layers for further processing. The first `Dense` layer has 256 neurons, followed by a dropout layer.
  - **Output Layer:** The final `Dense` layer has `n_vocab` neurons, one for each unique note/chord. The `softmax` activation function converts the output into a probability distribution, where each value represents the probability of the corresponding note/chord being the next in the sequence.
  - **Compilation:** We compile the model, specifying the `loss` function (`categorical_crossentropy` for multi-class classification) and the `optimizer` (`adam` is a popular and effective choice).

#### Cell 4: Training the Model

**What:**

```python
# Train the model
model.fit(network_input, network_output, epochs=50, batch_size=64)

# Save the trained model weights
model.save_weights('music_generator_weights.h5')
```

**Why:**
This is the training loop.

  - **`model.fit`:** This is where the model learns from the data.
      - `network_input` and `network_output` are the sequences we prepared earlier.
      - `epochs=50` means the model will iterate over the entire dataset 50 times.
      - `batch_size=64` means the data is divided into chunks of 64 samples, and the model's weights are updated after each batch.
  - **Saving Weights:** Once training is complete, we save the model's learned weights to a file (`music_generator_weights.h5`). This allows us to load the trained model later for music generation without having to retrain it.

#### Cell 5: Music Generation

**What:**

```python
# This cell is for generating music using the trained model

# Load the trained weights
# model.load_weights('music_generator_weights.h5') # Uncomment if you're loading a saved model

# Pick a random starting sequence from the training data
start = np.random.randint(0, len(network_input) - 1)
pattern = network_input[start]
generated_notes = [int_to_note[i] for i in pattern]

# Generate new notes based on the pattern
for _ in range(500): # Generate 500 new notes
    # Reshape the input pattern for the model
    input_pattern = np.reshape(pattern, (1, len(pattern), 1))
    
    # Predict the next note
    prediction = model.predict(input_pattern, verbose=0)
    
    # Get the predicted note index from the probability distribution
    index = np.argmax(prediction)
    result = int_to_note[index]
    
    # Append the new note and update the pattern
    generated_notes.append(result)
    
    # Shift the window to include the new note
    pattern = np.flatten(pattern)
    pattern = np.append(pattern, index)
    pattern = pattern[1:len(pattern)]
    pattern = np.reshape(pattern, (len(pattern), 1))
    
# Convert the generated notes back into a MIDI file
offset = 0
output_stream = stream.Stream()

for element in generated_notes:
    if ('.' in element) or element.isdigit():
        notes_in_chord = element.split('.')
        chord_notes = [note.Note(int(c)) for c in notes_in_chord]
        new_chord = chord.Chord(chord_notes)
        new_chord.offset = offset
        output_stream.append(new_chord)
    else:
        new_note = note.Note(element)
        new_note.offset = offset
        output_stream.append(new_note)
    offset += 0.5 # A simple fixed offset for note duration

# Save the generated MIDI file
output_stream.write('midi', fp='generated_music.mid')
print("Generated a new MIDI file: generated_music.mid")
```

**Why:**
This cell uses the trained model to create new music.

  - **Loading Weights:** We load the saved model weights to ensure we are using the trained model.
  - **Starting Sequence:** We need to provide the model with a "seed" sequence to start the generation process. We randomly pick a sequence from our training data.
  - **Generation Loop:** We loop a specified number of times (e.g., 500) to generate new notes.
      - Inside the loop, we feed the current `pattern` into the model.
      - `model.predict` returns the probability distribution for the next note.
      - We use `np.argmax` to select the note with the highest probability.
      - We then append this new note to our `generated_notes` list and update the `pattern` by removing the first note and adding the newly generated note. This is a sliding window approach that allows the model to continuously generate new music based on what it just generated.
  - **MIDI Conversion:** Finally, we take our list of generated note strings and convert them back into a musical stream using `music21`. We iterate through the list, create `note.Note` or `chord.Chord` objects, and append them to an `output_stream`. A simple fixed `offset` is used to give a basic rhythm.
  - **Saving:** We save the final stream to a new MIDI file (`generated_music.mid`), which can be opened and played in any music software.
