To **shorten** and **keep only what's necessary** from your full GPT-style transformer implementation (Experiment 7), here’s a **cleaned-down version** containing only essential parts.

---

### ✅ **Minimal Working Code for Transformer-based Language Model (GPT-style) in TensorFlow 2**

---

### **1. Install & Import Dependencies**

```python
!pip install tensorflow numpy

import tensorflow as tf
import numpy as np
import re
```

---

### **2. Download & Preprocess Text**

```python
import urllib.request

url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
urllib.request.urlretrieve(url, "shakespeare.txt")

with open("shakespeare.txt", 'r', encoding='utf-8') as f:
    text = re.sub(r'\s+', ' ', f.read().lower())

vocab = sorted(set(text))
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

text_as_int = np.array([char2idx[c] for c in text])
vocab_size = len(vocab)
```

---

### **3. Create Dataset**

```python
seq_length = 100
examples_per_epoch = len(text) // seq_length

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)

def split_input_target(chunk):
    return chunk[:-1], chunk[1:]

dataset = sequences.map(split_input_target)
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
```

---

### **4. Build Simple GPT-style Model**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout

class SimpleTransformer(tf.keras.Model):
    def __init__(self, vocab_size, num_heads=2, d_model=128):
        super().__init__()
        self.embedding = Embedding(vocab_size, d_model)
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.norm = LayerNormalization()
        self.dense = Dense(vocab_size)

    def call(self, x):
        x = self.embedding(x)
        attn_out = self.att(x, x, x)
        x = self.norm(x + attn_out)
        return self.dense(x)

model = SimpleTransformer(vocab_size)
```

---

### **5. Compile & Train**

```python
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer="adam", loss=loss_fn)

model.fit(dataset, epochs=3)
```

---

### **6. Generate Text**

```python
def generate_text(model, start_string, num_generate=300):
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []

    for _ in range(num_generate):
        predictions = model(input_eval)
        predictions = predictions[:, -1, :]
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        input_eval = tf.concat([input_eval, [[predicted_id]]], axis=1)
        text_generated.append(idx2char[predicted_id])

    return start_string + ''.join(text_generated)

print(generate_text(model, start_string="To be, or not to be"))
```

---

