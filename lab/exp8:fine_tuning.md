### Fine-Tuning a Pre-trained Language Model (e.g., GPT, BERT)

This experiment focuses on leveraging the power of pre-trained language models to perform a specific task by fine-tuning them on a smaller, domain-specific dataset. This is a common and highly effective technique in natural language processing (NLP). We'll use the Hugging Face `transformers` library, a popular choice for working with these models.

#### What and Why for each cell:

#### Cell 1: Setup and Library Installation

**What:**

```python
# Install the necessary libraries
!pip install transformers datasets accelerate torch

# Import required libraries
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
```

**Why:**
This cell ensures all the required Python packages are installed. We need `transformers` for accessing the pre-trained model and tokenizer, `datasets` for easily loading and handling our data, `accelerate` for efficient training, and `torch` as our deep learning framework. We then import the specific classes we'll use: `BertTokenizer` to convert text into a format the model understands, `BertForSequenceClassification` which is a BERT model configured for classification tasks, `Trainer` and `TrainingArguments` to simplify the fine-tuning process, and `load_dataset` to get our data.

#### Cell 2: Load Dataset and Preprocess Data

**What:**

```python
# Load a domain-specific dataset. For this example, we'll use a sentiment analysis dataset.
# Replace this with your own dataset if you have one.
dataset = load_dataset("imdb")

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define a function to tokenize the text
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

# Apply the tokenizer to the dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Rename columns and set format for PyTorch
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "token_type_ids", "labels"])
```

**Why:**
This is the data preparation step. We first load a dataset. The `imdb` dataset is used here as a simple example for a sentiment analysis task (classifying movie reviews as positive or negative), but you would replace this with your own domain-specific data. We then load the `bert-base-uncased` tokenizer, which is the pre-trained tokenizer that corresponds to our model. The `tokenize_function` takes our text and converts it into numerical input IDs, attention masks, and token type IDs, which are the inputs our BERT model expects. The `map` function applies this tokenization to the entire dataset efficiently. Finally, we rename the `label` column to `labels` (a requirement for the `Trainer` API) and set the format to `torch` to make it compatible with our PyTorch model.

#### Cell 3: Load the Pre-trained Model

**What:**

```python
# Load the pre-trained BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
```

**Why:**
Here we load the actual pre-trained model. `BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)` does a few things:

1.  It downloads the pre-trained BERT model weights from the Hugging Face model hub.
2.  It adds a classification layer on top of the BERT model.
3.  The `num_labels=2` argument configures this classification layer for our specific task (e.g., positive vs. negative sentiment).

The model's weights are initialized with the pre-trained BERT values, which have been learned on a massive text corpus. This provides a strong starting point, and we only need to "fine-tune" the model on our much smaller dataset to adapt it to our specific task.

#### Cell 4: Fine-Tuning the Model

**What:**

```python
# Define the training arguments
training_args = TrainingArguments(
    output_dir='./results',          # directory to store outputs
    num_train_epochs=3,              # total number of training epochs
    per_device_train_batch_size=8,   # batch size for training
    per_device_eval_batch_size=8,    # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=10,
    evaluation_strategy="epoch",     # evaluation is done at the end of each epoch
    save_strategy="epoch",           # model is saved at the end of each epoch
)

# Initialize the Trainer
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments
    train_dataset=tokenized_datasets['train'], # training dataset
    eval_dataset=tokenized_datasets['test'],   # evaluation dataset
)

# Start training
trainer.train()
```

**Why:**
This is the core fine-tuning step. The `TrainingArguments` class is a convenient way to specify all the hyperparameters for our training loop, such as the number of epochs, batch size, learning rate scheduler, and where to save outputs. The `Trainer` class then brings everything together: the model we want to train, the training arguments, and our training and evaluation datasets. Calling `trainer.train()` starts the fine-tuning process. The `Trainer` handles all the complexity of the training loop for us, including moving data to the GPU (if available), calculating loss, and performing backpropagation and optimization. It will use our pre-trained model as a starting point and update its weights to perform well on our specific classification task.

#### Cell 5: Evaluate the Fine-tuned Model (Optional but Recommended)

**What:**

```python
# Evaluate the fine-tuned model on the test set
trainer.evaluate()

# You can also make predictions on new data
# For example, let's predict the sentiment of a new review
new_text = "This movie was absolutely fantastic and I loved every minute of it!"
inputs = tokenizer(new_text, return_tensors='pt', padding=True, truncation=True)
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)

# Map prediction to label
label_map = {0: "negative", 1: "positive"}
predicted_label = label_map[predictions.item()]
print(f"The predicted sentiment is: {predicted_label}")
```

**Why:**
After fine-tuning, it's crucial to evaluate the model's performance on a held-out test set to ensure it generalizes well to unseen data. `trainer.evaluate()` calculates metrics like accuracy and loss on the evaluation set. We also demonstrate how to use the fine-tuned model to make predictions on new, single-text inputs. We tokenize the new text, pass it through our model, and interpret the output to get a final prediction. This shows how to use the model in a real-world scenario.
