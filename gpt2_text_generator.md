# GPT-2 Text Generator with Line-by-Line Explanation

## 📚 Imports

```python
import torch
```
**torch** is the main package of PyTorch, a deep learning framework. Used here to check if GPU is available (`torch.cuda.is_available()`).

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
```
**transformers** is a library from Hugging Face for state-of-the-art NLP models:
- **GPT2LMHeadModel**: GPT-2 model for language modeling (LM)
- **GPT2Tokenizer**: Breaks down text into tokens the model can understand
- **pipeline**: A high-level abstraction to simplify common tasks like text generation

## 🏗️ TextGenerator Class Definition

```python
class TextGenerator:
```
Defines a custom class `TextGenerator` to encapsulate GPT-2 model logic.

### 🔸 `__init__` Method – Class Constructor

```python
    def __init__(self, model_name='gpt2'):
```
The constructor initializes the model and tokenizer. `model_name='gpt2'` sets the default model to GPT-2, but you can pass a different Hugging Face model name.

```python
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
```
Downloads and loads the tokenizer for the specified model. The tokenizer converts text to input IDs (numbers) and vice versa.

```python
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
```
Loads the pretrained GPT-2 model with a language modeling head (used for text generation).

```python
        self.tokenizer.pad_token = self.tokenizer.eos_token
```
GPT-2 does not have a `pad_token` by default, so this line avoids a warning during generation by reusing the `eos_token` (end of sentence).

### 🔸 Create a Text Generation Pipeline

```python
        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )
```
- `pipeline('text-generation')`: sets up an easy-to-use text generator
- `device=0`: use GPU if available
- `device=-1`: fallback to CPU
- Combines the model and tokenizer into a single callable object

## ✅ `generate_story` Method

```python
    def generate_story(self, prompt, max_new_tokens=256, temperature=0.9):
```
This function takes a prompt (starting sentence) and generates a longer story:
- `max_new_tokens`: how many new tokens to generate
- `temperature`: controls randomness in predictions (higher = more creative)

```python
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id
        )
```
Parameters explained:
- `prompt`: initial input text
- `do_sample=True`: use sampling instead of greedy decoding (for more creative text)
- `top_p=0.95`: nucleus sampling – pick from the top tokens with 95% probability mass
- `repetition_penalty=1.3`: discourages repeating the same words/phrases
- `pad_token_id`: needed because GPT-2 has no built-in pad_token

```python
        return result[0]['generated_text']
```
`result` is a list of dictionaries. Each has a `'generated_text'` key. Returns the full generated string.

## ✅ `complete_sentence` Method

```python
    def complete_sentence(self, partial_text):
```
This function attempts to complete a short phrase or sentence.

```python
        return self.generate_story(partial_text, max_new_tokens=30)
```
Calls `generate_story()` with a smaller generation length. Useful for autocompleting user input or generating suggestions.

## 🚀 Usage Example Block

```python
if __name__ == "__main__":
```
Ensures that the code only runs when the file is executed directly, not when imported as a module.

```python
    text_gen = TextGenerator()
```
Creates an instance of the `TextGenerator` class, loading the model and tokenizer.

```python
    prompt = "Once upon a time in a magical forest"
```
A sample starting line for the story.

```python
    story = text_gen.generate_story(prompt)
```
Generates a story using the above prompt.

```python
    print("\nGenerated Story:\n")
    print(story)
```
Prints the story to the console.

```python
    # ✅ Save to .txt file
    with open("generated_story.txt", "w", encoding="utf-8") as f:
        f.write(story)
```
Opens a file called "generated_story.txt" in write mode with UTF-8 encoding and saves the generated story to it.

## 🧠 Summary of Key Concepts

| Concept | Meaning |
|---------|---------|
| **Tokenizer** | Converts text ↔ tokens (integers) |
| **Model** | GPT-2 for language generation |
| **Pipeline** | Simplifies model usage |
| **Sampling** | Introduces creativity/randomness |
| **Temperature** | Controls randomness (0 = deterministic) |
| **top_p** | Keeps only most probable words summing to 95% |
| **repetition_penalty** | Discourages loops/repetition |
| **max_new_tokens** | Number of tokens to generate (excluding input) |

## 📋 Complete Code

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline

class TextGenerator:
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

        # Set pad_token to eos_token to avoid warnings
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.generator = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1
        )

    def generate_story(self, prompt, max_new_tokens=256, temperature=0.9):
        result = self.generator(
            prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            repetition_penalty=1.3,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return result[0]['generated_text']

    def complete_sentence(self, partial_text):
        return self.generate_story(partial_text, max_new_tokens=30)

# --- ✅ Usage Example ---
if __name__ == "__main__":
    text_gen = TextGenerator()

    prompt = "Once upon a time in a magical forest"
    story = text_gen.generate_story(prompt)

    print("\nGenerated Story:\n")
    print(story)

    # ✅ Save to .txt file
    with open("generated_story.txt", "w", encoding="utf-8") as f:
        f.write(story)
```
