
# Advanced MongoDB RAG System with Groq

This project implements a basic Retrieval-Augmented Generation (RAG) system using **MongoDB** as the vector database, the **Groq API** for fast language model inference, and **Sentence-Transformers** for generating text embeddings. It's a complete, command-line-driven application that allows you to add documents, search for them semantically, and ask questions that get answered based on the stored information.

A key feature is its robustness: it automatically detects and processes documents that might have been added to MongoDB manually (without embeddings), ensuring your database is always ready for querying.

## 🌟 Key Features

  * **Vector Storage**: Uses a standard MongoDB collection to store text, metadata, and vector embeddings.
  * **Fast LLM Inference**: Leverages the high-speed Groq API for the generation step.
  * **Semantic Search**: Implements cosine similarity to find the most relevant documents for a given query.
  * **Robust Data Handling**: Automatically creates embeddings for documents added manually to the database.
  * **Batch Processing**: Efficiently adds multiple documents at once.
  * **Interactive CLI**: Provides a user-friendly command-line interface to interact with the system.
  * **Easy Setup**: Requires only a few environment variables to get started.

-----

## ⚙️ How It Works

The RAG process follows these steps:

1.  **Ingestion**: You add documents to the MongoDB collection. The system automatically generates a vector embedding for each document's content using a `SentenceTransformer` model and stores it alongside the text and metadata.
2.  **Retrieval**: When you ask a question (a query), the system generates an embedding for your query. It then calculates the cosine similarity between the query embedding and the embeddings of all stored documents to find the most relevant ones (the "context").
3.  **Generation**: The system sends the original question and the retrieved context to the Groq API. The Language Model (e.g., Llama 3) then generates a concise, accurate answer based *only* on the provided information.

This approach allows the LLM to answer questions about specific, private, or up-to-date information that it wasn't trained on.

## 🛠️ Setup and Installation

Follow these steps to set up your environment and run the project.

### Step 1: Clone the Repository

First, get the code on your local machine.

```bash
git clone <your-repository-url>
cd <repository-directory>
```

### Step 2: Create a Virtual Environment

It's best practice to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Required Libraries

Install all necessary Python packages using pip.

```bash
pip install pymongo sentence-transformers groq python-dotenv numpy
```

### Step 4: Install and Run MongoDB

You need a running MongoDB instance on your local machine.

1.  **Download MongoDB Community Server** from the [official MongoDB website](https://www.mongodb.com/try/download/community). Select the correct version for your operating system (Windows, macOS, or Linux) and download the installer.
2.  **Install MongoDB**. Follow the installation instructions for your OS:
      * **Windows**: Run the `.msi` installer and follow the wizard. It is recommended to install MongoDB as a service so it runs automatically in the background.
      * **macOS**: The easiest way is using Homebrew.
        ```bash
        brew tap mongodb/brew
        brew install mongodb-community
        ```
      * **Linux (Ubuntu)**:
        ```bash
        sudo apt-get install gnupg
        wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -
        echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
        sudo apt-get update
        sudo apt-get install -y mongodb-org
        ```
3.  **Start the MongoDB Service**. Ensure the MongoDB service is running. If you installed it as a service, it should start automatically. If not, you may need to start it manually.
      * On macOS with Homebrew: `brew services start mongodb/brew/mongodb-community`
      * On Linux: `sudo systemctl start mongod`

The script will connect to the default MongoDB URI `mongodb://localhost:27017`, which works out-of-the-box with a standard local installation.

### Step 5: Get a Groq API Key

1.  Go to the [Groq Console](https://console.groq.com/keys).
2.  Sign up or log in.
3.  Create a new API key. Copy it securely.

### Step 6: Create the `.env` File

In the root directory of the project, create a file named `.env` and add your Groq API key to it.

```env
# .env file
GROQ_API_KEY="gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

-----

## 🚀 Running and Testing the Application

With everything set up, you are ready to run the interactive CLI and test its features.

### Command-by-Command Walkthrough

Follow this sequence in your terminal to see the system in action.

**1. Start the Application**

Run the Python script. Make sure your virtual environment is active and MongoDB is running.

```bash
python your_script_name.py
```

You should see the startup message:

```
🚀 Initializing Groq MongoDB RAG System...
✅ System initialized successfully!
🤖 Groq MongoDB RAG System
Commands: ask, search, stats, add, load_sample, process_manual, help, quit
--------------------------------------------------

GroqRAG>
```

**2. Load Sample Data**

Use the `load_sample` command to populate the database with a predefined set of documents.

```
GroqRAG> load_sample
```

Output:

```
📥 Loading sample data...
✅ Loaded 8 sample documents.
```

**3. Check Database Statistics**

Use the `stats` command to verify that the documents were added and see their category distribution.

```
GroqRAG> stats
```

Output:

```
📊 System Statistics:
    Total Documents: 8

📁 Categories:
    - technology: 2 documents
    - health: 2 documents
    - science: 2 documents
    - business: 2 documents
```

**4. Test Manual Document Processing**

This test simulates a scenario where a document is added to MongoDB from an external source (i.e., without an embedding).

  * **First, add data manually**: You can use a tool like MongoDB Compass or the MongoDB Shell (`mongosh`) to insert the following document directly into the `groq_rag` database and `documents` collection. This document intentionally lacks the `embedding` field.

    ```json
    {
      "doc_id": "history_001",
      "content": "The Roman Empire was one of the most influential civilizations in world history, known for its contributions to law, architecture, and engineering.",
      "metadata": { "category": "history", "source": "manual_import" },
      "created_at": new Date()
    }
    ```

  * **Now, run the `process_manual` command**: This tells the system to scan the database for any documents that are missing an embedding and create it.

    ```
    GroqRAG> process_manual
    ``

    Output:

    ```
    🔄 Processing manually added documents (generating missing embeddings)...
    INFO:__main__:Checking for documents without embeddings...
    INFO:__main__:Found 1 documents without embeddings. Processing them...
    INFO:__main__:Successfully added embeddings to 1 documents.
    Processing complete. Check logs for details.
    ```

**5. Verify the Manual Addition**

Check the stats again. You should see the total document count has increased.

```
GroqRAG> stats
```

Output:

```
📊 System Statistics:
    Total Documents: 9

📁 Categories:
    - ...
    - history: 1 documents
```

**6. Ask a Question**

Finally, test the end-to-end RAG pipeline by asking a question. The system will retrieve the most relevant documents and use them to generate an answer.

```
GroqRAG> ask what was the roman empire known for?
```

Output:

```
🔍 Processing: what was the roman empire known for?

💭 Answer: The Roman Empire was known for its significant contributions to law, architecture, and engineering.
📊 Retrieved 1 documents. Model: llama3-8b-8192.

📚 Sources:
  1. ID: history_001 (Similarity: 0.812)
```

**7. Quit the Application**

Use the `quit` command to exit the CLI.

```
GroqRAG> quit
```

Output:

```
Goodbye! 👋
```


## 🚀 Running the Application

With everything set up, you can now run the interactive command-line interface (CLI).

```bash
python your_script_name.py
```

You should see the following output, indicating the system is ready:

```
🚀 Initializing Groq MongoDB RAG System...
INFO:__main__:Checking for documents without embeddings...
INFO:__main__:All documents appear to have embeddings.
INFO:__main__:Groq MongoDB RAG initialized successfully
✅ System initialized successfully!
🤖 Groq MongoDB RAG System
Commands: ask, search, stats, add, load_sample, process_manual, help, quit
--------------------------------------------------

GroqRAG>
```

### CLI Commands

Here are the commands you can use in the CLI:

| Command                               | Description                                                                 |
| ------------------------------------- | --------------------------------------------------------------------------- |
| `ask <question>`                      | Asks a question. The system will retrieve relevant docs and generate an answer. |
| `search <query>`                      | Performs a semantic search and returns the top matching documents.             |
| `add <id>|<content>|<category>`       | Adds a new document to the database.                                        |
| `load_sample`                         | Ingests a predefined set of sample documents into the database.             |
| `stats`                               | Displays statistics about the documents in the database.                    |
| `process_manual`                      | Manually triggers embedding generation for documents that are missing them. |
| `help`                                | Shows the list of available commands.                                       |
| `quit`, `exit`, `q`                   | Exits the application.                                                      |

### Example Walkthrough

1.  **Load sample data**:
    `GroqRAG> load_sample`

2.  **Ask a question**:
    `GroqRAG> ask What is machine learning?`

3.  **Search for documents**:
    `GroqRAG> search benefits of a healthy lifestyle`

4.  **Add a custom document**:
    `GroqRAG> add space_001|The James Webb Space Telescope is the successor to Hubble.|science`

5.  **Ask a question about the new document**:
    `GroqRAG> ask what is the successor to the hubble telescope?`

-----

## manualmente Inserting Documents into MongoDB

The `process_manual` command is designed to handle cases where documents are added to MongoDB from an external source (e.g., another application, a database import, or manual insertion via MongoDB Compass). These documents might not have an `embedding` field.

### Sample Documents for Manual Insertion

Here is a JSON array of documents you can import directly into your `groq_rag.documents` collection using MongoDB Compass or `mongoimport`. Notice they are missing the `embedding` field.

```json
[
  {
    "doc_id": "history_001",
    "content": "The Roman Empire was one of the most influential civilizations in world history, known for its contributions to law, architecture, and engineering. It fell in 476 AD.",
    "metadata": { "category": "history", "source": "manual_import" },
    "created_at": { "$date": "2025-07-28T02:00:00Z" }
  },
  {
    "doc_id": "art_001",
    "content": "The Renaissance was a fervent period of European cultural, artistic, political and economic 'rebirth' following the Middle Ages. Famous artists include Leonardo da Vinci and Michelangelo.",
    "metadata": { "category": "art", "source": "manual_import" },
    "created_at": { "$date": "2025-07-28T02:01:00Z" }
  }
]
```

### How to Process Them

1.  **Insert the data**: Use your favorite MongoDB tool to insert the JSON documents above.
2.  **Run the command in the CLI**:
    `GroqRAG> process_manual`

The application will find these two new documents, generate their embeddings, and update them in the database, making them available for searching and questioning. The logs will show the progress.

-----

## 🐍 Code with Comments

Here is the complete, commented Python script.

```python
# filename: groq_rag_system.py

# Basic MongoDB RAG System with Groq
# Requirements: pip install pymongo sentence-transformers groq python-dotenv numpy

import os
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

# Core imports for database, embeddings, and LLM
import pymongo
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Configure basic logging to show informational messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Document:
    """A dataclass to represent the structure of a document (optional, for type hinting)."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None

class GroqMongoRAG:
    """A class that encapsulates the entire RAG system logic."""
    
    def __init__(self, 
                 mongo_uri: str = "mongodb://localhost:27017/",
                 database_name: str = "groq_rag",
                 collection_name: str = "documents",
                 groq_api_key: str = None,
                 embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initializes the RAG system by connecting to services and setting up the database.
        
        Args:
            mongo_uri (str): The connection string for the MongoDB instance.
            database_name (str): The name of the database to use.
            collection_name (str): The name of the collection to store documents in.
            groq_api_key (str): The API key for the Groq service.
            embedding_model (str): The name of the SentenceTransformer model for embeddings.
        """
        
        # Initialize the MongoDB client and select the database and collection
        self.client = MongoClient(mongo_uri)
        self.db = self.client[database_name]
        self.collection = self.db[collection_name]
        
        # Initialize the Groq client with the API key from args or environment variables
        self.groq_client = Groq(
            api_key=groq_api_key or os.getenv("GROQ_API_KEY")
        )
        
        # Load the specified sentence-transformer model for creating embeddings
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Create necessary indexes in the MongoDB collection for performance
        self._create_indexes()
        
        # Automatically process any documents that might be in the DB without an embedding
        self._process_unembedded_documents()
        
        logger.info("Groq MongoDB RAG initialized successfully")
    
    def _create_indexes(self):
        """Creates indexes on the MongoDB collection to speed up queries."""
        try:
            # Create a unique index on 'doc_id' to prevent duplicate documents
            self.collection.create_index([("doc_id", 1)], unique=True)
            # Create a text index on 'content' for standard text searches (not vector search)
            self.collection.create_index([("content", "text")])
            # Create an index on metadata fields for faster filtering
            self.collection.create_index([("metadata.category", 1)])
            # Create an index on the 'embedding' field. While not for vector search itself
            # in this simple implementation, it helps quickly find docs that have embeddings.
            self.collection.create_index([("embedding", 1)]) 
            logger.info("MongoDB indexes created")
        except Exception as e:
            # Log a warning if index creation fails (e.g., they already exist)
            logger.warning(f"Index creation warning: {e}")
            
    def _process_unembedded_documents(self):
        """
        Finds and processes documents that were added to MongoDB without an embedding.
        This makes the system resilient to manual data imports.
        """
        logger.info("Checking for documents without embeddings...")
        try:
            # Find all documents where the 'embedding' field does not exist
            unembedded_docs = list(self.collection.find({"embedding": {"$exists": False}}))
            
            if unembedded_docs:
                logger.info(f"Found {len(unembedded_docs)} documents without embeddings. Processing them...")
                
                updates = []
                for doc in unembedded_docs:
                    doc_id = doc.get("doc_id", str(doc["_id"])) # Use MongoDB's internal _id as a fallback
                    content = doc.get("content")
                    
                    if content:
                        try:
                            # Generate the embedding for the document's content
                            embedding = self.embedding_model.encode(content).tolist()
                            # Prepare a bulk update operation to add the embedding
                            updates.append(
                                pymongo.UpdateOne(
                                    {"_id": doc["_id"]}, # Target the document by its unique _id
                                    {"$set": {"embedding": embedding, "created_at": datetime.utcnow()}}
                                )
                            )
                        except Exception as e:
                            logger.error(f"Error generating embedding for document {doc_id}: {e}")
                    else:
                        logger.warning(f"Document {doc_id} has no content, skipping embedding.")
                
                # Perform all the updates in a single, efficient bulk write operation
                if updates:
                    try:
                        result = self.collection.bulk_write(updates)
                        logger.info(f"Successfully added embeddings to {result.modified_count} documents.")
                    except Exception as e:
                        logger.error(f"Error performing bulk update for embeddings: {e}")
            else:
                logger.info("All documents appear to have embeddings.")
        except Exception as e:
            logger.error(f"Error processing unembedded documents: {e}")

    def add_document(self, doc_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Adds a single document to the collection after generating its embedding."""
        try:
            # Generate the vector embedding from the document's content
            embedding = self.embedding_model.encode(content).tolist()
            
            # Construct the document dictionary
            document = {
                "doc_id": doc_id,
                "content": content,
                "metadata": metadata or {},
                "embedding": embedding,
                "created_at": datetime.utcnow()
            }
            
            # Use replace_one with upsert=True to insert the document or update it if it already exists
            self.collection.replace_one(
                {"doc_id": doc_id}, 
                document, 
                upsert=True
            )
            
            logger.info(f"Document {doc_id} added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error adding document {doc_id}: {e}")
            return False
    
    def add_documents_batch(self, documents: List[Dict[str, Any]]) -> int:
        """Adds multiple documents in a single batch operation for efficiency."""
        success_count = 0
        batch_docs = []
        
        # Prepare each document by generating its embedding
        for doc_data in documents:
            try:
                doc_id = doc_data["doc_id"]
                content = doc_data["content"]
                metadata = doc_data.get("metadata", {})
                
                embedding = self.embedding_model.encode(content).tolist()
                
                document = {
                    "doc_id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "embedding": embedding,
                    "created_at": datetime.utcnow()
                }
                
                batch_docs.append(document)
                
            except Exception as e:
                logger.error(f"Error preparing document {doc_data.get('doc_id', 'unknown')}: {e}")
        
        # Perform the bulk insert/update operation
        if batch_docs:
            try:
                # Create a list of ReplaceOne operations for the bulk write
                operations = [
                    pymongo.ReplaceOne(
                        {"doc_id": doc["doc_id"]}, 
                        doc, 
                        upsert=True
                    ) for doc in batch_docs
                ]
                
                result = self.collection.bulk_write(operations)
                success_count = result.upserted_count + result.modified_count
                logger.info(f"Batch insert: {success_count} documents processed")
                
            except Exception as e:
                logger.error(f"Batch insert error: {e}")
        
        return success_count
    
    def search_documents(self, 
                         query: str, 
                         top_k: int = 5,
                         similarity_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """
        Searches for documents by comparing the query's embedding with stored embeddings.
        Note: This performs a brute-force search. For large datasets, a dedicated vector
        search index (like MongoDB Atlas Vector Search) would be much more performant.
        """
        try:
            # Generate an embedding for the user's query
            query_embedding = self.embedding_model.encode(query)
            
            # Retrieve all documents that have an embedding
            documents = list(self.collection.find({"embedding": {"$exists": True}}))
            
            if not documents:
                return []
            
            results = []
            for doc in documents:
                doc_embedding = np.array(doc["embedding"])
                
                # Calculate cosine similarity: (A . B) / (||A|| * ||B||)
                dot_product = np.dot(query_embedding, doc_embedding)
                norm_query = np.linalg.norm(query_embedding)
                norm_doc = np.linalg.norm(doc_embedding)

                # Avoid division by zero if an embedding is a zero vector
                if norm_query == 0 or norm_doc == 0:
                    similarity = 0.0
                else:
                    similarity = dot_product / (norm_query * norm_doc)
                
                # Only include documents that meet the similarity threshold
                if similarity >= similarity_threshold:
                    doc_result = {
                        "doc_id": doc["doc_id"],
                        "content": doc["content"],
                        "metadata": doc["metadata"],
                        "similarity_score": float(similarity),
                        "created_at": doc["created_at"]
                    }
                    results.append(doc_result)
            
            # Sort the results by similarity score in descending order and return the top K
            results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []
    
    def generate_answer(self, 
                        question: str, 
                        context_docs: List[Dict[str, Any]],
                        model: str = "llama3-8b-8192",
                        max_tokens: int = 512) -> Dict[str, Any]:
        """Generates an answer using the Groq API based on the provided context."""
        try:
            if not context_docs:
                return {
                    "answer": "I couldn't find relevant information to answer your question.",
                    "sources": [],
                    "model_used": model
                }
            
            # Prepare the context string and list of sources for the response
            context_parts = []
            sources = []
            
            for i, doc in enumerate(context_docs):
                context_parts.append(f"Source {i+1}: {doc['content']}")
                sources.append({
                    "doc_id": doc["doc_id"],
                    "similarity_score": doc["similarity_score"],
                    "metadata": doc.get("metadata", {})
                })
            
            context = "\n\n".join(context_parts)
            
            # Create the prompt for the LLM, instructing it how to behave
            prompt = f"""Based on the following context, please answer the question accurately and concisely. If the answer cannot be found in the context, please say so.

Context:
{context}

Question: {question}

Answer:"""
            
            # Make the API call to Groq's chat completion endpoint
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that answers questions based on the provided context. Be accurate and concise."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                model=model,
                max_tokens=max_tokens,
                temperature=0.1, # Low temperature for more deterministic, factual answers
                stop=None,
            )
            
            answer = chat_completion.choices[0].message.content.strip()
            
            # Return a structured response with the answer, sources, and metadata
            return {
                "answer": answer,
                "sources": sources,
                "model_used": model,
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"Answer generation error: {e}")
            return {
                "answer": f"Error generating answer: {str(e)}",
                "sources": [],
                "model_used": model
            }
    
    def query(self, question: str, top_k: int = 3, model: str = "llama3-8b-8192") -> Dict[str, Any]:
        """The main RAG pipeline: search for documents and then generate an answer."""
        # Step 1: Retrieve relevant documents (Retrieval)
        context_docs = self.search_documents(question, top_k=top_k)
        
        # Step 2: Generate an answer based on the retrieved context (Generation)
        result = self.generate_answer(question, context_docs, model=model)
        
        # Add additional metadata to the final result
        result["question"] = question
        result["retrieved_docs"] = len(context_docs)
        result["timestamp"] = datetime.utcnow()
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Retrieves statistics about the collection from MongoDB."""
        total_docs = self.collection.count_documents({})
        
        # Use MongoDB's aggregation pipeline to count documents by category
        pipeline = [
            {"$group": {"_id": "$metadata.category", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        categories = list(self.collection.aggregate(pipeline))
        
        return {
            "total_documents": total_docs,
            "categories": categories
        }


def generate_sample_data() -> List[Dict[str, Any]]:
    """A helper function to create a list of sample documents for testing."""
    sample_docs = [
        {"doc_id": "tech_001", "content": "Artificial Intelligence (AI) is a branch of computer science...", "metadata": {"category": "technology"}},
        {"doc_id": "tech_002", "content": "Machine Learning is a subset of AI that enables computers to learn...", "metadata": {"category": "technology"}},
        {"doc_id": "health_001", "content": "Regular exercise provides numerous health benefits...", "metadata": {"category": "health"}},
        {"doc_id": "health_002", "content": "A balanced diet should include a variety of foods...", "metadata": {"category": "health"}},
        {"doc_id": "science_001", "content": "Climate change refers to long-term shifts in global temperatures...", "metadata": {"category": "science"}},
        {"doc_id": "science_002", "content": "Photosynthesis is the process by which plants convert light energy...", "metadata": {"category": "science"}},
        {"doc_id": "business_001", "content": "Digital transformation involves integrating digital technologies...", "metadata": {"category": "business"}},
        {"doc_id": "business_002", "content": "Supply chain management coordinates the flow of goods...", "metadata": {"category": "business"}}
    ]
    return sample_docs


class GroqRAGCLI:
    """A simple Command-Line Interface (CLI) to interact with the RAG system."""
    
    def __init__(self, rag_system: GroqMongoRAG):
        """Initializes the CLI with an instance of the RAG system."""
        self.rag = rag_system
    
    def run(self):
        """Starts the main interactive loop for the CLI."""
        print("🤖 Groq MongoDB RAG System")
        print("Commands: ask, search, stats, add, load_sample, process_manual, help, quit")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                command = input("\nGroqRAG> ").strip()
                
                # Command handling logic
                if command.lower() in ["quit", "exit", "q"]:
                    print("Goodbye! 👋")
                    break
                elif command.lower() == "help":
                    self.show_help()
                elif command.startswith("ask "):
                    self.handle_question(command[4:])
                elif command.startswith("search "):
                    self.handle_search(command[7:])
                elif command.lower() == "stats":
                    self.show_stats()
                elif command.lower() == "load_sample":
                    self.load_sample_data()
                elif command.lower() == "process_manual":
                    self.process_manual_documents()
                elif command.startswith("add "):
                    self.handle_add_document(command[4:])
                else:
                    print("Unknown command. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
    
    def show_help(self):
        """Displays the help message with all available commands."""
        print("\n📚 Available Commands:")
        print("  ask <question>                - Ask a question using RAG")
        print("  search <query>                - Search for documents")
        print("  stats                         - Show system statistics")
        print("  add <id>|<content>|<category> - Add a document (use '|' as separator)")
        print("  load_sample                   - Load sample data")
        print("  process_manual                - Process documents added manually to MongoDB")
        print("  help                          - Show this help message")
        print("  quit/exit/q                   - Exit the program")
    
    def handle_question(self, question: str):
        """Handles the 'ask' command."""
        if not question:
            print("Please provide a question after 'ask'.")
            return
        print(f"🔍 Processing: {question}")
        result = self.rag.query(question, top_k=3)
        print(f"\n💭 Answer: {result['answer']}")
        print(f"📊 Retrieved {result['retrieved_docs']} documents. Model: {result['model_used']}.")
        if result['sources']:
            print(f"\n📚 Sources:")
            for i, source in enumerate(result['sources'], 1):
                print(f"  {i}. ID: {source['doc_id']} (Similarity: {source['similarity_score']:.3f})")
    
    def handle_search(self, query: str):
        """Handles the 'search' command."""
        if not query:
            print("Please provide a query after 'search'.")
            return
        print(f"🔎 Searching: {query}")
        results = self.rag.search_documents(query, top_k=5)
        if results:
            print(f"\n📋 Found {len(results)} documents:")
            for i, doc in enumerate(results, 1):
                print(f"\n{i}. ID: {doc['doc_id']} (Similarity: {doc['similarity_score']:.3f})")
                print(f"    Content: {doc['content'][:120]}...")
        else:
            print("No relevant documents found.")
    
    def show_stats(self):
        """Handles the 'stats' command."""
        stats = self.rag.get_stats()
        print(f"\n📊 System Statistics:")
        print(f"    Total Documents: {stats['total_documents']}")
        if stats['categories']:
            print(f"\n📁 Categories:")
            for cat in stats['categories']:
                print(f"    - {cat['_id'] or 'Uncategorized'}: {cat['count']} documents")
    
    def load_sample_data(self):
        """Handles the 'load_sample' command."""
        print("📥 Loading sample data...")
        sample_docs = generate_sample_data()
        count = self.rag.add_documents_batch(sample_docs)
        print(f"✅ Loaded {count} sample documents.")
    
    def process_manual_documents(self):
        """Handles the 'process_manual' command."""
        print("🔄 Processing manually added documents (generating missing embeddings)...")
        self.rag._process_unembedded_documents()
        print("Processing complete. Check logs for details.")
            
    def handle_add_document(self, doc_string: str):
        """Handles the 'add' command."""
        parts = doc_string.split("|", 2)
        if len(parts) < 2:
            print("Usage: add doc_id|content|category (category is optional)")
            return
        
        doc_id, content = parts[0].strip(), parts[1].strip()
        category = parts[2].strip() if len(parts) > 2 else "general"
        
        if self.rag.add_document(doc_id=doc_id, content=content, metadata={"category": category}):
            print(f"✅ Document '{doc_id}' added successfully.")
        else:
            print(f"❌ Failed to add document '{doc_id}'.")


# Main execution block
def main():
    """The main entry point of the script."""
    print("🚀 Initializing Groq MongoDB RAG System...")
    
    # Ensure the required API key is set in the environment
    if not os.getenv("GROQ_API_KEY"):
        print("❌ FATAL: GROQ_API_KEY environment variable is not set.")
        print("Please create a .env file with your key. Get one from: https://console.groq.com/keys")
        return
    
    try:
        # Initialize the RAG system
        rag_system = GroqMongoRAG(
            mongo_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017/"),
            groq_api_key=os.getenv("GROQ_API_KEY")
        )
        
        print("✅ System initialized successfully!")
        
        # Start the interactive command-line interface
        cli = GroqRAGCLI(rag_system)
        cli.run()
        
    except Exception as e:
        print(f"❌ Initialization failed: {e}")
        print("Please ensure MongoDB is running and your .env file is configured correctly.")


if __name__ == "__main__":
    main()
```



This is a MongoDB `insertMany` operation. 
---

## What it Does

The command `db.documents.insertMany([...])` is used in MongoDB to **insert multiple new documents into the `documents` collection** within your current database. Each object within the square brackets `[]` represents a single document to be inserted.

---

## Breakdown of Each Document

Let's look at the structure of a typical document being inserted:

* **`doc_id`**: This is a unique identifier for the document, like `"doc_021"`.
* **`content`**: This field holds the main textual information or description, for example, `"Quantum computing leverages quantum bits to perform computations beyond classical capabilities."`.
* **`metadata`**: This is an embedded document (or an object within the main document) that holds additional descriptive information. In this case, it contains:
    * **`category`**: A string indicating the broad topic or classification of the content, such as `"Quantum Computing"`, `"Machine Learning"`, `"Blockchain"`, `"AI"`, or `"Cloud Computing"`.
* **`created_at`**: This field stores a timestamp indicating when the document was created. `new Date()` automatically generates the current date and time.

---
```javascript
db.documents.insertMany([
  {
    doc_id: "doc_021",
    content: "Quantum computing leverages quantum bits to perform computations beyond classical capabilities.",
    metadata: { category: "Quantum Computing" },
    created_at: new Date()
  },
  {
    doc_id: "doc_022",
    content: "Reinforcement learning is used to train agents to make sequential decisions by maximizing reward.",
    metadata: { category: "Machine Learning" },
    created_at: new Date()
  },
  {
    doc_id: "doc_023",
    content: "Blockchain ensures data integrity through a distributed, immutable ledger system.",
    metadata: { category: "Blockchain" },
    created_at: new Date()
  },
  {
    doc_id: "doc_024",
    content: "Neural networks are inspired by biological neurons and form the foundation of deep learning.",
    metadata: { category: "AI" },
    created_at: new Date()
  },
  {
    doc_id: "doc_025",
    content: "Cloud computing provides scalable resources and services over the internet on demand.",
    metadata: { category: "Cloud Computing" },
    created_at: new Date()
  }
]);
```
