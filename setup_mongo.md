# MongoDB Collection Setup for Documents

This guide details the MongoDB commands for creating a `documents` collection, applying schema validation, setting up essential indexes, and inserting initial data.

## 1\. Create Collection with Schema Validation

The `db.createCollection()` command is used to create a new collection named `documents`. We've added a `$jsonSchema` validator to enforce the structure and data types of documents inserted into this collection.

```javascript
db.createCollection("documents", {
  validator: {
    $jsonSchema: {
      bsonType: "object", // Documents must be BSON objects
      required: ["doc_id", "content"], // 'doc_id' and 'content' fields are mandatory
      properties: {
        doc_id: {
          bsonType: "string", // 'doc_id' must be a string
          description: "must be a string and is required"
        },
        content: {
          bsonType: "string", // 'content' must be a string
          description: "must be a string and is required"
        },
        metadata: {
          bsonType: "object", // 'metadata' must be an object (optional)
          description: "must be an object"
        },
        embedding: {
          bsonType: "array", // 'embedding' must be an array
          items: {
            bsonType: "double" // Elements within the 'embedding' array must be doubles (floating-point numbers)
          },
          description: "must be an array of numbers"
        },
        created_at: {
          bsonType: "date", // 'created_at' must be a date (optional)
          description: "must be a date"
        }
      }
    }
  }
});
```

**Explanation of Schema Fields:**

  * `doc_id`: A unique string identifier for each document.
  * `content`: The main textual content of the document.
  * `metadata`: An optional object to store additional, unstructured information about the document.
  * `embedding`: An optional array of numbers, typically used for storing vector embeddings for similarity searches in RAG (Retrieval Augmented Generation) systems.
  * `created_at`: An optional date field to record when the document was created.

## 2\. Create Indexes

Indexes are crucial for improving query performance. Here, we create several indexes on commonly queried fields.

```javascript
// Create a unique index on 'doc_id' to ensure no duplicate document IDs
db.documents.createIndex({ "doc_id": 1 }, { unique: true });

// Create a text index on 'content' to enable full-text search capabilities
db.documents.createIndex({ "content": "text" });

// Create an index on 'metadata.category' for efficient filtering by category
db.documents.createIndex({ "metadata.category": 1 });

// Create an index on 'created_at' for efficient time-based queries
db.documents.createIndex({ "created_at": 1 });
```

**Explanation of Indexes:**

  * `doc_id: 1` (Unique Index): Ensures that each `doc_id` is unique across the collection. The `1` indicates an ascending order index.
  * `content: "text"` (Text Index): Allows for efficient full-text searches on the `content` field.
  * `metadata.category: 1` (Single Field Index): Improves the performance of queries that filter or sort by the `category` field within the `metadata` object.
  * `created_at: 1` (Single Field Index): Optimizes queries that involve date ranges or sorting by creation date.

## 3\. Insert Initial Data

Finally, we insert a set of sample documents into the `documents` collection using `insertMany()`. These documents adhere to the defined schema.

```javascript
db.documents.insertMany([
  {
    doc_id: "doc001",
    content: "The Great Wall of China was built to protect against invasions."
  },
  {
    doc_id: "doc002",
    content: "Photosynthesis allows plants to convert sunlight into chemical energy."
  },
  {
    doc_id: "doc003",
    content: "Mahatma Gandhi advocated non-violent resistance to lead India to independence."
  },
  {
    doc_id: "doc004",
    content: "The water cycle consists of evaporation, condensation, precipitation, and collection."
  },
  {
    doc_id: "doc005",
    content: "Python is a versatile programming language widely used in data science and AI."
  },
  {
    doc_id: "doc006",
    content: "The moon orbits the Earth approximately every 27.3 days."
  },
  {
    doc_id: "doc007",
    content: "Electric vehicles help reduce greenhouse gas emissions and reliance on fossil fuels."
  },
  {
    doc_id: "doc008",
    content: "Machine learning models improve automatically through experience and data."
  },
  {
    doc_id: "doc009",
    content: "The Indus Valley Civilization was one of the world's earliest urban cultures."
  },
  {
    doc_id: "doc010",
    content: "The human brain processes visual information faster than textual data."
  }
]);
```

