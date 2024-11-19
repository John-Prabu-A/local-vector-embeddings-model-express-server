# Real-Time Embedding and Similarity Service

A Node.js-based API for generating text embeddings and calculating cosine similarity between them. This service uses the `@xenova/transformers` library for feature extraction and provides endpoints for embedding generation and similarity computation.

## Features

- **Text Embedding Generation**: Converts input text into numerical embeddings using a pre-trained model.
- **Cosine Similarity Calculation**: Computes the similarity between two embeddings to determine their closeness.
- **Sorting by Similarity**: Returns sorted results based on similarity scores.

## Prerequisites

- **Node.js** (v16 or higher)
- **npm** or **yarn**
- PostgreSQL (if applicable)

## Installation

1. **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd backend
    ```

2. **Install dependencies**:
    ```bash
    npm install
    ```

3. **Set up environment variables**:

    Create a `.env` file in the root directory with the following content:
    ```env
    PORT=3000
    SERVER_URL=<your_server_url>
    ```

## Usage

1. **Start the server**:
    ```bash
    npm start
    ```

2. **Access the API**:
    The server will run on `http://localhost:3000` by default.

## Endpoints

### 1. **GET /**

   **Description**: Basic health check endpoint.

   **Response**:
   ```json
   {
     "message": "Hello World"
   }
   ```

---

### 2. **POST /embed**

   **Description**: Generates embeddings for a given text.

   **Request**:
   ```json
   {
     "text": "Your input text here"
   }
   ```

   **Response**:
   ```json
   {
     "embeddings": [/* Array of numerical values */],
     "size": 384
   }
   ```

---

### 3. **POST /similarity**

   **Description**: Calculates similarity between a user embedding and a list of event embeddings.

   **Request**:
   ```json
   {
     "userEmbedding": "[/* JSON stringified embedding array */]",
     "events": [
       { "id": 1, "embedding": "[/* JSON stringified embedding array */]" },
       { "id": 2, "embedding": "[/* JSON stringified embedding array */]" }
     ]
   }
   ```

   **Response**:
   ```json
   [1, 2] // Array of event IDs sorted by similarity
   ```

## Error Handling

- **503 Service Unavailable**: Model is still loading.
- **400 Bad Request**: Missing required parameters.
- **500 Internal Server Error**: Errors during embedding generation or similarity calculation.

## Scripts

- **Start the server**:
    ```bash
    npm start
    ```

## Technologies Used

- **Node.js**: Server-side JavaScript runtime.
- **Express.js**: Web framework for building APIs.
- **@xenova/transformers**: Pre-trained models for feature extraction.
- **Cors**: Middleware to handle Cross-Origin Resource Sharing (CORS).

## Future Enhancements

- Add authentication for secure API access.
- Implement caching for frequently used embeddings.
- Expand support for other model pipelines (e.g., text classification).

## License

This project is licensed under the [MIT License](LICENSE).

---

## Author

**John Prabu A**  
*B.Tech in Information Technology*  
Madras Institute of Technology, Anna University.