import express from "express";
import { pipeline } from "@xenova/transformers";
import cors from "cors";
import bodyParser from "body-parser";

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(express.json({ limit: "10mb" }));
app.use(bodyParser.json());
app.use(cors());

let modelPipeline;
let generator;

// Function to load the model
async function loadModel() {
  try {
    console.log("Loading the Feature Extraction model...");
    modelPipeline = await pipeline("feature-extraction");
    console.log("Model loaded successfully");
    console.log("Loading the GPT-Neo model...");
    generator = await pipeline("text-generation", "Xenova/gpt2"); // Replace with your model
    console.log("Model loaded successfully!");
  } catch (error) {
    console.error("Error loading model:", error);
    process.exit(1); // Exit if the model fails to load
  }
}

// Load the model at startup
loadModel();

// Utility function: Cosine similarity
function cosineSimilarity(a, b) {
  if (a.length !== b.length) {
    throw new Error("Both arguments must have the same length");
  }
  return a.reduce((sum, ai, i) => sum + ai * b[i], 0);
}

// Routes
app.get("/", (req, res) => {
  res.send("Hello World");
});

app.post("/embed", async (req, res) => {
  if (!modelPipeline) {
    return res
      .status(503)
      .json({ error: "Model is loading, please try again later" });
  }

  const { text } = req.body;
  if (!text) {
    return res.status(400).json({ error: "Text is required" });
  }

  try {
    const output = await modelPipeline(text, {
      pooling: "mean",
      normalize: true,
    });
    res.json({
      embeddings: Object.values(output.data),
      size: output.data.length,
    });
  } catch (error) {
    console.error("Error generating embeddings:", error);
    res.status(500).json({ error: "Error generating embeddings" });
  }
});

app.post("/similarity", (req, res) => {
  const { userEmbedding, events } = req.body;

  if (!userEmbedding || !events) {
    return res
      .status(400)
      .json({ error: "userEmbedding and events are required" });
  }

  try {
    const parsedUserEmbedding = JSON.parse(userEmbedding);
    const output = events.map((event) => {
      const parsedEventEmbedding = JSON.parse(event.embedding);
      const similarity = cosineSimilarity(
        parsedUserEmbedding,
        parsedEventEmbedding
      );
      return { id: event.id, similarity };
    });

    // Sort by similarity in descending order and return only the IDs
    res.json(
      output.sort((a, b) => b.similarity - a.similarity).map((item) => item.id)
    );
  } catch (error) {
    console.error("Error calculating similarity:", error);
    res.status(500).json({ error: "Error calculating similarity" });
  }
});

// Endpoint to generate a response
app.post("/chat", async (req, res) => {
  const { message } = req.body;

  if (!message) {
    return res.status(400).json({ error: "Message is required" });
  }

  try {
    const output = await generator(message, {
      max_length: 50,
      num_return_sequences: 1,
      temperature: 0.7,
    });

    const reply = output[0].generated_text;
    res.json({ reply });
  } catch (error) {
    console.error("Error generating response:", error);
    res.status(500).json({ error: "Failed to generate response" });
  }
});

// Start the server
app.listen(port, () => {
  console.log(
    `Server is running on ${
      process.env.SERVER_URL || `http://localhost:${port}`
    }`
  );
});
