import express from 'express';
import { pipeline } from '@xenova/transformers';
import cors from 'cors';

const app = express();
const port = process.env.PORT || 3000;

// Middleware to parse JSON requests
app.use(express.json({ limit: '10mb' }));
app.use(cors());

let modelPipeline;

async function loadModel() {

  try {
    modelPipeline = await pipeline('feature-extraction');
    console.log('Model loaded successfully');
  } catch (error) {
    console.error('Error loading model:', error);
    process.exit(1);
  }
}

// Load the model at startup
loadModel();

function cosineSimilarity(a, b) {
  if (a.length !== b.length) {
      throw new Error('Both arguments must have the same length');
  }
  let result = 0;
  for (let i = 0; i < a.length; i++) {
      result += a[i] * b[i];
  }
  return result;
}

app.get('/', (req, res) => {
  res.send('Hello World');
});

app.post('/embed', async (req, res) => {
  if (!modelPipeline) {
    return res.status(503).json({ error: 'Model is loading, please try again later' });
  }
  // console.log(req.body);

  const { text } = req.body;
  if (!text) {
    return res.status(400).json({ error: 'Text is required' });
  }

  try {
    const output = await modelPipeline(text, { pooling: 'mean', normalize: true });
    res.json({ embeddings: Object.values(output.data), size: output.data.length });
  } catch (error) {
    console.error('Error generating embeddings:', error);
    res.status(500).json({ error: 'Error generating embeddings' });
  }
});

app.post('/similarity', async (req, res) => {
  const data = req.body;
  // console.log("Similarity :", data);
  const userEmbedding = data.userEmbedding;
  const eventEmbeddings = data.events;
  const output = [];

  for (const [index, eventEmbedding] of eventEmbeddings.entries()) {
    const embedding = JSON.parse(eventEmbedding.embedding);
    const uEmbedding = JSON.parse(userEmbedding);
    const similarity = cosineSimilarity(uEmbedding, embedding);
    output.push({ id: eventEmbedding.id, similarity: similarity });
  }

  // console.log("Output: ",output);

  // Sort output based on similarity from max to min
  res.json(output.sort((a, b) => b.similarity - a.similarity).map((item) => item.id));
});


// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});