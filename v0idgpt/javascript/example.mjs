import { loadModel } from "./model.mjs";

const model = await loadModel();

// Generate text (temperature + top-k sampling; pass a seed for reproducible output)
const text = model.generate("hello there", { maxTokens: 40, temperature: 0.7, topK: 50 });
console.log(text);

// Full details: generated ids/tokens, settings and why it stopped
const run = model.generateDetailed("hello there", { maxTokens: 20, seed: 42 });
console.log(run.fullText, run.stoppedBy);

// Next-token distribution
console.log(model.predict("hello there").top.slice(0, 5));
