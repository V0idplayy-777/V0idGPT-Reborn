export interface PredictionResult {
  task: "binary" | "multiclass" | "regression" | "reconstruction" | "language" | "raw";
  raw: number[];
  predicted?: number;
  label?: string;
  probability?: number;
  probabilities?: number[];
  scores?: number[];
  score?: number;
  value?: number;
  values?: number[];
  named?: Record<string, number>;
  reconstruction?: number[];
  reconstructionError?: number;
  next?: string | null;
  top?: { index?: number; id?: number; label?: string; token?: string; value?: number; probability?: number }[];
  output?: number[];
  shape?: number[];
}

export interface GenerateOptions {
  maxTokens?: number;
  temperature?: number;
  topK?: number;
  seed?: number;
  stop?: string[];
  stopAtEos?: boolean;
  includePrompt?: boolean;
}

export interface GenerationResult {
  text: string;
  fullText: string;
  ids: number[];
  tokens: string[];
  promptIds: number[];
  stoppedBy: string | null;
  settings: { maxTokens: number; temperature: number; topK: number; seed: number; stop: string[] };
}

export interface VerifyReport {
  ok: boolean;
  runtimeVersion: string;
  checks: { name: string; status: "pass" | "fail" | "warn"; detail: string }[];
}

export interface ModelInfo {
  name: string;
  task: PredictionResult["task"];
  inputShape: number[];
  outputShape: number[];
  paramCount: number;
  layers: { name: string; type: string; inputShape: number[]; outputShape: number[]; params: Record<string, number> }[];
  classNames: string[] | null;
  targetNames: string[] | null;
  outputsAreProbabilities: boolean;
  tokenizer: { level: "char" | "word" | "bpe"; vocabSize: number; seqLen: number; padToken: string; merges?: number; eos: string | null } | null;
  preprocessing: { kind: "tabular"; columns: string[]; normalize: string; missing: string } | null;
  image: { height: number; width: number; channels: number; layout: "HWC"; range: [number, number]; description: string } | null;
  sampling: { temperature: number; topK: number };
  runtimeVersion: string;
  exportedAt: string;
  limitations: string[];
}

export type ImageInput = { data: ArrayLike<number>; width: number; height: number; channels?: number } | ArrayLike<number>;

export interface NeuronForgeModel {
  info: ModelInfo;
  forward(input: ArrayLike<number> | number[] | number[][], batchSize?: number): { shape: number[]; data: Float32Array };
  predict(input: unknown, options?: { preprocessed?: boolean }): PredictionResult;
  predictBatch(inputs: unknown[], options?: { preprocessed?: boolean }): PredictionResult[];
  encode(row: Record<string, unknown> | unknown[]): number[];
  tokenize(text: string): number[];
  detokenize(ids: number[]): string;
  nextToken(text: string): PredictionResult;
  generate(prompt: string, options?: GenerateOptions): string;
  generateDetailed(prompt: string, options?: GenerateOptions): GenerationResult;
  imageToInput(image: ImageInput, options?: { scale?: number; invert?: boolean }): number[];
  predictImage(image: ImageInput, options?: { scale?: number; invert?: boolean }): PredictionResult;
  verify(): VerifyReport;
  getWeights(): Record<string, Float32Array>;
}

export declare function loadModel(): Promise<NeuronForgeModel>;
export declare function loadModelSync(): NeuronForgeModel;
export declare function createModel(bundle: unknown, options?: { skipChecksums?: boolean }): NeuronForgeModel;
export declare const bundle: unknown;
export declare const NF_RUNTIME_VERSION: string;
export default loadModel;
