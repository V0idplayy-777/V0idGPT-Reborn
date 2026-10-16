import { v } from "convex/values";
import { internalAction } from "./_generated/server.js";
import { ACTIVE_MODEL_ID } from "./lib/limits.js";
import { loadModelSync } from "../v0idgpt/javascript/model.mjs";

let loadedModel: ReturnType<typeof loadModelSync> | undefined;
function getModel() {
  if (!loadedModel) loadedModel = loadModelSync();
  return loadedModel;
}

export const generate = internalAction({
  args: {
    prompt: v.string(),
    maxTokens: v.number(),
    temperature: v.number(),
    topK: v.number(),
    seed: v.optional(v.number()),
    stop: v.optional(v.array(v.string())),
  },
  handler: async (_ctx, args) => {
    const model = getModel();
    const promptTokens = model.tokenize(args.prompt).length;
    const result = model.generateDetailed(args.prompt, {
      maxTokens: args.maxTokens,
      temperature: args.temperature,
      topK: args.topK,
      ...(args.seed === undefined ? {} : { seed: args.seed }),
      ...(args.stop === undefined ? {} : { stop: args.stop }),
    });
    return {
      text: result.text,
      finishReason: result.stoppedBy ?? "length",
      promptTokens,
      completionTokens: result.ids.length,
      model: ACTIVE_MODEL_ID,
    };
  },
});
