/* eslint-disable */
/**
 * Generated Convex function references. Run `npx convex codegen` after adding
 * or renaming Convex functions.
 */
import {
  anyApi,
  type ApiFromModules,
  type FilterApi,
  type FunctionReference,
} from "convex/server";
import type * as apiKeys from "../apiKeys.js";
import type * as apiFunctions from "../api.js";
import type * as billing from "../billing.js";
import type * as dashboard from "../dashboard.js";
import type * as inference from "../inference.js";

const fullApi: ApiFromModules<{
  apiKeys: typeof apiKeys;
  api: typeof apiFunctions;
  billing: typeof billing;
  dashboard: typeof dashboard;
  inference: typeof inference;
}> = anyApi as any;

export const api: FilterApi<typeof fullApi, FunctionReference<any, "public">> = anyApi as any;
export const internal: FilterApi<typeof fullApi, FunctionReference<any, "internal">> = anyApi as any;
