import { createClient } from "npm:@supabase/supabase-js@2";

declare const EdgeRuntime: {
  waitUntil(promise: Promise<unknown>): void;
};

type ChatMessage = {
  role: "system" | "user" | "assistant";
  content: string;
};

type AskRequest = {
  firstName: string;
  lastName: string;
  universityId: string;
  lectureId: string;
  question: string;
  model?: string;
  context?: unknown;
};

type QualityScore = {
  score: number | null;
  rationale: string | null;
};

const corsHeaders = {
  "Access-Control-Allow-Origin": Deno.env.get("ALLOWED_ORIGIN") || "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

function jsonResponse(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      ...corsHeaders,
      "Content-Type": "application/json",
    },
  });
}

function errorMessage(error: unknown) {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  if (error && typeof error === "object") {
    const record = error as Record<string, unknown>;
    if (typeof record.message === "string") {
      return record.message;
    }
    if (typeof record.error === "string") {
      return record.error;
    }
    return JSON.stringify(record);
  }
  return String(error);
}

function requiredText(value: unknown, field: string, maxLength: number) {
  if (typeof value !== "string") {
    throw new Error(`${field} is required`);
  }
  const normalized = value.trim();
  if (!normalized) {
    throw new Error(`${field} is required`);
  }
  if (normalized.length > maxLength) {
    throw new Error(`${field} is too long`);
  }
  return normalized;
}

function serviceRoleKey() {
  const secretKeys = Deno.env.get("SUPABASE_SECRET_KEYS");
  if (secretKeys) {
    return JSON.parse(secretKeys).default as string;
  }
  return Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function messagesToPrompt(messages: ChatMessage[]) {
  return messages
    .map((message) => `${message.role.toUpperCase()}:\n${message.content}`)
    .join("\n\n")
    .slice(0, 6000);
}

async function callModel(
  model: string,
  messages: ChatMessage[],
  maxTokens: number,
) {
  const endpoint = Deno.env.get("QA_MODEL_ENDPOINT") ||
    "https://text.pollinations.ai/openai";
  const apiKey = Deno.env.get("QA_MODEL_API_KEY");

  let lastError = "";
  for (let attempt = 0; attempt < 3; attempt += 1) {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(apiKey ? { Authorization: `Bearer ${apiKey}` } : {}),
      },
      body: JSON.stringify({
        model,
        temperature: 0.2,
        max_tokens: maxTokens,
        messages,
      }),
    });

    if (response.ok) {
      const data = await response.json();
      return (
        data?.choices?.[0]?.message?.content ||
        data?.choices?.[0]?.text ||
        JSON.stringify(data)
      ).trim();
    }

    lastError = `HTTP ${response.status}: ${
      (await response.text()).slice(0, 240)
    }`;
    await sleep(500 * (attempt + 1));
  }

  const fallbackBase = Deno.env.get("QA_TEXT_FALLBACK_ENDPOINT") ||
    "https://text.pollinations.ai";
  if (fallbackBase) {
    const fallbackPrompt = messagesToPrompt(messages);
    const fallbackResponse = await fetch(
      `${fallbackBase.replace(/\/$/, "")}/${
        encodeURIComponent(fallbackPrompt)
      }`,
    );
    if (fallbackResponse.ok) {
      return (await fallbackResponse.text()).trim();
    }
    lastError += `; fallback HTTP ${fallbackResponse.status}: ${
      (await fallbackResponse.text()).slice(0, 240)
    }`;
  }

  throw new Error(`Model request failed: ${lastError}`);
}

function clampUnderstandingScore(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return null;
  }
  return Math.max(1, Math.min(5, Math.round(value)));
}

function parseUnderstandingQuality(text: string): QualityScore {
  const fallback = {
    score: null,
    rationale: text.slice(0, 240) || null,
  };

  const match = text.match(/\{[\s\S]*\}/);
  if (!match) {
    return fallback;
  }

  try {
    const parsed = JSON.parse(match[0]);
    const score = clampUnderstandingScore(parsed.score);

    return {
      score,
      rationale: typeof parsed.rationale === "string" && parsed.rationale.trim()
        ? parsed.rationale.trim().slice(0, 240)
        : null,
    };
  } catch {
    return fallback;
  }
}

async function evaluateUnderstandingAndUpdate(
  supabase: any,
  eventId: string,
  model: string,
  lectureId: string,
  question: string,
  context: unknown,
) {
  const qualityStartedAt = Date.now();
  try {
    const qualityText = await callModel(model, [
      {
        role: "system",
        content:
          "You are an experienced educator in optimization at a top-tier institute. " +
          "Evaluate only the student's question as evidence of how deeply the student understands the material on the current page. " +
          "Return only compact JSON with fields: score, rationale. " +
          "score must be an integer from 1 to 5, where 1 means very shallow or confused understanding, 3 means reasonable basic understanding, and 5 means unusually deep, precise, and conceptually connected understanding. " +
          "The rationale should be one concise sentence.",
      },
      {
        role: "user",
        content: "Lecture ID:\n" + lectureId +
          "\n\nCurrent page context:\n" +
          JSON.stringify(context, null, 2).slice(0, 6000) +
          "\n\nStudent question:\n" +
          question,
      },
    ], 120);
    const quality = parseUnderstandingQuality(qualityText);
    const { error } = await supabase.from("qa_events").update({
      quality_score: quality.score,
      quality_relevance: null,
      quality_specificity: null,
      quality_math_depth: null,
      quality_effort: null,
      quality_rationale: quality.rationale,
      quality_elapsed_ms: Date.now() - qualityStartedAt,
    }).eq("id", eventId);
    if (error) {
      console.error("quality update failed", error);
    }
  } catch (error) {
    const { error: updateError } = await supabase.from("qa_events").update({
      quality_rationale: `AI understanding evaluation failed: ${
        errorMessage(error)
      }`.slice(0, 240),
      quality_elapsed_ms: Date.now() - qualityStartedAt,
    }).eq("id", eventId);
    if (updateError) {
      console.error("quality failure update failed", updateError);
    }
  }
}

Deno.serve(async (request) => {
  if (request.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }
  if (request.method !== "POST") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }

  const startedAt = Date.now();

  try {
    const body = (await request.json()) as Partial<AskRequest>;
    const firstName = requiredText(body.firstName, "firstName", 80);
    const lastName = requiredText(body.lastName, "lastName", 80);
    const universityId = requiredText(body.universityId, "universityId", 80);
    const lectureId = requiredText(body.lectureId, "lectureId", 120);
    const question = requiredText(body.question, "question", 4000);
    const model = typeof body.model === "string" && body.model.trim()
      ? body.model.trim()
      : "openai-fast";

    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const supabaseKey = serviceRoleKey();
    if (!supabaseUrl || !supabaseKey) {
      throw new Error("Supabase service credentials are not configured");
    }

    const supabase = createClient(supabaseUrl, supabaseKey);
    const { data: student, error: studentError } = await supabase
      .from("students")
      .select("id, first_name, last_name, active")
      .eq("university_id", universityId)
      .eq("active", true)
      .maybeSingle();

    if (studentError) {
      throw studentError;
    }
    if (!student) {
      return jsonResponse(
        { error: "Student was not found on the roster" },
        403,
      );
    }
    if (
      student.first_name.trim().toLowerCase() !== firstName.toLowerCase() ||
      student.last_name.trim().toLowerCase() !== lastName.toLowerCase()
    ) {
      return jsonResponse({
        error: "Name does not match the roster entry for this university ID",
      }, 403);
    }

    const context = body.context ?? {};
    const answerMaxTokens = Number(Deno.env.get("QA_ANSWER_MAX_TOKENS") || 450);
    const answerStartedAt = Date.now();
    const answer = await callModel(model, [
      {
        role: "system",
        content:
          "You are a concise teaching assistant for a graduate design optimization class. " +
          "Use the supplied lecture notes and notebook context. If the answer is not supported by context, say what is missing. " +
          "When writing equations, use LaTeX delimiters \\(...\\) for inline math and \\[...\\] for display math so MathJax can render them.",
      },
      {
        role: "user",
        content: "Context JSON:\n" +
          JSON.stringify(context, null, 2) +
          "\n\nStudent question:\n" +
          question,
      },
    ], Number.isFinite(answerMaxTokens) ? answerMaxTokens : 450);
    const answerElapsedMs = Date.now() - answerStartedAt;

    const totalElapsedMs = Date.now() - startedAt;
    const { data: event, error: insertError } = await supabase.from("qa_events")
      .insert({
        student_id: student.id,
        lecture_id: lectureId,
        question,
        model,
        answer_elapsed_ms: answerElapsedMs,
        quality_elapsed_ms: null,
        total_elapsed_ms: totalElapsedMs,
        quality_score: null,
        quality_relevance: null,
        quality_specificity: null,
        quality_math_depth: null,
        quality_effort: null,
        quality_rationale: "AI understanding evaluation pending.",
      }).select("id").single();

    if (insertError) {
      throw insertError;
    }

    if (event?.id) {
      const qualityTask = evaluateUnderstandingAndUpdate(
        supabase,
        event.id,
        model,
        lectureId,
        question,
        context,
      );
      EdgeRuntime.waitUntil(qualityTask);
    }

    return jsonResponse({
      answer,
      metrics: {
        answerElapsedMs,
        qualityElapsedMs: null,
        totalElapsedMs,
        quality: {
          score: null,
          rationale:
            "AI understanding evaluation is running in the background.",
        },
      },
    });
  } catch (error) {
    return jsonResponse({ error: errorMessage(error) }, 400);
  }
});
