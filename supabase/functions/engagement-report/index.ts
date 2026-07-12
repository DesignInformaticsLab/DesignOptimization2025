import { createClient } from "npm:@supabase/supabase-js@2";

const corsHeaders = {
  "Access-Control-Allow-Origin": Deno.env.get("ALLOWED_ORIGIN") || "*",
  "Access-Control-Allow-Headers":
    "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
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

function serviceRoleKey() {
  const secretKeys = Deno.env.get("SUPABASE_SECRET_KEYS");
  if (secretKeys) {
    return JSON.parse(secretKeys).default as string;
  }
  return Deno.env.get("SUPABASE_SERVICE_ROLE_KEY") || "";
}

function requireInstructor(request: Request) {
  const expected = Deno.env.get("ENGAGEMENT_REPORT_TOKEN");
  if (!expected) {
    throw new Error("ENGAGEMENT_REPORT_TOKEN is not configured");
  }

  const auth = request.headers.get("authorization") || "";
  const token = auth.toLowerCase().startsWith("bearer ")
    ? auth.slice(7).trim()
    : "";
  if (token !== expected) {
    return false;
  }
  return true;
}

Deno.serve(async (request) => {
  if (request.method === "OPTIONS") {
    return new Response("ok", { headers: corsHeaders });
  }
  if (request.method !== "GET") {
    return jsonResponse({ error: "Method not allowed" }, 405);
  }

  try {
    if (!requireInstructor(request)) {
      return jsonResponse({ error: "Unauthorized" }, 401);
    }

    const supabaseUrl = Deno.env.get("SUPABASE_URL");
    const supabaseKey = serviceRoleKey();
    if (!supabaseUrl || !supabaseKey) {
      throw new Error("Supabase service credentials are not configured");
    }

    const supabase = createClient(supabaseUrl, supabaseKey);
    const { data, error } = await supabase
      .from("qa_events")
      .select(`
        id,
        lecture_id,
        question,
        model,
        answer_elapsed_ms,
        quality_elapsed_ms,
        total_elapsed_ms,
        quality_score,
        quality_relevance,
        quality_specificity,
        quality_math_depth,
        quality_effort,
        quality_rationale,
        created_at,
        students!inner (
          university_id_hash,
          first_initial,
          last_initial,
          section
        )
      `)
      .order("quality_score", { ascending: false, nullsFirst: false })
      .order("created_at", { ascending: false });

    if (error) {
      throw error;
    }

    const events = (data || []).map((event) => {
      const student = Array.isArray(event.students)
        ? event.students[0]
        : event.students;
      return {
        id: event.id,
        lecture_id: event.lecture_id,
        question: event.question,
        model: event.model,
        answer_elapsed_ms: event.answer_elapsed_ms,
        quality_elapsed_ms: event.quality_elapsed_ms,
        total_elapsed_ms: event.total_elapsed_ms,
        quality_score: event.quality_score,
        quality_relevance: event.quality_relevance,
        quality_specificity: event.quality_specificity,
        quality_math_depth: event.quality_math_depth,
        quality_effort: event.quality_effort,
        quality_rationale: event.quality_rationale,
        created_at: event.created_at,
        student: {
          student_key: student?.university_id_hash,
          first_initial: student?.first_initial,
          last_initial: student?.last_initial,
          section: student?.section,
        },
      };
    });

    const students = new Map<string, {
      student_key: string;
      first_initial: string;
      last_initial: string;
      section: string | null;
      question_count: number;
      avg_quality_score: number | null;
      total_answer_elapsed_ms: number;
      total_elapsed_ms: number;
      questions: typeof events;
    }>();

    for (const event of events) {
      const id = event.student.student_key || "unknown";
      if (!students.has(id)) {
        students.set(id, {
          student_key: id,
          first_initial: event.student.first_initial || "",
          last_initial: event.student.last_initial || "",
          section: event.student.section || null,
          question_count: 0,
          avg_quality_score: null,
          total_answer_elapsed_ms: 0,
          total_elapsed_ms: 0,
          questions: [],
        });
      }
      const student = students.get(id)!;
      student.question_count += 1;
      student.total_answer_elapsed_ms += event.answer_elapsed_ms || 0;
      student.total_elapsed_ms += event.total_elapsed_ms || 0;
      student.questions.push(event);
    }

    const studentRows = Array.from(students.values()).map((student) => {
      student.questions.sort((a, b) => {
        const aScore = Number(a.quality_score ?? -1);
        const bScore = Number(b.quality_score ?? -1);
        return bScore - aScore ||
          Date.parse(b.created_at) - Date.parse(a.created_at);
      });
      const scores = student.questions
        .map((question) => Number(question.quality_score))
        .filter((score) => Number.isFinite(score));
      student.avg_quality_score = scores.length
        ? scores.reduce((sum, score) => sum + score, 0) / scores.length
        : null;
      return student;
    }).sort((a, b) => {
      const qualityDelta = Number(b.avg_quality_score ?? -1) -
        Number(a.avg_quality_score ?? -1);
      return qualityDelta || b.question_count - a.question_count;
    });

    return jsonResponse({
      generated_at: new Date().toISOString(),
      rubric: {
        score_range: "1 to 5",
        score_prompt:
          "As an experienced educator in optimization at a top-tier institute, from 1-5 where 5 is the best, how deep do you think the student understands the materials on current page?",
        note:
          "This is an AI-evaluated engagement signal from the student's question, not a high-stakes grade.",
      },
      students: studentRows,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return jsonResponse({ error: message }, 400);
  }
});
