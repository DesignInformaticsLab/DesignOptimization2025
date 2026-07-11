create extension if not exists pgcrypto;

create table if not exists public.students (
  id uuid primary key default gen_random_uuid(),
  university_id text not null unique,
  first_name text not null,
  last_name text not null,
  section text,
  active boolean not null default true,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint students_university_id_not_blank check (length(trim(university_id)) > 0),
  constraint students_first_name_not_blank check (length(trim(first_name)) > 0),
  constraint students_last_name_not_blank check (length(trim(last_name)) > 0)
);

create table if not exists public.lectures (
  id text primary key,
  title text not null,
  created_at timestamptz not null default now()
);

insert into public.lectures (id, title)
values ('gradient_descent_2025', 'Gradient descent for unconstrained optimization')
on conflict (id) do update set title = excluded.title;

create table if not exists public.qa_events (
  id uuid primary key default gen_random_uuid(),
  student_id uuid not null references public.students(id),
  lecture_id text not null references public.lectures(id),
  question text not null,
  model text not null,
  answer_elapsed_ms integer not null,
  quality_elapsed_ms integer,
  total_elapsed_ms integer not null,
  quality_score numeric(3, 2),
  quality_relevance smallint,
  quality_specificity smallint,
  quality_math_depth smallint,
  quality_effort smallint,
  quality_rationale text,
  created_at timestamptz not null default now(),
  constraint qa_events_question_not_blank check (length(trim(question)) > 0),
  constraint qa_events_answer_elapsed_nonnegative check (answer_elapsed_ms >= 0),
  constraint qa_events_quality_elapsed_nonnegative check (quality_elapsed_ms is null or quality_elapsed_ms >= 0),
  constraint qa_events_total_elapsed_nonnegative check (total_elapsed_ms >= 0),
  constraint qa_events_quality_score_range check (quality_score is null or quality_score between 0 and 3),
  constraint qa_events_quality_relevance_range check (quality_relevance is null or quality_relevance between 0 and 3),
  constraint qa_events_quality_specificity_range check (quality_specificity is null or quality_specificity between 0 and 3),
  constraint qa_events_quality_math_depth_range check (quality_math_depth is null or quality_math_depth between 0 and 3),
  constraint qa_events_quality_effort_range check (quality_effort is null or quality_effort between 0 and 3)
);

create index if not exists qa_events_student_created_idx
  on public.qa_events (student_id, created_at desc);

create index if not exists qa_events_lecture_created_idx
  on public.qa_events (lecture_id, created_at desc);

create or replace view public.engagement_summary as
select
  s.university_id,
  s.first_name,
  s.last_name,
  s.section,
  e.lecture_id,
  count(e.id) as question_count,
  avg(e.quality_score) as avg_quality_score,
  sum(e.answer_elapsed_ms) as total_answer_elapsed_ms,
  sum(e.total_elapsed_ms) as total_elapsed_ms,
  min(e.created_at) as first_question_at,
  max(e.created_at) as last_question_at
from public.students s
left join public.qa_events e on e.student_id = s.id
group by s.id, s.university_id, s.first_name, s.last_name, s.section, e.lecture_id;

alter table public.students enable row level security;
alter table public.lectures enable row level security;
alter table public.qa_events enable row level security;

comment on table public.students is 'Roster imported by instructor. University ID should be the stable roster key.';
comment on table public.qa_events is 'One row per Q&A interaction. Generated answers are not stored by default.';
