alter table public.qa_events
  drop constraint if exists qa_events_student_id_fkey;

drop view if exists public.engagement_summary;

alter table public.students
  add column if not exists university_id_hash text,
  add column if not exists identity_hash text,
  add column if not exists first_initial text,
  add column if not exists last_initial text;

update public.students
set
  university_id_hash = case university_id
    when '123456789' then 'f793242b96ac6cde726f99e1fa68fd98895b1eba564b580ecf48be4939f1d0ea'
    when '987654321' then '922ef86cac0252d696c6c98be655601146f4ed5020dc8451e9a04d9806044d33'
    else university_id_hash
  end,
  identity_hash = case university_id
    when '123456789' then 'de62686ef17379401ff1a5a211396361b8906ab9bdec0ea0af70248caf538326'
    when '987654321' then '2b96c1db363d3d35403f4abb7fef059848970e5f461a578ac2d267a31bb328aa'
    else identity_hash
  end,
  first_initial = upper(left(trim(first_name), 1)),
  last_initial = upper(left(trim(last_name), 1));

alter table public.students
  drop constraint if exists students_university_id_key,
  drop constraint if exists students_university_id_not_blank,
  drop constraint if exists students_first_name_not_blank,
  drop constraint if exists students_last_name_not_blank;

alter table public.students
  alter column university_id_hash set not null,
  alter column identity_hash set not null,
  alter column first_initial set not null,
  alter column last_initial set not null;

alter table public.students
  add constraint students_university_id_hash_key unique (university_id_hash),
  add constraint students_identity_hash_key unique (identity_hash),
  add constraint students_university_id_hash_format check (university_id_hash ~ '^[0-9a-f]{64}$'),
  add constraint students_identity_hash_format check (identity_hash ~ '^[0-9a-f]{64}$'),
  add constraint students_first_initial_not_blank check (length(trim(first_initial)) > 0),
  add constraint students_last_initial_not_blank check (length(trim(last_initial)) > 0);

alter table public.students
  drop column if exists university_id,
  drop column if exists first_name,
  drop column if exists last_name;

alter table public.qa_events
  add constraint qa_events_student_id_fkey
  foreign key (student_id) references public.students(id);

create or replace view public.engagement_summary as
select
  s.university_id_hash as student_key,
  s.first_initial,
  s.last_initial,
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
group by s.id, s.university_id_hash, s.first_initial, s.last_initial, s.section, e.lecture_id;

comment on table public.students is 'Pseudonymous course roster. Raw university IDs and names are intentionally not stored.';
