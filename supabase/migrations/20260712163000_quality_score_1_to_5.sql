alter table public.qa_events
  drop constraint if exists qa_events_quality_score_range;

alter table public.qa_events
  add constraint qa_events_quality_score_range
  check (quality_score is null or quality_score between 1 and 5);
