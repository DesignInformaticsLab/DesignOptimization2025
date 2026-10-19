insert into public.lectures (id, title)
values
  ('constrained_optimization_pt1_2026', 'Constrained optimization, part 1: KKT conditions')
on conflict (id) do update set title = excluded.title;
