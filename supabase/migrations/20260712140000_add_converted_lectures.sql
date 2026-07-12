insert into public.lectures (id, title)
values
  ('optimization_prelim_2025', 'Preliminaries: calculus, convexity, and Taylor expansion'),
  ('optimization_gradient_descent_advanced_2025', 'Gradient-based methods for unconstrained optimization, part 3')
on conflict (id) do update set title = excluded.title;
