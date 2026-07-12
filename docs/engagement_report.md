# Q&A engagement report

This instructor-facing report shows recorded Q&A activity grouped by student.
Questions are ranked by the current quality score.

```{raw} html
<div class="engagement-report" data-report-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/engagement-report">
  <div class="engagement-controls">
    <label>
      Instructor report token
      <input class="engagement-token" type="password" autocomplete="off" />
    </label>
    <button class="engagement-load" type="button">Load report</button>
  </div>
  <p class="engagement-note">The report token is sent to the Supabase Edge Function as a bearer token. It is not stored by this page.</p>
  <div class="engagement-summary" aria-live="polite"></div>
  <div class="engagement-table"></div>
</div>
```

## Current quality rubric

The current production setting does **not** use a second AI call to score
question quality, because that made the Q&A response much slower. Instead it
uses a fast deterministic rubric with four components, each scored from 1 to 3.
The final score is the average.

| Component | Rule |
|---|---|
| Relevance | 3 if the question contains an optimization/math term from the rubric list; otherwise 1 |
| Specificity | 3 for 12+ words, 2 for 6-11 words, 1 for fewer than 6 words |
| Math depth | 3 when the question asks why/how/derive/prove/compare and includes a math term; 2 when it includes a math term; otherwise 1 |
| Effort | 3 when it has a question word and 8+ words, 2 for 4+ words, 1 otherwise |

Terms currently counted as optimization/math terms:

```text
gradient, hessian, convex, armijo, line search, newton, bfgs, descent,
convergence, condition, objective, derivative, minimizer, optimization
```

This score is an engagement signal, not a high-stakes grade.
