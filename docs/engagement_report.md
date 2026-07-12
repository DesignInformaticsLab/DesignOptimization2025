# Q&A engagement report

This instructor-facing report shows recorded Q&A activity grouped by student.
Questions are ranked by the current quality score.

Supabase stores pseudonymous student keys, initials, section, and Q&A activity.
It does not store raw university IDs or full names. To view actual names and
IDs on this page, select the original roster CSV and enter the private roster
hash salt. That join happens locally in your browser; the roster file and salt
are not uploaded by this page.

```{raw} html
<div class="engagement-report" data-report-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/engagement-report">
  <div class="engagement-controls">
    <label>
      Instructor report token
      <input class="engagement-token" type="password" autocomplete="off" />
    </label>
    <label>
      Local roster CSV
      <input class="engagement-roster" type="file" accept=".csv,text/csv" />
    </label>
    <label>
      Roster hash salt
      <input class="engagement-hash-salt" type="password" autocomplete="off" />
    </label>
    <button class="engagement-load" type="button">Load report</button>
  </div>
  <p class="engagement-note">The report token is sent to the Supabase Edge Function as a bearer token. The optional roster CSV and hash salt stay in this browser tab and are only used to display actual names and IDs locally.</p>
  <div class="engagement-summary" aria-live="polite"></div>
  <div class="engagement-table"></div>
</div>
```

## Current quality metric

The current quality score is a single AI-evaluated metric. The evaluator prompt
is:

> As an experienced educator in optimization at a top-tier institute, from 1-5
> where 5 is the best, how deep do you think the student understands the
> materials on current page?

The score is an integer from 1 to 5:

- `1`: very shallow or confused understanding
- `3`: reasonable basic understanding
- `5`: unusually deep, precise, and conceptually connected understanding

The Q&A answer is returned before this evaluation runs. The evaluation is
scheduled in the background and updates the report row once complete. This
keeps student-facing answer latency low while still recording an AI quality
metric.

This score is an engagement signal, not a high-stakes grade.
