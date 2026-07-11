# Engagement Tracking Setup

This repository includes a Supabase scaffold for roster-linked Q&A engagement tracking.

The current public book still works without Supabase. Engagement logging starts only after a deployed Supabase Edge Function URL is pasted into the Q&A widget in `docs/gradient_descent_2025.md`.

Project created for this course:

```text
Project ref: gpmprmejteppxxpxtlfk
Project URL: https://gpmprmejteppxxpxtlfk.supabase.co
Function URL: https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa
```

## Current Deployment Status

Done:

- Supabase project exists and is linked locally.
- Supabase CLI is installed in this repo.
- Edge Function secrets are set:
  - `QA_MODEL_ENDPOINT=https://text.pollinations.ai/openai`
  - `ALLOWED_ORIGIN=https://designinformaticslab.github.io`
- Edge Function `qa` is deployed.
- Function smoke test passes:
  - `OPTIONS /functions/v1/qa` returns `200`
  - empty `POST /functions/v1/qa` returns the expected validation error

Still needed before enabling the live book endpoint:

- Apply `supabase/migrations/20260711162000_engagement_tracking.sql`.
- Import the roster into `public.students`.
- Test one real roster student.
- Paste the function URL into `docs/gradient_descent_2025.md`.

## What Is Already Implemented

- Database migration for:
  - `public.students`: course roster
  - `public.lectures`: lecture/page registry
  - `public.qa_events`: one logged Q&A interaction per row
  - `public.engagement_summary`: roster-level reporting view
- Supabase Edge Function:
  - validates first name + last name + university ID against the roster
  - calls the AI endpoint to answer the question
  - calls the AI endpoint again to score question quality
  - stores timing and quality metrics
  - does not store generated answers by default
- Static book widget:
  - collects identity fields only after a Supabase endpoint is configured
  - falls back to the current direct model call while the endpoint is blank
  - renders LaTeX through MathJax

## Student Data Stored

The database stores:

- university ID
- first name
- last name
- optional section
- lecture ID
- student question text
- model name
- server-side answer time
- server-side quality-scoring time
- total server time
- AI question-quality score and rubric fields

Generated answers are intentionally not stored.

## TODO Checklist

### 1. Create Supabase Project

Done.

The project ref is `gpmprmejteppxxpxtlfk`.

### 2. Install Supabase CLI

Done. The CLI is installed as a local project dev dependency:

```bash
npm install supabase --save-dev
```

Use it through npm scripts or `npx`:

```bash
npx supabase --version
npm run supabase:status
```

### 3. Log In And Link The Project

Done enough for API operations. The CLI can list the linked project and deploy functions.

Applying migrations through the CLI still needs the remote Postgres connection to succeed. If `db push` hangs at `Initialising login role...`, use the SQL-editor fallback in step 4 or relink with the database password.

Option A: run locally in your terminal:

```bash
npx supabase login
npm run supabase:link
```

If `supabase link` asks for a database password, use the database password you set when creating the Supabase project. You can also pass it explicitly:

```bash
npx supabase link --project-ref gpmprmejteppxxpxtlfk --password '<database-password>'
```

Token fallback:

1. Go to https://supabase.com/dashboard/account/tokens.
2. Create a new access token.
3. Use it locally, or send it to me only if you are comfortable sharing it in this environment.
4. If you also want me to run `db push`, I will need the project database password as well. I do not recommend sharing long-lived passwords in chat; running the link/db commands locally is safer.
5. With the token, I can run:

   ```bash
   npx supabase login --token <token>
   npx supabase link --project-ref gpmprmejteppxxpxtlfk --password '<database-password>'
   ```

Do not commit the token. Revoke it after setup if you share it.

### 4. Apply The Database Migration

The migration is already in:

```text
supabase/migrations/20260711162000_engagement_tracking.sql
```

Apply it:

```bash
npm run supabase:db:push
```

Expected result: Supabase creates `students`, `lectures`, `qa_events`, and `engagement_summary`.

If `db push` hangs, use this dashboard fallback:

1. Open https://supabase.com/dashboard/project/gpmprmejteppxxpxtlfk/sql/new.
2. Open `supabase/migrations/20260711162000_engagement_tracking.sql` locally.
3. Paste the full SQL into the SQL editor.
4. Click `Run`.
5. Confirm the tables/views exist in Table Editor.

### 5. Import The Roster

Use this CSV column format:

```csv
university_id,first_name,last_name,section,active
123456789,Ada,Lovelace,MAE598,true
987654321,Grace,Hopper,MAE598,true
```

Recommended beginner path:

1. Open Supabase dashboard.
2. Go to Table Editor.
3. Open `students`.
4. Import CSV.
5. Make sure `university_id`, `first_name`, and `last_name` are populated.

Roster matching is case-insensitive for names, but university ID must match exactly after trimming whitespace.

### 6. Set Edge Function Secrets

Use the CLI:

```bash
npx supabase secrets set QA_MODEL_ENDPOINT=https://text.pollinations.ai/openai
npx supabase secrets set ALLOWED_ORIGIN=https://designinformaticslab.github.io
```

If you later switch to a paid model provider, add:

```bash
npx supabase secrets set QA_MODEL_API_KEY=<provider-api-key>
```

Do not commit API keys, service-role keys, or `.env` files.

### 7. Deploy The Edge Function

Deploy only the Q&A function:

```bash
npm run supabase:functions:deploy
```

The deployed function URL will be:

```text
https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa
```

### 8. Connect The Book To Supabase

In `docs/gradient_descent_2025.md`, find:

```html
data-engagement-endpoint=""
```

Replace it with:

```html
data-engagement-endpoint="https://gpmprmejteppxxpxtlfk.supabase.co/functions/v1/qa"
```

After this is committed and deployed, the Q&A widget will show first name, last name, and university ID fields.

### 9. Test With One Roster Student

Use a real roster row or temporary test row.

1. Open the live lecture page.
2. Enter the matching first name, last name, and university ID.
3. Ask a short question.
4. Confirm the answer appears.
5. In Supabase Table Editor, open `qa_events`.
6. Confirm one new row appears.
7. Open `engagement_summary`.
8. Confirm the student has `question_count = 1`.

### 10. Push The Book Deployment

After the endpoint is configured and tested locally:

```bash
git add docs/gradient_descent_2025.md
git commit -m "Enable Q&A engagement logging"
git push origin main
```

GitHub Pages will rebuild the book.

## What I Can Do After You Create Supabase

Once CLI login works, I can:

- run `npx supabase link`
- run `npx supabase db push`
- deploy the `qa` Edge Function
- paste the function URL into the book
- build/test the book
- push the final deployment

I should not receive or commit private keys. If a model provider needs an API key, you should set it in Supabase secrets directly or approve a local command that reads it from your terminal.

## Metrics

The initial engagement metrics are:

- `question_count`: number of questions asked per student per lecture
- `total_answer_elapsed_ms`: server-side time spent generating answers
- `avg_quality_score`: average AI-evaluated quality score from 0 to 3
- `quality_relevance`: how tied the question is to the lecture
- `quality_specificity`: how precise the question is
- `quality_math_depth`: whether the question engages the mathematical content
- `quality_effort`: whether the question shows thoughtful engagement

Use `public.engagement_summary` for reporting.

## Implementation Challenges

- **Roster verification is not full authentication.** A student who knows another student’s university ID could impersonate them. Supabase Auth or university SSO would be the stronger version.
- **This is educational data.** University ID plus question text should be treated as protected course data. Restrict dashboard access and keep RLS enabled.
- **The Edge Function is public.** Roster validation blocks unknown students, but it is not rate limiting. Add rate limiting if the site gets heavy use.
- **AI quality scoring is subjective.** Use it as an engagement signal, not as a high-stakes grade.
- **Free-tier reliability.** Supabase free-tier projects can be enough for a class demo, but check the project before lecture.
- **Model stability.** The default endpoint is Pollinations. For production reliability, a controlled model provider with an API key is better.

## Official Docs

- Supabase local development and CLI: https://supabase.com/docs/guides/local-development
- Supabase database migrations: https://supabase.com/docs/guides/deployment/database-migrations
- Supabase Edge Function deployment: https://supabase.com/docs/guides/functions/deploy
- Supabase Edge Function secrets: https://supabase.com/docs/guides/functions/secrets
