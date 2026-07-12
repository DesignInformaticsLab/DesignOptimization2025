(function () {
  function formatMs(ms) {
    if (ms == null) {
      return "-";
    }
    return `${(ms / 1000).toFixed(1)}s`;
  }

  function formatScore(score) {
    if (score == null || Number.isNaN(Number(score))) {
      return "-";
    }
    return Number(score).toFixed(2);
  }

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function normalizeIdentityPart(value) {
    return String(value ?? "").trim().replace(/\s+/g, " ").toLowerCase();
  }

  async function sha256Hex(text) {
    const bytes = new TextEncoder().encode(text);
    const digest = await crypto.subtle.digest("SHA-256", bytes);
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }

  function parseCsvLine(line) {
    const values = [];
    let value = "";
    let quoted = false;

    for (let i = 0; i < line.length; i += 1) {
      const char = line[i];
      const next = line[i + 1];
      if (quoted && char === '"' && next === '"') {
        value += '"';
        i += 1;
      } else if (char === '"') {
        quoted = !quoted;
      } else if (!quoted && char === ",") {
        values.push(value);
        value = "";
      } else {
        value += char;
      }
    }
    values.push(value);
    return values;
  }

  async function rosterMapFromFile(file, salt) {
    if (!file) {
      return new Map();
    }
    if (!salt) {
      throw new Error("Enter the roster hash salt to join the local roster.");
    }

    const text = await file.text();
    const lines = text.split(/\r?\n/).filter((line) => line.trim());
    const header = parseCsvLine(lines[0]).map((column) => column.trim());
    const index = Object.fromEntries(header.map((column, i) => [column, i]));
    for (const required of ["university_id", "first_name", "last_name"]) {
      if (!(required in index)) {
        throw new Error(`Roster CSV is missing ${required}.`);
      }
    }

    const roster = new Map();
    for (const line of lines.slice(1)) {
      const row = parseCsvLine(line);
      const universityId = normalizeIdentityPart(row[index.university_id]);
      const firstName = String(row[index.first_name] ?? "").trim();
      const lastName = String(row[index.last_name] ?? "").trim();
      if (!universityId || !firstName || !lastName) {
        continue;
      }
      const studentKey = await sha256Hex(`${salt}:student_id:${universityId}`);
      roster.set(studentKey, {
        university_id: row[index.university_id],
        first_name: firstName,
        last_name: lastName,
        section: row[index.section],
      });
    }
    return roster;
  }

  function studentDisplay(student, roster) {
    const matched = roster.get(student.student_key);
    if (matched) {
      return {
        name: `${matched.first_name} ${matched.last_name}`,
        id: matched.university_id,
        section: matched.section || student.section,
      };
    }
    return {
      name: `${student.first_initial || "?"}. ${student.last_initial || "?"}.`,
      id: `${String(student.student_key || "unknown").slice(0, 12)}...`,
      section: student.section,
    };
  }

  function renderReport(container, data, roster) {
    const summary = container.querySelector(".engagement-summary");
    const table = container.querySelector(".engagement-table");
    const students = data.students || [];
    const questionCount = students.reduce((sum, student) => sum + student.question_count, 0);
    const matchedCount = students.filter((student) => roster.has(student.student_key)).length;
    const rosterText = roster.size ? ` ${matchedCount} matched to the local roster.` : " Showing pseudonymous records.";
    summary.textContent = `${students.length} students, ${questionCount} questions. Generated ${new Date(data.generated_at).toLocaleString()}.${rosterText}`;

    if (!students.length) {
      table.innerHTML = "<p>No Q&A records found.</p>";
      return;
    }

    table.innerHTML = students.map((student) => {
      const display = studentDisplay(student, roster);
      return `
      <section class="engagement-student">
        <h2>${escapeHtml(display.name)}</h2>
        <p class="engagement-meta">
          ID ${escapeHtml(display.id)}
          ${display.section ? ` · ${escapeHtml(display.section)}` : ""}
          · ${student.question_count} questions
          · avg quality ${formatScore(student.avg_quality_score)}
          · answer time ${formatMs(student.total_answer_elapsed_ms)}
        </p>
        <table>
          <thead>
            <tr>
              <th>Quality</th>
              <th>Lecture</th>
              <th>Question</th>
              <th>AI evaluation</th>
              <th>Time</th>
              <th>Asked</th>
            </tr>
          </thead>
          <tbody>
            ${student.questions.map((question) => `
              <tr>
                <td>${formatScore(question.quality_score)}</td>
                <td>${escapeHtml(question.lecture_id)}</td>
                <td>${escapeHtml(question.question)}</td>
                <td>${escapeHtml(question.quality_rationale || "Evaluation pending")}</td>
                <td>${formatMs(question.total_elapsed_ms)}</td>
                <td>${new Date(question.created_at).toLocaleString()}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </section>
    `;
    }).join("");
  }

  async function loadReport(container) {
    const endpoint = container.getAttribute("data-report-endpoint");
    const token = container.querySelector(".engagement-token").value.trim();
    const rosterFile = container.querySelector(".engagement-roster")?.files?.[0];
    const hashSalt = container.querySelector(".engagement-hash-salt")?.value.trim();
    const summary = container.querySelector(".engagement-summary");
    if (!token) {
      summary.textContent = "Enter the instructor report token first.";
      return;
    }

    summary.textContent = "Loading engagement report...";
    const roster = await rosterMapFromFile(rosterFile, hashSalt);
    const response = await fetch(endpoint, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Report request failed with HTTP ${response.status}`);
    }
    renderReport(container, data, roster);
  }

  function init() {
    document.querySelectorAll(".engagement-report").forEach((container) => {
      container.querySelector(".engagement-load")?.addEventListener("click", () => {
        loadReport(container).catch((error) => {
          container.querySelector(".engagement-summary").textContent = error.message;
        });
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
