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

  function renderReport(container, data) {
    const summary = container.querySelector(".engagement-summary");
    const table = container.querySelector(".engagement-table");
    const students = data.students || [];
    const questionCount = students.reduce((sum, student) => sum + student.question_count, 0);
    summary.textContent = `${students.length} students, ${questionCount} questions. Generated ${new Date(data.generated_at).toLocaleString()}.`;

    if (!students.length) {
      table.innerHTML = "<p>No Q&A records found.</p>";
      return;
    }

    table.innerHTML = students.map((student) => `
      <section class="engagement-student">
        <h2>${escapeHtml(student.first_name)} ${escapeHtml(student.last_name)}</h2>
        <p class="engagement-meta">
          ID ${escapeHtml(student.university_id)}
          ${student.section ? ` · ${escapeHtml(student.section)}` : ""}
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
              <th>Rubric</th>
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
                <td>
                  R${question.quality_relevance ?? "-"}
                  S${question.quality_specificity ?? "-"}
                  M${question.quality_math_depth ?? "-"}
                  E${question.quality_effort ?? "-"}
                </td>
                <td>${formatMs(question.total_elapsed_ms)}</td>
                <td>${new Date(question.created_at).toLocaleString()}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      </section>
    `).join("");
  }

  async function loadReport(container) {
    const endpoint = container.getAttribute("data-report-endpoint");
    const token = container.querySelector(".engagement-token").value.trim();
    const summary = container.querySelector(".engagement-summary");
    if (!token) {
      summary.textContent = "Enter the instructor report token first.";
      return;
    }

    summary.textContent = "Loading engagement report...";
    const response = await fetch(endpoint, {
      headers: {
        Authorization: `Bearer ${token}`,
      },
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || `Report request failed with HTTP ${response.status}`);
    }
    renderReport(container, data);
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
