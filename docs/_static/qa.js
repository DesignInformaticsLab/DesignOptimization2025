(function () {
  async function loadContext(widget) {
    const path = widget.getAttribute("data-context-file");
    const pageText = document.querySelector("main")?.innerText || document.body.innerText;
    let context = {};
    if (path) {
      const response = await fetch(path);
      if (response.ok) {
        context = await response.json();
      }
    }
    return {
      page_excerpt: pageText.slice(0, 12000),
      metadata: context,
    };
  }

  function setAnswer(widget, text) {
    const answer = widget.querySelector(".qa-answer");
    answer.textContent = text;
    if (window.MathJax?.typesetPromise) {
      window.MathJax.typesetPromise([answer]).catch(() => {});
    }
  }

  async function ask(widget) {
    const question = widget.querySelector(".qa-question").value.trim();
    const model = widget.querySelector(".qa-model").value.trim() || "openai";
    const button = widget.querySelector(".qa-submit");
    const startedAt = performance.now();
    if (!question) {
      setAnswer(widget, "Enter a question first.");
      return;
    }

    button.disabled = true;
    setAnswer(widget, "Loading context and asking the model...");

    try {
      const context = await loadContext(widget);
      const payload = {
        model,
        temperature: 0.2,
        max_tokens: 700,
        messages: [
          {
            role: "system",
            content:
              "You are a concise teaching assistant for a graduate design optimization class. " +
              "Use the supplied lecture notes and notebook context. If the answer is not supported by context, say what is missing. " +
              "When writing equations, use LaTeX delimiters \\(...\\) for inline math and \\[...\\] for display math so MathJax can render them.",
          },
          {
            role: "user",
            content:
              "Context JSON:\n" +
              JSON.stringify(context, null, 2) +
              "\n\nStudent question:\n" +
              question,
          },
        ],
      };

      const response = await fetch("https://text.pollinations.ai/openai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!response.ok) {
        throw new Error(`API request failed with HTTP ${response.status}`);
      }

      const data = await response.json();
      const answer =
        data?.choices?.[0]?.message?.content ||
        data?.choices?.[0]?.text ||
        JSON.stringify(data, null, 2);
      const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
      setAnswer(widget, `${answer.trim()}\n\nAnswered in ${elapsedSeconds}s.`);
    } catch (error) {
      setAnswer(
        widget,
        "The Q&A request failed. Check network access, browser CORS policy, or the configured model.\n\n" +
          error.message
      );
    } finally {
      button.disabled = false;
    }
  }

  function init() {
    document.querySelectorAll(".qa-widget").forEach((widget) => {
      widget.querySelector(".qa-submit")?.addEventListener("click", () => ask(widget));
      widget.querySelector(".qa-question")?.addEventListener("keydown", (event) => {
        if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
          ask(widget);
        }
      });
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
