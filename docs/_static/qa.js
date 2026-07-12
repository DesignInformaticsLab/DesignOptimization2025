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
      page_excerpt: pageText.slice(0, 6000),
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

  function readIdentity(widget) {
    return {
      firstName: widget.querySelector(".qa-first-name")?.value.trim() || "",
      lastName: widget.querySelector(".qa-last-name")?.value.trim() || "",
      universityId: widget.querySelector(".qa-university-id")?.value.trim() || "",
    };
  }

  function saveIdentity(identity) {
    window.localStorage?.setItem("designopt.qa.identity", JSON.stringify(identity));
  }

  function restoreIdentity(widget) {
    const saved = window.localStorage?.getItem("designopt.qa.identity");
    if (!saved) {
      return;
    }
    try {
      const identity = JSON.parse(saved);
      widget.querySelector(".qa-first-name").value = identity.firstName || "";
      widget.querySelector(".qa-last-name").value = identity.lastName || "";
      widget.querySelector(".qa-university-id").value = identity.universityId || "";
    } catch {
      window.localStorage?.removeItem("designopt.qa.identity");
    }
  }

  async function askDirectModel(model, context, question) {
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
    return (
      data?.choices?.[0]?.message?.content ||
      data?.choices?.[0]?.text ||
      JSON.stringify(data, null, 2)
    ).trim();
  }

  async function ask(widget) {
    const question = widget.querySelector(".qa-question").value.trim();
    const model = widget.querySelector(".qa-model").value.trim() || "openai";
    const endpoint = widget.getAttribute("data-engagement-endpoint")?.trim();
    const lectureId = widget.getAttribute("data-lecture-id") || document.location.pathname;
    const identity = readIdentity(widget);
    const button = widget.querySelector(".qa-submit");
    const startedAt = performance.now();
    if (endpoint && (!identity.firstName || !identity.lastName || !identity.universityId)) {
      setAnswer(widget, "Enter your first name, last name, and university ID first.");
      return;
    }
    if (!question) {
      setAnswer(widget, "Enter a question first.");
      return;
    }

    button.disabled = true;
    setAnswer(widget, "Loading context and asking the model...");

    try {
      const context = await loadContext(widget);
      if (endpoint) {
        saveIdentity(identity);
        const response = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...identity,
            lectureId,
            question,
            model,
            context,
          }),
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data?.error || `API request failed with HTTP ${response.status}`);
        }
        const elapsedSeconds = ((data?.metrics?.totalElapsedMs || 0) / 1000).toFixed(1);
        setAnswer(widget, `${data.answer.trim()}\n\nAnswered in ${elapsedSeconds}s.`);
      } else {
        const answer = await askDirectModel(model, context, question);
        const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
        setAnswer(widget, `${answer}\n\nAnswered in ${elapsedSeconds}s. Engagement logging is not configured yet.`);
      }
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
      const endpoint = widget.getAttribute("data-engagement-endpoint")?.trim();
      if (endpoint) {
        restoreIdentity(widget);
      } else {
        widget.classList.add("qa-no-engagement");
      }
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
