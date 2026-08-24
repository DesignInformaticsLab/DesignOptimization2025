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
    const contextStr = JSON.stringify(context, null, 2);
    const trimmedContext = contextStr.length > 3000 ? contextStr.slice(0, 3000) + "..." : contextStr;
    const payload = {
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
            "Context:\n" + trimmedContext +
            "\n\nStudent question:\n" + question,
        },
      ],
      model: model || "openai-fast",
      temperature: 0.2,
      max_tokens: 450,
    };

    const body = JSON.stringify(payload);
    for (let attempt = 0; attempt < 2; attempt++) {
      const response = await fetch("https://text.pollinations.ai/openai", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body,
      });
      if (response.ok) {
        const data = await response.json();
        return (
          data?.choices?.[0]?.message?.content ||
          data?.choices?.[0]?.text ||
          JSON.stringify(data, null, 2)
        ).trim();
      }
      if (response.status === 402 && attempt === 0) {
        await new Promise((r) => setTimeout(r, 16000));
        continue;
      }
      throw new Error(`API request failed with HTTP ${response.status}`);
    }
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
      let answer = "";
      let source = "";
      if (endpoint) {
        saveIdentity(identity);
        try {
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
            throw new Error(data?.error || `HTTP ${response.status}`);
          }
          const elapsedSeconds = ((data?.metrics?.totalElapsedMs || 0) / 1000).toFixed(1);
          answer = data.answer.trim();
          source = `Answered in ${elapsedSeconds}s.`;
        } catch {
          answer = await askDirectModel(model, context, question);
          const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
          source = `Answered in ${elapsedSeconds}s (direct, engagement not logged).`;
        }
      } else {
        answer = await askDirectModel(model, context, question);
        const elapsedSeconds = ((performance.now() - startedAt) / 1000).toFixed(1);
        source = `Answered in ${elapsedSeconds}s.`;
      }
      setAnswer(widget, `${answer}\n\n${source}`);
    } catch (error) {
      setAnswer(
        widget,
        "The Q&A request failed. Check network access or try again.\n\n" +
          error.message
      );
    } finally {
      button.disabled = false;
    }
  }

  function init() {
    document.querySelectorAll(".qa-widget").forEach((widget) => {
      const endpoint = widget.getAttribute("data-engagement-endpoint")?.trim();
      if (!endpoint) {
        widget.classList.add("qa-no-engagement");
      }
      widget.querySelector(".qa-submit")?.addEventListener("click", () => ask(widget));
      widget.querySelector(".qa-question")?.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
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
