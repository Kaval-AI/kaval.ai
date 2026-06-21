/*
 * Kaval.AI docs playground.
 *
 * Adds a "Run in browser" button to Python code blocks the author marked
 * `run-in-browser` (opt-in). Clicking it opens a side-panel (a right-hand
 * drawer on desktop, a full-width vertical split on mobile) with an editable
 * copy of the snippet and an output pane, and runs it through Pyodide with the
 * kavalai wheel installed. A WebLLM bridge lets "browser/..." models run too.
 *
 * NOTE: the docs must be served over http(s); opening the built HTML via a
 * file:// path stops the browser from fetching the kavalai wheel.
 *
 * Pyodide is booted lazily on the first run so page loads stay fast. The booted
 * interpreter, the installed wheel and the CodeMirror assets are all cached and
 * reused across every snippet on the site.
 *
 * Based on python-playground.html, productised for the docs.
 */
(function () {
  "use strict";

  // -- Configuration --------------------------------------------------------
  // window.KAVAL_PLAYGROUND_CONFIG is emitted by the Sphinx extension
  // (docs/_ext/kaval_playground.py) and loaded before this file.
  var CONFIG = window.KAVAL_PLAYGROUND_CONFIG || {};

  // Resolve the _static/pyodide/ base from this script's own URL so that the
  // wheel and asset URLs work no matter how deep the current page is
  // (/index.html vs /api/llm_clients.html).
  var SELF_SRC = (document.currentScript && document.currentScript.src) || "";
  var STATIC_BASE = SELF_SRC.replace(/playground\.js(\?.*)?$/, "");

  var PYODIDE_URL =
    CONFIG.pyodideUrl ||
    "https://cdn.jsdelivr.net/pyodide/v314.0.0/full/pyodide.js";
  var PYODIDE_INDEX = PYODIDE_URL.replace(/pyodide\.js(\?.*)?$/, "");
  var WHEEL_URL = CONFIG.wheelName ? STATIC_BASE + CONFIG.wheelName : null;
  var CM_BASE = "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/";
  var WEBLLM_URL = "https://esm.run/@mlc-ai/web-llm";

  var LS_OPENAI = "kaval-pg-openai-key";
  var LS_GEMINI = "kaval-pg-gemini-key";
  var LS_MODEL = "kaval-pg-browser-model";

  // In-browser chat models offered in the drawer's model picker. q4f32 builds
  // run on GPUs without FP16 shaders (e.g. GTX 10xx); q4f16 needs an FP16 GPU.
  // The chosen id is exposed to Python as the KAVAL_BROWSER_MODEL global so
  // "browser/..." examples don't hardcode it. (Mirrors python-playground.html.)
  var BROWSER_MODELS = [
    ["Llama-3.2-1B-Instruct-q4f32_1-MLC", "Llama-3.2-1B · q4f32 (≈1.1 GB)"],
    ["Llama-3.2-3B-Instruct-q4f32_1-MLC", "Llama-3.2-3B · q4f32 (≈2.9 GB)"],
    ["Qwen2.5-1.5B-Instruct-q4f32_1-MLC", "Qwen2.5-1.5B · q4f32 (≈1.6 GB)"],
    ["Qwen2.5-0.5B-Instruct-q4f32_1-MLC", "Qwen2.5-0.5B · q4f32 (≈0.6 GB)"],
    ["Llama-3.2-1B-Instruct-q4f16_1-MLC", "Llama-3.2-1B · q4f16 (FP16 GPU)"],
    ["Llama-3.2-3B-Instruct-q4f16_1-MLC", "Llama-3.2-3B · q4f16 (FP16 GPU)"],
  ];
  // A small, full-precision (q0f32) embedding model — runs without an FP16 GPU.
  var DEFAULT_EMBED_MODEL = "snowflake-arctic-embed-s-q0f32-MLC-b4";

  // -- Tiny DOM helpers -----------------------------------------------------
  function loadScript(src) {
    return new Promise(function (resolve, reject) {
      var s = document.createElement("script");
      s.src = src;
      s.onload = resolve;
      s.onerror = function () {
        reject(new Error("Failed to load script: " + src));
      };
      document.head.appendChild(s);
    });
  }

  function loadCss(href) {
    var link = document.createElement("link");
    link.rel = "stylesheet";
    link.href = href;
    document.head.appendChild(link);
  }

  // -- Lazy singletons ------------------------------------------------------
  var pyodidePromise = null; // resolves to the booted pyodide instance
  var cmPromise = null; // resolves once CodeMirror is loaded
  var panel = null; // the drawer DOM (built once, reused)
  var editor = null; // the CodeMirror instance
  var originalCode = ""; // snippet the drawer was opened with (for Reset)

  // -- Output pane helpers (operate on the single shared drawer) ------------
  function outputEl() {
    return panel && panel.querySelector('[data-pg="output"]');
  }

  function appendOutput(text, kind) {
    var out = outputEl();
    if (!out) return;
    var span = document.createElement("span");
    if (kind) span.className = "kaval-pg-" + kind;
    span.textContent = text;
    out.appendChild(span);
    out.scrollTop = out.scrollHeight;
  }

  function clearOutput() {
    var out = outputEl();
    if (out) out.textContent = "";
  }

  function setStatus(text) {
    var s = panel && panel.querySelector('[data-pg="status"]');
    if (s) s.textContent = text;
  }

  // -- Pyodide bootstrap ----------------------------------------------------
  function ensurePyodide() {
    if (!pyodidePromise) pyodidePromise = bootPyodide();
    return pyodidePromise;
  }

  async function bootPyodide() {
    setStatus("Loading Pyodide…");
    await loadScript(PYODIDE_URL);
    var pyodide = await loadPyodide({ indexURL: PYODIDE_INDEX });

    pyodide.setStdout({ batched: function (s) { appendOutput(s + "\n"); } });
    pyodide.setStderr({
      batched: function (s) { appendOutput(s + "\n", "stderr"); },
    });

    setStatus("Loading micropip…");
    await pyodide.loadPackage("micropip");
    var micropip = pyodide.pyimport("micropip");

    // Route httpx/requests/urllib through the browser's fetch so the LLM
    // clients have a chance of reaching providers (subject to provider CORS).
    setStatus("Installing pyodide-http…");
    await micropip.install("pyodide-http");
    await pyodide.runPythonAsync(
      "import pyodide_http; pyodide_http.patch_all()"
    );

    if (WHEEL_URL) {
      setStatus("Installing kavalai (downloading dependencies)…");
      await micropip.install(WHEEL_URL);
    } else {
      appendOutput(
        "Note: the kavalai wheel was not bundled with these docs, so only " +
          "plain Python (Pyodide stdlib) is available. Build it with " +
          "`uv build --wheel` and rebuild the docs to enable kavalai.\n",
        "meta"
      );
    }

    setStatus("Ready");
    return pyodide;
  }

  // -- In-browser LLM/embeddings engine (WebLLM) ----------------------------
  // Exposes window.kavalBrowserLLM.{chat,embed} so kavalai's "browser/..."
  // clients run fully client-side over WebGPU — no API key, no server, no CORS.
  // WebLLM and the model are downloaded lazily on first use and cached by the
  // browser. This is the docs port of python-playground.html's bridge.
  var webllmModulePromise = null;
  var enginePromise = null;
  var loadedModel = null;

  function loadWebLLM() {
    if (!webllmModulePromise) webllmModulePromise = import(WEBLLM_URL);
    return webllmModulePromise;
  }

  async function getWebLLMEngine(modelId) {
    if (!navigator.gpu) {
      throw new Error(
        "WebGPU is not available in this browser, so 'browser/...' models " +
          "cannot run. Use a recent Chrome/Edge (or Firefox with " +
          "dom.webgpu.enabled). Verify at chrome://gpu."
      );
    }
    // Reuse the engine while the model is unchanged; rebuild it on a switch
    // (e.g. moving from a chat model to an embedding model).
    if (enginePromise && loadedModel === modelId) return enginePromise;
    loadedModel = modelId;
    var webllm = await loadWebLLM();
    enginePromise = webllm.CreateMLCEngine(modelId, {
      initProgressCallback: function (report) {
        setStatus(report && report.text ? report.text : "Loading model…");
      },
    });
    return enginePromise;
  }

  function installBrowserLLMBridge() {
    if (window.kavalBrowserLLM) return;
    window.kavalBrowserLLM = {
      // {model, messages, temperature?, top_p?, response_format?} -> {content, usage} | {error}
      chat: async function (requestJson) {
        try {
          var req =
            typeof requestJson === "string" ? JSON.parse(requestJson) : requestJson;
          var engine = await getWebLLMEngine(req.model);
          var opts = { messages: req.messages, stream: false };
          if (req.temperature != null) opts.temperature = req.temperature;
          if (req.top_p != null) opts.top_p = req.top_p;
          if (req.max_tokens != null) opts.max_tokens = req.max_tokens;
          if (req.response_format) {
            var rf = { type: req.response_format.type || "json_object" };
            if (req.response_format.schema != null) {
              // WebLLM expects the JSON schema as a string.
              rf.schema =
                typeof req.response_format.schema === "string"
                  ? req.response_format.schema
                  : JSON.stringify(req.response_format.schema);
            }
            opts.response_format = rf;
          }
          var reply = await engine.chat.completions.create(opts);
          var choice = reply.choices && reply.choices[0];
          return JSON.stringify({
            content: (choice && choice.message && choice.message.content) || "",
            usage: reply.usage || {},
          });
        } catch (err) {
          return JSON.stringify({ error: String((err && err.message) || err) });
        }
      },
      // {model, input: [texts]} -> {embeddings, usage} | {error}
      embed: async function (requestJson) {
        try {
          var req =
            typeof requestJson === "string" ? JSON.parse(requestJson) : requestJson;
          var engine = await getWebLLMEngine(req.model);
          var reply = await engine.embeddings.create({
            input: req.input,
            model: req.model,
          });
          return JSON.stringify({
            embeddings: (reply.data || []).map(function (d) {
              return d.embedding;
            }),
            usage: reply.usage || {},
          });
        } catch (err) {
          return JSON.stringify({ error: String((err && err.message) || err) });
        }
      },
    };
  }

  // -- CodeMirror bootstrap -------------------------------------------------
  function ensureCodeMirror() {
    if (!cmPromise) {
      loadCss(CM_BASE + "codemirror.min.css");
      loadCss(CM_BASE + "theme/dracula.min.css");
      cmPromise = loadScript(CM_BASE + "codemirror.min.js").then(function () {
        return loadScript(CM_BASE + "mode/python/python.min.js");
      });
    }
    return cmPromise;
  }

  // -- The drawer -----------------------------------------------------------
  function buildPanel() {
    if (panel) return panel;
    panel = document.createElement("div");
    panel.className = "kaval-pg-panel";
    panel.setAttribute("role", "dialog");
    panel.setAttribute("aria-label", "Kaval.AI browser playground");
    panel.innerHTML =
      '<div class="kaval-pg-header">' +
      '  <span class="kaval-pg-title">🐍 Run in browser</span>' +
      '  <span class="kaval-pg-status" data-pg="status">Idle</span>' +
      '  <button class="kaval-pg-icon" data-pg="close" aria-label="Close playground" title="Close (Esc)">✕</button>' +
      "</div>" +
      '<details class="kaval-pg-keys">' +
      "  <summary>API keys — optional, stored only in this browser</summary>" +
      '  <label>OpenAI<input type="password" data-pg="openai" placeholder="sk-…" autocomplete="off" spellcheck="false"></label>' +
      '  <label>Gemini<input type="password" data-pg="gemini" placeholder="AIza…" autocomplete="off" spellcheck="false"></label>' +
      '  <p class="kaval-pg-keys-note">Injected as <code>OPENAI_API_KEY</code> / <code>GEMINI_API_KEY</code> before your code runs. Calls may still be blocked by provider CORS.</p>' +
      "</details>" +
      '<div class="kaval-pg-editor" data-pg="editor"></div>' +
      '<div class="kaval-pg-toolbar">' +
      '  <select class="kaval-pg-model" data-pg="model" title="In-browser model used by browser/... examples (exposed to Python as KAVAL_BROWSER_MODEL)">' +
      BROWSER_MODELS.map(function (m) {
        return '<option value="' + m[0] + '">' + m[1] + "</option>";
      }).join("") +
      "  </select>" +
      '  <button class="kaval-pg-run" data-pg="run">Run ▶</button>' +
      '  <button class="kaval-pg-ghost" data-pg="reset" title="Restore the original snippet">Reset</button>' +
      '  <button class="kaval-pg-ghost" data-pg="clear" title="Clear the output">Clear output</button>' +
      "</div>" +
      '<div class="kaval-pg-output-label">Output</div>' +
      '<pre class="kaval-pg-output" data-pg="output"></pre>';
    document.body.appendChild(panel);

    panel.querySelector('[data-pg="close"]').addEventListener("click", closePanel);
    panel.querySelector('[data-pg="run"]').addEventListener("click", runCode);
    panel.querySelector('[data-pg="clear"]').addEventListener("click", clearOutput);
    panel.querySelector('[data-pg="reset"]').addEventListener("click", function () {
      if (editor) editor.setValue(originalCode);
    });

    // API keys persist in localStorage (this browser only).
    wireKeyInput('[data-pg="openai"]', LS_OPENAI);
    wireKeyInput('[data-pg="gemini"]', LS_GEMINI);
    // Remember the chosen in-browser model across snippets and visits.
    wireKeyInput('[data-pg="model"]', LS_MODEL);

    return panel;
  }

  function wireKeyInput(selector, storageKey) {
    var input = panel.querySelector(selector);
    try {
      // Only restore when a value was saved, so a <select> keeps its default
      // first option instead of blanking out.
      var stored = window.localStorage.getItem(storageKey);
      if (stored) input.value = stored;
    } catch (e) {
      /* localStorage may be unavailable (private mode) — ignore. */
    }
    input.addEventListener("change", function () {
      try {
        window.localStorage.setItem(storageKey, input.value);
      } catch (e) {
        /* ignore */
      }
    });
  }

  async function ensureEditor() {
    buildPanel();
    if (editor) return editor;
    await ensureCodeMirror();
    editor = window.CodeMirror(panel.querySelector('[data-pg="editor"]'), {
      mode: "python",
      theme: "dracula",
      lineNumbers: true,
      indentUnit: 4,
      viewportMargin: Infinity,
      extraKeys: { "Ctrl-Enter": runCode, "Cmd-Enter": runCode },
    });
    return editor;
  }

  async function openPanel(code) {
    await ensureEditor();
    originalCode = code;
    editor.setValue(code);
    panel.classList.add("open");
    document.body.classList.add("kaval-pg-open");
    setStatus("Idle");
    // CodeMirror must re-measure once it is visible.
    setTimeout(function () {
      editor.refresh();
      editor.focus();
    }, 0);
  }

  function closePanel() {
    if (!panel) return;
    panel.classList.remove("open");
    document.body.classList.remove("kaval-pg-open");
  }

  // -- Running --------------------------------------------------------------
  async function runCode() {
    var runBtn = panel.querySelector('[data-pg="run"]');
    runBtn.disabled = true;
    clearOutput();

    // Browsers block fetch() of file:// URLs, so the kavalai wheel (and Pyodide
    // packages) can't be loaded when the docs are opened from disk. Point the
    // user at a local web server instead of failing with a cryptic error.
    if (window.location.protocol === "file:") {
      appendOutput(
        "This page was opened via a file:// path, so the browser won't let the " +
          "playground download the kavalai wheel.\n\nServe the built docs over " +
          "HTTP instead — from docs/_build/html run:\n\n" +
          "    python -m http.server\n\n" +
          "then browse to http://localhost:8000/ and try again.\n",
        "stderr"
      );
      setStatus("Open over http:// to run");
      runBtn.disabled = false;
      return;
    }

    try {
      var pyodide = await ensurePyodide();

      // Inject the API keys as environment variables. Passing them through
      // pyodide.globals.set avoids any string-escaping issues.
      var openai = panel.querySelector('[data-pg="openai"]').value.trim();
      var gemini = panel.querySelector('[data-pg="gemini"]').value.trim();
      pyodide.globals.set("_kaval_openai_key", openai);
      pyodide.globals.set("_kaval_gemini_key", gemini);
      await pyodide.runPythonAsync(
        "import os as _os\n" +
          "if _kaval_openai_key:\n" +
          "    _os.environ['OPENAI_API_KEY'] = _kaval_openai_key\n" +
          "if _kaval_gemini_key:\n" +
          "    _os.environ['GEMINI_API_KEY'] = _kaval_gemini_key\n" +
          "    _os.environ.setdefault('GOOGLE_API_KEY', _kaval_gemini_key)\n"
      );

      // Expose the chosen in-browser model to Python so "browser/..." examples
      // can build a client without hardcoding an id (mirrors the standalone
      // playground). The embedding default is provided alongside it.
      var modelSel = panel.querySelector('[data-pg="model"]');
      pyodide.globals.set(
        "KAVAL_BROWSER_MODEL",
        (modelSel && modelSel.value) || BROWSER_MODELS[0][0]
      );
      pyodide.globals.set("KAVAL_BROWSER_EMBED_MODEL", DEFAULT_EMBED_MODEL);

      setStatus("Running…");
      var code = editor.getValue();
      // Auto-load any bundled Pyodide packages referenced by imports
      // (numpy, pandas, …). Unknown imports like kavalai are ignored here.
      await pyodide.loadPackagesFromImports(code);
      var result = await pyodide.runPythonAsync(code);
      if (result !== undefined) {
        appendOutput("=> " + result + "\n", "meta");
      }
      setStatus("Done");
    } catch (err) {
      appendOutput(String((err && err.message) || err) + "\n", "stderr");
      setStatus("Error");
    } finally {
      runBtn.disabled = false;
    }
  }

  // -- Button injection -----------------------------------------------------
  function injectButtons() {
    // Skip auto-generated source listings (viewcode) — they are whole modules,
    // not runnable examples.
    if (/\/_modules\//.test(window.location.pathname)) return;

    // Opt-in: only blocks the author marked as browser-runnable get a button.
    // In reStructuredText add `:class: run-in-browser` to the code-block (the
    // class lands on the highlight div); a wrapping `.. container:: run-in-browser`
    // works too. In notebooks, a `run-in-browser` cell tag becomes the
    // `tag_run-in-browser` class on the cell. Examples that need a provider key
    // (openai/gemini) or anything not Pyodide-compatible are simply left
    // unmarked, so they render as plain, un-runnable snippets.
    var blocks = document.querySelectorAll(
      'div[class*="highlight-python"], div[class*="highlight-ipython"]'
    );
    Array.prototype.forEach.call(blocks, function (block) {
      var marked =
        block.classList.contains("run-in-browser") ||
        block.closest(".run-in-browser, .tag_run-in-browser");
      if (!marked) return;
      // A nested `no-run` (class or cell tag) still wins as a hard opt-out.
      if (block.closest(".no-run, .tag_no-run")) return;
      if (block.querySelector(".kaval-pg-run-btn")) return;

      var pre = block.querySelector("pre");
      var code = (pre || block).textContent.replace(/\n+$/, "");
      if (!code.trim()) return;

      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "kaval-pg-run-btn";
      btn.textContent = "Run in browser ▶";
      btn.addEventListener("click", function () {
        openPanel(code);
      });
      block.appendChild(btn);
      block.classList.add("kaval-pg-has-btn");
    });
  }

  // -- Init -----------------------------------------------------------------
  function init() {
    // Make the in-browser LLM/embeddings bridge available before any snippet
    // runs, so "browser/..." clients find window.kavalBrowserLLM.
    installBrowserLLMBridge();
    injectButtons();
    document.addEventListener("keydown", function (e) {
      if (e.key === "Escape" && panel && panel.classList.contains("open")) {
        closePanel();
      }
    });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
