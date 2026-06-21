/*
 * Kaval.AI docs playground.
 *
 * Adds a "Run in browser" button to every Python code block in the Sphinx HTML
 * docs. Clicking it opens a side-panel (a right-hand drawer on desktop, a
 * full-width vertical split on mobile) with an editable copy of the snippet and
 * an output pane, and runs it through Pyodide with the kavalai wheel installed.
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

  var LS_OPENAI = "kaval-pg-openai-key";
  var LS_GEMINI = "kaval-pg-gemini-key";

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

    return panel;
  }

  function wireKeyInput(selector, storageKey) {
    var input = panel.querySelector(selector);
    try {
      input.value = window.localStorage.getItem(storageKey) || "";
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

    var blocks = document.querySelectorAll('div[class*="highlight-python"]');
    Array.prototype.forEach.call(blocks, function (block) {
      // Opt out with `:class: no-run` on the directive.
      if (block.closest(".no-run")) return;
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
