# Kaval.AI web widget

An embeddable, in-browser playground that runs Python — and Kaval.AI — entirely
client-side via [Pyodide](https://pyodide.org), with a [WebLLM](https://github.com/mlc-ai/web-llm)
bridge so `browser/...` models work with **no API key, no server and no CORS**.

All Kaval.AI LLM calls here run fully client-side via the WebLLM bridge
(`browser/...` models) — there are **no provider API keys** in the widgets.

This folder is the single source of truth for the playground. It powers:

- the **docs** "Run in browser ▶" buttons (`docs/_ext/kaval_playground.py` copies
  these files into the Sphinx build — nothing is duplicated);
- the **chat widget** (`window.KavalChat`) and the standalone
  **`chat-playground.html`**, where you talk to a workflow you defined in the
  playground; and
- any **third-party page** that wants an embedded Kaval.AI playground or chat.

## Files

| File | Purpose |
|------|---------|
| `kaval-playground.js`  | The engine + widget (`window.KavalPlayground`). |
| `kaval-playground.css` | Shared styling (drawer, embed widget, code-block button). |
| `kaval-chat.js`        | The chat widget (`window.KavalChat`) — a UI driven by a pluggable `send` callback. |
| `kaval-chat.css`       | Chat widget styling (matches the playground console). |
| `chat-playground.html` | Standalone page: chat on the left, the workflow definition on the right. |

## Quick start (standalone)

The page must be served over HTTP so the browser can fetch the kavalai wheel:

```bash
uv build --wheel                 # produces dist/kavalai-*.whl
python -m http.server            # from the repo root
# open http://localhost:8000/webwidget/chat-playground.html
```

## Embed on your own page

```html
<link rel="stylesheet" href="kaval-playground.css">
<script src="kaval-playground.js"></script>

<div id="pg"></div>
<script>
  KavalPlayground.configure({
    // Where to install kavalai from (a PyPI name also works once published):
    wheelUrl: "https://example.com/kavalai-1.0.0-py3-none-any.whl",
  });
  KavalPlayground.mount("#pg", {
    code: 'from kavalai import make_client\n' +
          'print(await make_client(f"browser/{KAVAL_BROWSER_MODEL}").prompt("Hi!"))',
    showPackages: false,
  });
</script>
```

Or, fully declarative — the element's text becomes the initial code:

```html
<div data-kaval-playground>
print("hello from the browser")
</div>
```

### `window.KavalPlayground`

| Method | Description |
|--------|-------------|
| `configure(opts)` | Merge config: `pyodideUrl`, `wheelUrl`, `models`, `embedModel`, `webllmUrl`, `cmBase`. |
| `mount(el, opts)` | Render an inline widget into `el`; returns an instance with `setCode(code)`, `run()`, `getCode()`, `clear()`. `opts`: `code`, `examples` (`{label: code}`), `showModel`, `showPackages`, `title`. |
| `attachButtons()` | Add a Run button to every `.run-in-browser` code block (used by the docs). |
| `open(code)` | Load `code` into the shared drawer and open it. |
| `bridge()` | The WebLLM bridge (`window.kavalBrowserLLM`). |
| `workflowBridge(opts)` | Build a chat bridge to a `workflow` defined in the playground; returns `{ send, reset, sessionId }`. Hand `send` to `KavalChat.mount`. `opts`: `inputKey` (default `user_message`), `replyKey` (default `agent_response`). |

## Chat with a workflow

The chat widget is intentionally **decoupled** from the playground: it knows
nothing about Pyodide or Kaval.AI and just renders a conversation on top of a
single `send(message)` callback. Pair it with `KavalPlayground.workflowBridge()`
and it talks to a `workflow` you defined and ran in the playground — see the
standalone `chat-playground.html`.

```html
<link rel="stylesheet" href="kaval-playground.css">
<link rel="stylesheet" href="kaval-chat.css">
<script src="kaval-playground.js"></script>
<script src="kaval-chat.js"></script>

<div id="pg"></div>
<div id="chat"></div>
<script>
  KavalPlayground.mount("#pg", {
    code: 'from kavalai.workflow import WorkflowEngine, InMemoryDataStorage\n' +
          'workflow = WorkflowEngine.from_yaml(YAML, storage=InMemoryDataStorage())',
  });

  // The bridge runs `workflow.run({user_message}, session_id=...)`, reusing one
  // session so history-aware nodes (use_history, on by default) see the whole
  // conversation. InMemoryDataStorage is thread-free, so history works under
  // Pyodide (aiosqlite can't start its worker thread there). If the workflow
  // has no storage, the bridge attaches one. The chat works once a `workflow`
  // has been Run ▶.
  var bridge = KavalPlayground.workflowBridge();
  KavalChat.mount("#chat", { send: bridge.send, onReset: bridge.reset });
</script>
```

`KavalChat.mount(el, opts)` takes `send` (required — `(message) => Promise<string
| {reply} | {error}>`), plus optional `title`, `placeholder`, `greeting` and
`onReset`. It returns an instance with `addMessage(role, text)`, `clear()`,
`focus()` and `setStatus(text)`. Because `send` is just a callback, the chat
widget can front any backend, not only the playground.

Two globals are exposed to the running Python: `KAVAL_BROWSER_MODEL` (from the
model picker) and `KAVAL_BROWSER_EMBED_MODEL`, so examples can build a
`browser/...` client without hardcoding a model id.

## Requirements

`browser/...` models need a WebGPU-capable browser (recent Chrome/Edge, or
Firefox with `dom.webgpu.enabled`). Models download on first use and are cached
by the browser. Plain Python and the rest of Kaval.AI work without WebGPU.
