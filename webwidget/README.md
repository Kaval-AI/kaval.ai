# Kaval.AI web widget

An embeddable, in-browser playground that runs Python — and Kaval.AI — entirely
client-side via [Pyodide](https://pyodide.org), with a [WebLLM](https://github.com/mlc-ai/web-llm)
bridge so `browser/...` models work with **no API key, no server and no CORS**.

This folder is the single source of truth for the playground. It powers:

- the **docs** "Run in browser ▶" buttons (`docs/_ext/kaval_playground.py` copies
  these files into the Sphinx build — nothing is duplicated);
- the standalone **`python-playground.html`** full-page playground; and
- any **third-party page** that wants an embedded Kaval.AI playground.

## Files

| File | Purpose |
|------|---------|
| `kaval-playground.js`  | The engine + widget (`window.KavalPlayground`). |
| `kaval-playground.css` | Shared styling (drawer, embed widget, code-block button). |
| `python-playground.html` | Standalone full-page playground built on the widget. |

## Quick start (standalone)

The page must be served over HTTP so the browser can fetch the kavalai wheel:

```bash
uv build --wheel                 # produces dist/kavalai-*.whl
python -m http.server            # from the repo root
# open http://localhost:8000/webwidget/python-playground.html
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
| `mount(el, opts)` | Render an inline widget into `el`; returns an instance with `setCode(code)`, `run()`, `getCode()`, `clear()`. `opts`: `code`, `examples` (`{label: code}`), `showModel`, `showKeys`, `showPackages`, `title`. |
| `attachButtons()` | Add a Run button to every `.run-in-browser` code block (used by the docs). |
| `open(code)` | Load `code` into the shared drawer and open it. |
| `bridge()` | The WebLLM bridge (`window.kavalBrowserLLM`). |

Two globals are exposed to the running Python: `KAVAL_BROWSER_MODEL` (from the
model picker) and `KAVAL_BROWSER_EMBED_MODEL`, so examples can build a
`browser/...` client without hardcoding a model id.

## Requirements

`browser/...` models need a WebGPU-capable browser (recent Chrome/Edge, or
Firefox with `dom.webgpu.enabled`). Models download on first use and are cached
by the browser. Plain Python and the rest of Kaval.AI work without WebGPU.
