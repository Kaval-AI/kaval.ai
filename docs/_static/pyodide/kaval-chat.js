/*
 * Kaval.AI chat widget.
 *
 * A small, dependency-free chat UI that is deliberately decoupled from the
 * playground: it knows nothing about Pyodide, WebLLM or Kaval.AI. You hand it a
 * single `send(message) -> Promise<string | {reply} | {error}>` callback and it
 * renders the conversation, a composer and a typing indicator on top of it.
 *
 * Paired with the playground it talks to a `workflow` the user defined in the
 * editor — wire `send` to `KavalPlayground.workflowBridge().send` (see
 * webwidget/chat-playground.html). But any async function works, so the widget
 * can front any backend.
 *
 * Public API (window.KavalChat):
 *   configure(opts)   merge defaults: { title, placeholder, greeting }
 *   mount(el, opts)   render a chat widget into `el` and return its instance;
 *                     opts: { send (required), title, placeholder, greeting,
 *                     onReset }. The instance exposes:
 *                       addMessage(role, text)  append a bubble
 *                       clear()                 reset the transcript
 *                       focus()                 focus the composer
 *                       setStatus(text)         set the header status text
 */
(function () {
  "use strict";

  var CONFIG = {
    title: "Chat",
    placeholder: "Message your workflow…",
    greeting: "",
  };

  // Kaval.AI icon mark (matches the playground widget), inlined.
  var LOGO_SVG =
    '<svg class="kaval-chat-logo" viewBox="0 0 79.375 79.375" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">' +
    '<g transform="translate(-27.011657,-24.494702)">' +
    '<g transform="matrix(0.26458333,0,0,0.26458333,-506.65292,-7.2552973)">' +
    '<path d="m 2317,120 h -300 v 300 h 300 z" fill="#002626"/>' +
    '<path d="m 2145.42,235.72 c -1.16,-1.164 -1.82,-2.752 -1.82,-4.424 V 120 H 2017 v 1.842 L 2138.82,233.9 c 1.29,1.185 2.01,2.857 2.01,4.593 V 420 h 176.15 v -12.679 z" fill="#acc12f"/>' +
    "</g></g></svg>";

  function esc(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  function createInstance(root, opts) {
    opts = opts || {};
    var send = opts.send;
    if (typeof send !== "function") {
      throw new Error("KavalChat.mount: a `send(message)` callback is required");
    }
    var title = opts.title != null ? opts.title : CONFIG.title;
    var placeholder = opts.placeholder != null ? opts.placeholder : CONFIG.placeholder;
    var greeting = opts.greeting != null ? opts.greeting : CONFIG.greeting;

    root.className = "kaval-chat" + (root.className ? " " + root.className : "");
    root.setAttribute("role", "group");
    root.setAttribute("aria-label", "Kaval.AI chat");

    root.innerHTML =
      '<div class="kaval-chat-header">' +
      '<span class="kaval-chat-title">' +
      LOGO_SVG +
      "<span>" +
      esc(title) +
      "</span></span>" +
      '<span class="kaval-chat-status" data-chat="status">Idle</span>' +
      '<button class="kaval-chat-icon" data-chat="reset" title="New conversation" aria-label="New conversation">⟳</button>' +
      "</div>" +
      '<div class="kaval-chat-log" data-chat="log" aria-live="polite"></div>' +
      '<form class="kaval-chat-composer" data-chat="composer">' +
      '<textarea class="kaval-chat-input" data-chat="input" rows="1" ' +
      'placeholder="' +
      esc(placeholder) +
      '" autocomplete="off"></textarea>' +
      '<button class="kaval-chat-send" data-chat="send" type="submit">Send</button>' +
      "</form>";

    var q = function (sel) {
      return root.querySelector(sel);
    };
    var logEl = q('[data-chat="log"]');
    var statusEl = q('[data-chat="status"]');
    var formEl = q('[data-chat="composer"]');
    var inputEl = q('[data-chat="input"]');
    var sendBtn = q('[data-chat="send"]');
    var resetBtn = q('[data-chat="reset"]');

    var busy = false;

    function setStatus(text) {
      statusEl.textContent = text;
    }

    function scrollToBottom() {
      logEl.scrollTop = logEl.scrollHeight;
    }

    // kind: "user" | "assistant" | "error". Returns the bubble element so a
    // pending message can be replaced in place once the reply arrives.
    function addMessage(kind, text) {
      var row = document.createElement("div");
      row.className = "kaval-chat-msg kaval-chat-msg--" + kind;
      var bubble = document.createElement("div");
      bubble.className = "kaval-chat-bubble";
      bubble.textContent = text;
      row.appendChild(bubble);
      logEl.appendChild(row);
      scrollToBottom();
      return row;
    }

    function addTyping() {
      var row = document.createElement("div");
      row.className = "kaval-chat-msg kaval-chat-msg--assistant";
      row.innerHTML =
        '<div class="kaval-chat-bubble kaval-chat-typing">' +
        '<span></span><span></span><span></span>' +
        "</div>";
      logEl.appendChild(row);
      scrollToBottom();
      return row;
    }

    // Render quick-reply choices (a structured `choices: string[]` from the
    // reply) as clickable chips; clicking one sends it as the next message.
    function addChoices(choices) {
      if (!Array.isArray(choices) || !choices.length) return;
      var row = document.createElement("div");
      row.className = "kaval-chat-choices";
      choices.forEach(function (choice) {
        var chip = document.createElement("button");
        chip.type = "button";
        chip.className = "kaval-chat-choice";
        chip.textContent = choice;
        chip.addEventListener("click", function () {
          if (busy) return;
          inputEl.value = String(choice);
          submit();
        });
        row.appendChild(chip);
      });
      logEl.appendChild(row);
      scrollToBottom();
    }

    function autoGrow() {
      inputEl.style.height = "auto";
      // Cap growth so the composer never eats the whole panel.
      inputEl.style.height = Math.min(inputEl.scrollHeight, 160) + "px";
    }

    function clear() {
      logEl.textContent = "";
      if (greeting) addMessage("assistant", greeting);
      setStatus("Idle");
    }

    async function submit() {
      if (busy) return;
      var text = inputEl.value.trim();
      if (!text) return;

      busy = true;
      sendBtn.disabled = true;
      addMessage("user", text);
      inputEl.value = "";
      autoGrow();
      setStatus("Thinking…");
      var typing = addTyping();

      try {
        var result = await send(text);
        // Accept a bare string or a structured {reply} / {error}.
        if (typeof result === "string") result = { reply: result };
        result = result || {};
        logEl.removeChild(typing);
        if (result.error) {
          addMessage("error", result.error);
          setStatus("Error");
        } else {
          addMessage("assistant", result.reply != null ? result.reply : "");
          // Surface a structured `choices` list (top-level or nested under the
          // workflow's full `data`) as quick-reply chips.
          addChoices(
            result.choices || (result.data && result.data.choices) || null
          );
          setStatus("Idle");
        }
      } catch (err) {
        if (typing.parentNode === logEl) logEl.removeChild(typing);
        addMessage("error", String((err && err.message) || err));
        setStatus("Error");
      } finally {
        busy = false;
        sendBtn.disabled = false;
        inputEl.focus();
      }
    }

    formEl.addEventListener("submit", function (e) {
      e.preventDefault();
      submit();
    });
    // Enter sends; Shift+Enter inserts a newline.
    inputEl.addEventListener("keydown", function (e) {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submit();
      }
    });
    inputEl.addEventListener("input", autoGrow);
    resetBtn.addEventListener("click", function () {
      clear();
      if (typeof opts.onReset === "function") opts.onReset();
      inputEl.focus();
    });

    if (greeting) addMessage("assistant", greeting);

    return {
      root: root,
      addMessage: addMessage,
      clear: clear,
      focus: function () {
        inputEl.focus();
      },
      setStatus: setStatus,
    };
  }

  var KavalChat = {
    configure: function (opts) {
      if (opts) Object.assign(CONFIG, opts);
      return KavalChat;
    },
    mount: function (el, opts) {
      var root = typeof el === "string" ? document.querySelector(el) : el;
      if (!root) throw new Error("KavalChat.mount: container not found");
      return createInstance(root, opts || {});
    },
  };
  window.KavalChat = KavalChat;
})();
