Agentic Workflows: A Technical Guide
===================================

This guide explores **Agentic Workflows**, a core component of Kaval.AI designed for orchestrating complex AI behaviors.

In Kaval.AI, a workflow is a structured sequence of **Tasks** that collaborate to solve problems. Unlike simple LLM calls, workflows allow for state management, tool integration, and conditional execution, providing a robust framework for building production-ready AI agents.

Core Concepts
-------------

1. **Declarative Definition:** Workflows are defined in YAML, separating logic from implementation.
2. **Strict Data Typing:** Input and output schemas are defined using JSON Schema, ensuring data integrity across tasks.
3. **Task Orchestration:** A variety of task types (LLM, Python, REST, MCP, etc.) can be combined to form a complex execution graph.
4. **Context Management:** Data flows seamlessly between tasks using a shared execution context.

Defining Data Types
-------------------

The ``data_types`` section acts as a contract for your workflow. It defines the structure of the ``input`` the workflow expects and the ``output`` it will eventually produce. You can also define intermediate types for internal task communication.

.. code-block:: yaml

    data_types:
      # The main input to the workflow
      input:
        type: object
        properties:
          user_query: {type: string}
          max_results: {type: integer, default: 5}

      # An internal data structure
      SearchResult:
        type: object
        properties:
          title: {type: string}
          url: {type: string}
          relevance: {type: number}

      # The final output of the workflow
      output:
        type: object
        properties:
          answer: {type: string}
          references: {type: array, items: {$ref: SearchResult}}

Task Types and Tool Connectivity
--------------------------------

Tasks are the building blocks of a workflow. Each task has a ``type`` and a set of ``inputs`` and ``outputs``.

LLM Task (``llm``)
~~~~~~~~~~~~~~~~~~

Executes a prompt using an LLM.

.. code-block:: yaml

    - name: generate_summary
      type: llm
      prompt: Summarize the findings for the user.
      inputs:
        findings: {type: context, value: search_step.results}
      output: output

Kaval.AI automatically constructs a final prompt by appending an **INPUT DATA** section to your ``prompt`` string. This section contains all the resolved values from the ``inputs`` dictionary. This ensures the LLM has access to all necessary context without manual string interpolation.

Python Task (``python``)
~~~~~~~~~~~~~~~~~~~~~~~~

Calls a Python function directly. This is ideal for data processing or using existing Python libraries.

.. code-block:: yaml

    - name: process_data
      type: python
      python_tool: my_package.utils.data_cleaner
      inputs:
        raw_data: {type: context, value: input.user_query}
      output: cleaned_data

REST Task (``rest``)
~~~~~~~~~~~~~~~~~~~~

Interacts with external APIs. You must define a ``rest_servers`` section to configure the base URL and authentication.

.. code-block:: yaml

    rest_servers:
      - name: weather_api
        url: "https://api.weather.com/v1"

    tasks:
      - name: get_weather
        type: rest
        rest_server: weather_api
        tool: "/current"
        inputs:
          city: {type: context, value: input.city}
        output: weather_data

MCP Task (``mcp``)
~~~~~~~~~~~~~~~~~~

Uses the **Model Context Protocol** to interact with specialized tool servers (via stdio or HTTP).

.. code-block:: yaml

    mcp_servers:
      - name: filesystem
        command: "npx"
        args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/data"]

    tasks:
      - name: read_file
        type: mcp
        mcp_server: filesystem
        tool: "read_file"
        inputs:
          path: {type: literal, value: "report.txt"}
        output: file_content

RAG Task (``rag_query``)
~~~~~~~~~~~~~~~~~~~~~~~~

Performs a semantic search against a vector database.

.. code-block:: yaml

    - name: search_knowledge_base
      type: rag_query
      text: {type: context, value: input.user_query}
      top_k: 3
      collection_name: "documentation"
      output: retrieved_docs

Planning Agent (``agent``)
~~~~~~~~~~~~~~~~~~~~~~~~~~

A higher-level task that uses an LLM to autonomously decide which tools (REST or MCP) to call to solve a problem.

.. code-block:: yaml

    - name: autonomous_researcher
      type: agent
      prompt: "Research the given topic and provide a detailed report."
      max_steps: 10
      allowed_tools: ["mcp://filesystem.*", "mcp://google_search.*"]
      inputs:
        topic: {type: context, value: input.user_query}
      output: output

Example: RSS News Agent
-----------------------

Let's combine these concepts into a practical agent that reads RSS feeds and discusses them.

.. code-block:: yaml

    name: News commentator bot.
    description: Converses with the user on the topic of latest news available to it.
    version: "1.0"
    llm_model: openai/gpt-4o-mini
    data_types:
      RssFeedItem:
        type: object
        properties:
          title: {type: string}
          link: {type: string}
          summary: {type: string}
      Feed:
        type: object
        properties:
          title: {type: string}
          url: {type: string}
          items: {type: array, items: {$ref: RssFeedItem}}
      hacker_news_feed: {$ref: Feed}
      antarctica_news_feed: {$ref: Feed}
      input:
        type: object
        properties:
          user_message: {type: string}
      output:
        type: object
        properties:
          agent_response: {type: string}
    tasks:
      - name: Fetch Hacker news RSS
        type: python
        python_tool: kavalai.tools.rss.get_rss_feed
        inputs:
          url: {type: literal, value: "https://news.ycombinator.com/rss"}
          max_results: {type: literal, value: 15}
        output: hacker_news_feed
      - name: Fetch Antarctica news
        type: python
        python_tool: kavalai.tools.rss.get_rss_feed
        inputs:
          url: {type: literal, value: "https://feeds.feedburner.com/princess_elisabeth_station/news"}
          max_results: {type: literal, value: 15}
        output: antarctica_news_feed
      - name: discuss_news
        type: llm
        prompt: |
          Task: Summarize the latest news with two sentences less than 200 characters. Tell them to user and discuss the news with them.
        inputs:
          hacker_news_feed: {type: context}
          antarctica_news_feed: {type: context}
          user_message: {type: context, value: input.user_message}
        output: output

Running Workflows in Python
---------------------------

The ``Workflow`` class provides a simple interface for loading and executing these definitions.

.. code-block:: python

    from kavalai.agents.workflow import Workflow
    import asyncio

    async def run_agent():
        workflow = Workflow.from_yaml(rss_agent_yaml) # rss_agent_yaml is the string above

        user_input = "hi, how are you?"
        print(f"User: {user_input}")
        result = await workflow.run({"user_message": user_input})
        print(f"Agent: {result.data.agent_response}")

    if __name__ == "__main__":
        asyncio.run(run_agent())

**Output:**

.. code-block:: text

    User: hi, how are you?
    Agent: I'm doing well, thanks! Have you heard about the latest news? The Princess Elisabeth Antarctica station has officially closed for the season, and the team is heading home after a successful scientific season.

Conditional Logic and Control Flow
----------------------------------

Workflows support conditional execution using the ``when`` field. This allows tasks to run only when specific criteria are met, enabling dynamic execution paths.

.. code-block:: yaml

    tasks:
      - name: check_sentiment
        type: llm
        prompt: Is the following text positive?
        inputs: {text: {type: context, value: input.text}}
        output: sentiment

      - name: thank_user
        type: llm
        when: {eq: [{"type": "context", "value": "sentiment.is_positive"}, true]}
        prompt: The user was positive. Say thank you.
        output: output

      - name: apologize
        type: llm
        when: {eq: [{"type": "context", "value": "sentiment.is_positive"}, false]}
        prompt: The user was negative. Apologize.
        output: output

Advanced Orchestration: A Jury of Agents
-----------------------------------------

This example demonstrates a multi-agent "Jury" workflow where different agents (Optimist, Pessimist, and Realist) analyze a case independently. A final "Lead Judge" then synthesizes their findings.

This pattern uses the **CombineTask**, which merges multiple inputs into a single object.

.. code-block:: yaml

    name: Jury Workflow
    data_types:
      input: {type: object, properties: {case: {type: string}}}
      Summary:
        type: object
        properties: {analysis: {type: string}, decision: {type: string}}
      AllSummaries:
        type: object
        properties:
          opt: {$ref: Summary}
          pess: {$ref: Summary}
          real: {$ref: Summary}
      output: {type: object, properties: {verdict: {type: string}, reasoning: {type: string}}}

    tasks:
      - name: judge_optimist
        type: llm
        prompt: You are an optimist judge. Analyze the case.
        inputs: {case: {type: context, value: input.case}}
        output: summary_opt

      - name: judge_pessimist
        type: llm
        prompt: You are a pessimist judge. Analyze the case.
        inputs: {case: {type: context, value: input.case}}
        output: summary_pess

      - name: judge_realist
        type: llm
        prompt: You are a realist judge. Analyze the case.
        inputs: {case: {type: context, value: input.case}}
        output: summary_real

      - name: combine_summaries
        type: combine
        inputs:
          opt: {type: context, value: summary_opt}
          pess: {type: context, value: summary_pess}
          real: {type: context, value: summary_real}
        output: AllSummaries

      - name: final_judge
        type: llm
        prompt: |
          Consider these three analyses and make a final verdict.
        inputs:
          opt: {type: context, value: AllSummaries.opt}
          pess: {type: context, value: AllSummaries.pess}
          real: {type: context, value: AllSummaries.real}
        output: output

Conclusion
----------

Kaval.AI Workflows provide the structure needed to move from simple AI experiments to production-grade agentic systems. By combining declarative YAML definitions, strict data typing, and a diverse set of task types, you can build agents that are reliable, maintainable, and highly capable.

You can find more advanced examples in the `notebooks/` directory.

Happy coding!
