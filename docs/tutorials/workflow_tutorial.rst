.. _workflow_tutorial:

Agentic Workflows: A Technical Guide
===================================

This tutorial explores **Agentic Workflows**, a core concept in Kaval.AI for orchestrating complex AI behaviors.

In Kaval.AI, a workflow is a structured sequence of **Tasks** that collaborate to solve problems. Think of it as a blueprint for how different AI components—or "agents"—interact and share data.

Why Use Workflows?
------------------

While calling an LLM directly works for simple tasks, complex logic often requires more structure. Workflows allow you to:

1.  **Decompose Problems**: Break a large challenge into smaller, manageable steps.
2.  **Ensure Reliability**: Add validation steps and multi-agent reviews.
3.  **Integrate Tools**: Mix LLM reasoning with Python code or external APIs.
4.  **Control Flow**: Use conditional logic to determine the execution path based on intermediate results.

The Anatomy of a Workflow
-------------------------

A workflow is typically defined in a YAML file, which makes it readable and easy to share. Let's look at the main sections:

*   **name & description**: The identity of your agent.
*   **data_types**: This is your "contract." It defines the shape of the input your agent expects and the output it must produce. We use standard JSON Schema types here.
*   **tasks**: The heart of the workflow. A list of steps to be executed.

Our First Example: Socrates
---------------------------

We'll start by creating a philosophical AI that uses the Socratic method.

**YAML Definition (socrates.yaml)**

.. code-block:: yaml

    name: Socrates
    description: A philosophical AI that answers questions using the Socratic method.
    temperature: null # Let the model decide its own creativity!
    data_types:
      input:
        type: object
        properties:
          question: {type: string}
      output:
        type: object
        properties:
          answer: {type: string}

    tasks:
      - name: generate_answer
        type: llm
        prompt: |
          You are Socrates. Answer the following question.
        inputs:
          question: {type: context, value: input.question}
        output: output

**Running it in Python**

The ``kavalai`` package provides a clean interface for loading and running workflows.

.. code-block:: python

    import asyncio
    from kavalai import Workflow

    async def main():
        # Load our philosopher
        workflow = Workflow.from_yaml_path("socrates.yaml")

        # Engage in dialogue
        result = await workflow.run(input_data={"question": "What is justice?"})

        print(f"Socrates says: {result.data.answer}")

    if __name__ == "__main__":
        asyncio.run(main())

**Expected Output**

.. code-block:: text

    Socrates says: Let us begin by asking what you mean by 'justice.' Is it simply
    lawful behavior, or something deeper about how a soul or a city ought to be ordered?...

Conditional Logic: Sentiment Analysis
-------------------------------------

Workflows support conditional execution using the ``when`` field. This allows tasks to run only when specific criteria are met.

.. code-block:: yaml

    name: Sentiment Analysis
    data_types:
      input: {type: object, properties: {text: {type: string}}}
      sentiment: {type: object, properties: {is_positive: {type: boolean}}}
      output: {type: object, properties: {reply: {type: string}}}
    tasks:
      - name: check_sentiment
        type: llm
        prompt: Is the following text positive?
        inputs:
          text: {type: context, value: input.text}
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

In this example, if the input text is positive, only the ``thank_user`` task executes. If it is negative, only the ``apologize`` task executes.

The PythonTask: Adding Custom Logic
-----------------------------------

Some tasks are better suited for Python code than an LLM, such as specific calculations or data manipulation. The ``PythonTask`` allows you to integrate custom logic directly into your workflow.

First, define your Python function:

.. code-block:: python

    def get_length(text: str) -> dict:
        return {"length": len(text)}

Then, reference it in your YAML using the ``python`` task type:

.. code-block:: yaml

    tasks:
      - name: calc_length
        type: python
        python_tool: __main__.get_length
        inputs:
          text: {type: context, value: input.text}
        output: output

Advanced Orchestration: A Jury of Agents
-----------------------------------------

This example demonstrates a multi-agent "Jury" workflow. Three different agents (Optimist, Pessimist, and Realist) analyze a case independently. A final "Lead Judge" then synthesizes their findings into a final decision.

This pattern uses the **CombineTask**, which merges multiple inputs into a single object for consumption by downstream tasks.

.. code-block:: yaml

    name: Jury Workflow
    data_types:
      input: {type: object, properties: {case: {type: string}}}
      summary: {type: object, properties: {analysis: {type: string}, decision: {type: string}}}
      all_summaries:
        type: object
        properties:
          opt: {type: object, properties: {analysis: {type: string}, decision: {type: string}}}
          pess: {type: object, properties: {analysis: {type: string}, decision: {type: string}}}
          real: {type: object, properties: {analysis: {type: string}, decision: {type: string}}}
      output: {type: object, properties: {verdict: {type: string}, reasoning: {type: string}}}

    tasks:
      # 1. The Optimist Judge
      - name: judge_optimist
        type: llm
        prompt: You are an optimist judge. Analyze the case.
        inputs: {case: {type: context, value: input.case}}
        output: summary_opt

      # 2. The Pessimist Judge
      - name: judge_pessimist
        type: llm
        prompt: You are a pessimist judge. Analyze the case.
        inputs: {case: {type: context, value: input.case}}
        output: summary_pess

      # 3. The Realist Judge
      - name: judge_realist
        type: llm
        prompt: You are a realist judge. Analyze the case.
        inputs: {case: {type: context, value: input.case}}
        output: summary_real

      # 4. Combine all summaries into one "docket"
      - name: combine_summaries
        type: combine
        inputs:
          opt: {type: context, value: summary_opt}
          pess: {type: context, value: summary_pess}
          real: {type: context, value: summary_real}
        output: all_summaries

      # 5. Final Decision
      - name: final_judge
        type: llm
        prompt: |
          Consider these three analyses and make a final verdict.
        inputs:
          opt: {type: context, value: all_summaries.opt}
          pess: {type: context, value: all_summaries.pess}
          real: {type: context, value: all_summaries.real}
        output: output

**Running the Jury**

.. code-block:: python

    import asyncio
    from kavalai import Workflow

    async def main():
        workflow = Workflow.from_yaml_path("jury.yaml")
        case = "A man stole bread to feed his hungry family."
        result = await workflow.run(input_data={"case": case})

        print(f"Final Verdict: {result.data.verdict}")
        print(f"Reasoning: {result.data.reasoning}")

    if __name__ == "__main__":
        asyncio.run(main())

**Expected Output**

.. code-block:: text

    Final Verdict: Acquittal under Necessity Defense or extreme leniency.
    Reasoning: While theft is generally illegal, all three judges agree that
    the imminent threat of starvation justifies an exception. The Optimist
    emphasizes human life, the Realist notes the lack of alternatives, and even
    the Pessimist admits that a harsh penalty would serve no societal good here.

Conclusion
----------

You have now seen how to build workflows that can branch, perform calculations, and orchestrate multiple LLM agents.

Workflows are a powerful tool for building reliable, maintainable AI systems. By breaking down complex logic into discrete tasks, you can ensure your agents behave predictably and effectively.

Happy coding!
