Getting Started
===============

This tutorial will guide you through the process of creating your first AI agent with Kaval.AI.

Installation
------------

To install Kaval.AI, run:

.. code-block:: bash

   pip install -e .

Creating a simple Agent
-----------------------

Create a file named `my_agent.yaml` with the following content:

.. code-block:: yaml

   name: My first agent
   tasks:
     - name: greet
       prompt: "Hello, I am Kaval.AI. How can I help you today?"

Run the agent:

.. code-block:: bash

   # Example command to run the agent
   python -m kavalai.tools.cli_chat --workflow my_agent.yaml
