Getting Started
===============

This tutorial will guide you through the process of creating your first AI agent with Kaval.AI.

Installation
------------

Kaval.AI can be installed from PyPI, Conda, or by cloning the repository from GitHub.

Installing from PyPI
^^^^^^^^^^^^^^^^^^^^

Once Kaval.AI is published to PyPI, you will be able to install it using pip:

.. code-block:: bash

   pip install kavalai

Installing from Conda
^^^^^^^^^^^^^^^^^^^^^

Once Kaval.AI is published to Conda, you will be able to install it using conda or mamba:

.. code-block:: bash

   conda install kavalai

Cloning the repository
^^^^^^^^^^^^^^^^^^^^^^

First, clone the repository to your local machine:

.. code-block:: bash

   git clone https://github.com/kaval-ai/kaval.ai.git
   cd kaval.ai

Installing the library
^^^^^^^^^^^^^^^^^^^^^^

It is recommended to use a virtual environment. Once you have activated your environment, you can install the library in editable mode:

.. code-block:: bash

   pip install -e .

This command uses the ``pyproject.toml`` file to manage dependencies and project metadata.

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
