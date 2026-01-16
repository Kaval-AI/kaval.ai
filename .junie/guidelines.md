# Junie Guidelines

Always refer to `junie.md` in the project root for an up-to-date overview of the project structure, components, and technical details. This file provides context on:
- Project architecture (SDK vs Backoffice)
- Directory structure and file purposes
- Key technical stacks (FastAPI, Angular, etc.)
- Workflow execution logic
- Security and isolation patterns

Before starting any task, check `junie.md` to understand the relevant modules and their roles.
After unit testing, update the `junie.md` for any important changes.

## Unit Testing
At the end of every coding task, you must update the relevant unit tests and run them to ensure that your changes haven't introduced any regressions and that the new functionality works as expected.

### Testing Goal
- Aim for **100% test coverage** for all new and modified code.
- Always run tests before submitting a task.
