# Junie Guidelines

Always refer to `junie.md` in the project root for an up-to-date overview of the project structure, components, and technical details. This file provides context on:
- Project architecture (SDK vs Backoffice)
- Directory structure and file purposes
- Key technical stacks (FastAPI, Angular, etc.)
- Workflow execution logic
- Security and isolation patterns

Before starting any task, check `junie.md` to understand the relevant modules and their roles.
After unit testing, update the `junie.md` for any important changes.

## Executing commands.

When the developer denies a delete/remove or other dangerous command, ask them to manually perform the command.

## Unit Testing
At the end of every coding task, you must update the relevant unit tests and run them to ensure that your changes haven't introduced any regressions and that the new functionality works as expected.

### Testing Goal
- Aim for **100% test coverage** for all new and modified code.
- Always run tests before submitting a task.
- If you know which tests fail, run only those tests to save time.

## Coding guidelines

- Don't use deprecated *ngIf for Angular.
- Try to refactor code blocks that perform specific tasks into dedicated functions.
- Keep tests for a single file also in a single file e.g agent.py has just test_agent.py.
- Don't update README.md

## Styling guidelines
- Prefer styles and components in common.css for styling CSS.
