# pydantic-ai-autobrowser

Technical demo of a human-assisted, multi-agent browser automation loop built with PydanticAI and Playwright MCP.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Workflow](#workflow)
- [Technical Stack](#technical-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [License](#license)

## Overview

`pydantic-ai-autobrowser` is an experimental browser automation demo built around a simple loop: Planner -> Browser Agent -> Critique. The planner decides the next atomic browser action, the browser agent executes it through Playwright MCP, and the critique agent evaluates the result before the next iteration.

The project is designed for tasks that may require real user involvement. Instead of refusing early, the workflow can pause for login, secrets, OTPs, manual browser actions, or explicit approval for risky steps, then continue from the current browser state.

## Features

- **Multi-agent closed loop**: planner, browser executor, and critique agent collaborate until the task is finished or explicitly terminated.
- **Playwright MCP foundation**: browser execution is built on top of Playwright MCP, giving the system a clean MCP-based interface for navigation, interaction, inspection, screenshots, and tab management.
- **Human in the loop where it matters**: supports secret input, manual browser actions, and approval gates for risky actions such as payments or deletions.
- **DOM-grounded browser execution**: the browser agent works from Playwright MCP outputs, fresh page snapshots, and real page state instead of answering from memory or guessing UI behavior.
- **Visual verification after actions**: the orchestrator captures before/after screenshots and runs screenshot analysis to validate whether the UI actually changed as expected.
- **Current-URL-aware replanning**: the planner uses the active browser URL to avoid redundant navigation and continue from the page already open.
- **Traceability and debugging**: logs, screenshots, and saved conversation artifacts are written under `logs/` for inspection.
- **Basic resilience**: transient model retries, anti-loop guidance, and critique-driven replanning are built into the workflow.

## How It Works

Each run starts with the Planner Agent, which turns the user request and current browser context into one focused next action instead of attempting the whole task at once.

That action is passed to the Browser Agent, which executes it through Playwright MCP using real page state, tool outputs, screenshots, and browser context rather than relying on guessed UI behavior.

After execution, the Critique Agent evaluates whether the intended result actually happened and decides whether the task is complete or whether structured feedback should be sent back to the planner for another iteration.

The loop ends when the goal is reached, the user stops a manual or sensitive step, or the system hits a terminal failure or retry limit.

## Workflow

**Step 1: Planning Phase**

- Receives the user request and the current browser URL.
- Builds or updates the step-by-step plan.
- Returns exactly one next browser action.
- Replans when the previous step failed, was too risky, or needs a different approach.

**Step 2: Execution Phase**

- Receives the current step and executes it through Playwright MCP.
- Performs focused browser actions such as navigation, clicks, typing, DOM inspection, tab management, and screenshot capture.
- Pauses for login, OTP, secret input, manual browser intervention, or approval for risky actions.

**Step 3: Evaluation Phase**

- Reviews the browser result together with tool output and before/after visual evidence.
- Checks whether the intended UI change actually happened.
- Decides whether the current step succeeded and whether the overall task is complete.
- Either finishes the run or sends structured feedback back to the planner.

**Run lifecycle**

- The loop continues until the task is completed, the user aborts a sensitive or manual step, or the system reaches a terminal failure condition.
- Each run produces logs, screenshots, and saved conversation artifacts for inspection and debugging.

## Technical Stack

- **Python**: core runtime for the orchestration loop, agents, and CLI entrypoint.
- **PydanticAI**: agent framework used to define the planner, browser, and critique agents.
- **Playwright MCP**: browser automation layer used for navigation, interaction, inspection, screenshots, and tab control.
- **Google Gemini / `google-genai`**: model provider and SDK currently used for text and screenshot-based reasoning.
- **Logfire**: tracing and observability for agent execution and runtime events.
- **Pillow**: image processing for screenshot comparison and analysis helpers.
- **`uv`**: dependency management and local project execution.
- **Node.js / `npx`**: required to start the Playwright MCP server and browser tooling.

## Project Structure

```text
.
|-- main.py
|-- src/
|   |-- agents/
|   |   |-- planner_agent.py
|   |   |-- browser_agent.py
|   |   `-- critique_agent.py
|   |-- orchestrator/
|   |   |-- workflow.py
|   |   |-- engine.py
|   |   |-- runner.py
|   |   `-- state.py
|   |-- ui/
|   |   `-- console.py
|   |-- utils/
|   |   |-- screenshot.py
|   |   |-- image_analysis.py
|   |   `-- msg_parser.py
|   |-- config.py
|   `-- logging_setup.py
`-- logs/
```

- `main.py`: CLI entrypoint that starts the interactive or one-shot run.
- `src/agents/`: planner, browser, and critique agents.
- `src/orchestrator/`: run loop, state management, retries, and event flow.
- `src/ui/`: console-facing interaction for prompts, pauses, and run output.
- `src/utils/`: screenshots, screenshot analysis, and message parsing helpers.
- `src/config.py`: environment-driven runtime settings.
- `src/logging_setup.py`: logging and observability setup.
- `logs/`: runtime artifacts such as logs, screenshots, and saved conversations.

## Quick Start

### Prerequisites

- Python 3.14+
- `uv`
- Node.js with `npx`
- A Google API key for Gemini

### Install

```bash
uv sync
```

### Configure

```bash
cp .env.example .env
```

On Windows PowerShell, you can use:

```powershell
Copy-Item .env.example .env
```

Set at least:

- `GOOGLE_API_KEY`
- `MODEL_NAME`

Optional:

- `LOGFIRE_TOKEN`
- `SCREENSHOT_ANALYSIS_MODEL`

The old Gemini-style values are expected here, for example:

- `MODEL_NAME=gemini-2.5-flash`
- `SCREENSHOT_ANALYSIS_MODEL=gemini-2.5-flash`

### Run

```bash
uv run python main.py "Go to my Gmail inbox and move the first 10 emails to spam"
```

You can also start in interactive mode:

```bash
uv run python main.py
```

If the run needs login, OTP, secret input, or a manual browser step, follow the console prompt and continue from there.

The Playwright MCP server is started automatically through `npx`, and logs, screenshots, and conversation artifacts are written to `logs/`.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
