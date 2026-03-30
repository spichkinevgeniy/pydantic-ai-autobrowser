import sys
from pathlib import Path

from src.orchestrator import OrchestratorEvent
from src.types import OrchestratorRunResult


class ConsoleProgressUI:
    RESET = "\033[0m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    CYAN = "\033[96m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    RED = "\033[91m"

    def __init__(self, log_file: Path | str) -> None:
        self.log_file = str(log_file)
        self._last_plan = ""
        self._use_color = sys.stdout.isatty()

    def handle_event(self, event: OrchestratorEvent) -> None:
        if event.event_type == "run_started":
            self._print_header(event)
            return

        if event.event_type == "iteration_started":
            print()
            print(self._line(f"[ITERATION {event.iteration}] starting", color=self.CYAN, bold=True))
            return

        if event.event_type == "planner_completed":
            self._section("PLANNER", self.BLUE)
            print(f"  Next step: {event.current_step or '-'}")
            if event.plan and event.plan != self._last_plan:
                self._last_plan = event.plan
                self._print_block("  Plan", event.plan)
            return

        if event.event_type == "browser_running":
            elapsed_seconds = event.data.get("elapsed_seconds")
            self._section("BROWSER", self.YELLOW)
            print(f"  Step: {event.current_step or '-'}")
            if elapsed_seconds:
                print(f"  Status: still working for ~{elapsed_seconds}s")
            else:
                print("  Status: still working")
            return

        if event.event_type == "browser_completed":
            browser_summary = str(event.data.get("browser_summary", "")).strip()
            ss_analysis = str(event.data.get("ss_analysis", "")).strip()
            self._section("BROWSER", self.YELLOW)
            print(f"  Step: {event.current_step or '-'}")
            if browser_summary:
                self._print_block("  Summary", browser_summary)
            if ss_analysis:
                self._print_block("  Visual check", ss_analysis)
            return

        if event.event_type == "critique_completed":
            terminate = bool(event.data.get("terminate"))
            feedback = str(event.data.get("feedback", "")).strip()
            self._section("CRITIQUE", self.GREEN)
            print(f"  Decision: {'finish' if terminate else 'continue'}")
            if feedback:
                self._print_block("  Feedback", feedback)
            return

        if event.event_type == "run_failed":
            print()
            print(self._line("RUN FAILED", color=self.RED, bold=True))
            print(f"  Error: {event.message or 'Unknown error'}")
            print(f"  Trace log: {self.log_file}")
            return

        if event.event_type == "run_finished":
            print("\nRun finished")
            if event.final_response:
                self._print_block("  Final response preview", event.final_response)
            print(f"  Trace log: {self.log_file}")

    def render_result(self, result: OrchestratorRunResult) -> None:
        print("\n" + "=" * 72)
        print(self._line("FINAL PLAN", color=self.CYAN, bold=True))
        print(result.plan or "-")
        if result.final_response:
            print()
            print(self._line("FINAL RESPONSE", color=self.GREEN, bold=True))
            print(result.final_response)
        else:
            print()
            print(self._line("NEXT STEP", color=self.YELLOW, bold=True))
            print(result.next_step or "-")
        print()
        print(self._line(f"Trace log: {self.log_file}", color=self.DIM))

    def _print_header(self, event: OrchestratorEvent) -> None:
        query = str(event.data.get("user_query", "")).strip()
        print("=" * 72)
        print(self._line("BROWSER AGENT RUN", color=self.CYAN, bold=True))
        if query:
            print(f"Query: {query}")
        print(self._line(f"Trace log: {self.log_file}", color=self.DIM))
        print("=" * 72)

    def _section(self, name: str, color: str) -> None:
        print(self._line(name, color=color, bold=True))

    def _print_block(self, title: str, text: str) -> None:
        print(f"{title}:")
        for line in str(text).splitlines():
            if line.strip():
                print(f"    {line}")
            else:
                print()

    def _line(self, text: str, *, color: str = "", bold: bool = False) -> str:
        if not self._use_color:
            return text

        prefix = ""
        if bold:
            prefix += self.BOLD
        prefix += color
        return f"{prefix}{text}{self.RESET}"
