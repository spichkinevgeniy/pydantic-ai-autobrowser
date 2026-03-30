import getpass
import sys
from pathlib import Path

from src.orchestrator import OrchestratorEvent
from src.types import HumanActionRequest, HumanActionResponse, OrchestratorRunResult


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
            current_url = str(event.data.get("current_url", "")).strip()
            if current_url:
                print(f"  Current URL: {current_url}")
            print(f"  Next step: {event.current_step or '-'}")
            if event.plan and event.plan != self._last_plan:
                self._last_plan = event.plan
                self._print_block("  Plan", event.plan)
            return

        if event.event_type == "run_paused":
            print()
            print(self._line("WAITING FOR HUMAN ACTION", color=self.RED, bold=True))
            return

        if event.event_type == "run_resumed":
            print(self._line("RESUMING EXECUTION", color=self.CYAN, bold=True))
            return

        if event.event_type == "human_input_requested":
            self._section("HUMAN INPUT REQUIRED", self.RED)
            if event.message:
                self._print_block("  Instruction", event.message)
            return

        if event.event_type == "human_manual_action_requested":
            self._section("HUMAN ACTION REQUIRED", self.RED)
            if event.message:
                self._print_block("  Instruction", event.message)
            return

        if event.event_type == "security_approval_requested":
            self._section("SECURITY APPROVAL REQUIRED", self.RED)
            if event.message:
                self._print_block("  Instruction", event.message)
            reason = str(event.data.get("reason", "")).strip()
            preview = str(event.data.get("preview", "")).strip()
            if reason:
                self._print_block("  Reason", reason)
            if preview:
                self._print_block("  Preview", preview)
            return

        if event.event_type == "human_input_received":
            self._section("HUMAN INPUT", self.CYAN)
            print("  Value received")
            return

        if event.event_type == "human_manual_action_confirmed":
            self._section("HUMAN ACTION", self.CYAN)
            print("  Manual browser action confirmed")
            return

        if event.event_type == "security_approval_received":
            self._section("SECURITY", self.CYAN)
            print("  Risky action approved")
            return

        if event.event_type == "security_action_rejected":
            self._section("SECURITY", self.YELLOW)
            print("  Risky action rejected; replanning")
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

    def request_human_action(self, request: HumanActionRequest) -> HumanActionResponse:
        print()
        self._section("ACTION NEEDED FROM YOU", self.RED)
        print(f"  Type: {request.kind}")
        self._print_block("  Instruction", request.instruction)

        if request.response_mode == "approval_confirmation":
            if request.reason:
                self._print_block("  Reason", request.reason)
            if request.preview:
                self._print_block("  Preview", request.preview)
            print("  Commands:")
            print("    approve  -> allow this risky action")
            print("    reject   -> do not perform it and replan")
            print("    abort    -> stop the run")
            while True:
                command = input("Command: ").strip().lower()
                if command == "approve":
                    return HumanActionResponse(action="approve")
                if command == "reject":
                    return HumanActionResponse(action="reject")
                if command == "abort":
                    return HumanActionResponse(action="abort")

        if request.response_mode == "provide_value":
            prompt = request.prompt or "Enter the requested value"
            if request.sensitive:
                print("  Commands:")
                print("    enter    -> input the value hidden")
                print("    /manual  -> do it yourself in the browser and then continue")
                print("    /abort   -> stop the run")
                while True:
                    command = input("Command: ").strip().lower()
                    if command == "/abort":
                        return HumanActionResponse(action="abort")
                    if command == "/manual":
                        self._print_manual_followup()
                        return self._await_manual_confirmation()
                    if command in {"", "enter"}:
                        secret_value = getpass.getpass(f"{prompt} (hidden): ").strip()
                        if secret_value:
                            return HumanActionResponse(action="provide_value", value=secret_value)
                return HumanActionResponse(action="abort")

            print("  Commands:")
            print("    /manual  -> do it yourself in the browser and then continue")
            print("    /abort   -> stop the run")
            while True:
                command = input(f"{prompt}: ").strip()
                if not command:
                    continue
                if command == "/abort":
                    return HumanActionResponse(action="abort")
                if command == "/manual":
                    self._print_manual_followup()
                    return self._await_manual_confirmation()
                return HumanActionResponse(action="provide_value", value=command)

        self._print_manual_followup()
        return self._await_manual_confirmation()

    def _print_block(self, title: str, text: str) -> None:
        print(f"{title}:")
        for line in str(text).splitlines():
            if line.strip():
                print(f"    {line}")
            else:
                print()

    def _print_manual_followup(self) -> None:
        print("  Complete the action manually in the browser, then type:")
        print("    done    -> continue")
        print("    abort   -> stop the run")

    def _await_manual_confirmation(self) -> HumanActionResponse:
        while True:
            command = input("Command: ").strip().lower()
            if command == "done":
                return HumanActionResponse(action="manual_done")
            if command == "abort":
                return HumanActionResponse(action="abort")

    def _line(self, text: str, *, color: str = "", bold: bool = False) -> str:
        if not self._use_color:
            return text

        prefix = ""
        if bold:
            prefix += self.BOLD
        prefix += color
        return f"{prefix}{text}{self.RESET}"
