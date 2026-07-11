"""Main entry point for PrivacyGuard Pipeline.

Provides:
- Async process() method for programmatic use.
- CLI mode: python main.py "text to process"
- Signal handling for graceful shutdown (SIGINT/Ctrl+C).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from privacyguard_pipeline.config import settings
from privacyguard_pipeline.exceptions import (
    PrivacyGuardError,
    TextTooLongError
)
from privacyguard_pipeline.pipeline import PrivacyGuardPipeline

# Configure UTF-8 for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

console = Console()
logger = logging.getLogger(__name__)

# Global pipeline reference for signal handler
_pipeline: PrivacyGuardPipeline | None = None


def _setup_logging() -> None:
    """Configure logging for the application."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                "logs/privacyguard.log",
                encoding="utf-8",
            ),
        ],
    )


def _handle_signal(sig: int, frame: object) -> None:
    """Handle SIGINT/Ctrl+C for graceful shutdown."""
    console.print("\n[yellow]Shutting down gracefully...[/yellow]")
    if _pipeline is not None:
        asyncio.create_task(_pipeline.close())
    sys.exit(0)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments using argparse.

    Args:
        argv: Optional argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed arguments namespace with 'text' and 'system_prompt' fields.
    """
    parser = argparse.ArgumentParser(
        description=(
            "PrivacyGuard Pipeline — PII anonymization"
            " for LLM API calls"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python main.py "Text with PII"\n'
            '  python main.py --system "Be concise" -- "Text with PII"\n'
            '  echo "Text with PII" | python main.py'
        ),
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to process (if omitted, reads from stdin)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Optional system prompt for the LLM",
    )
    args = parser.parse_args(argv)

    # If no text provided via args, try reading from stdin
    if not args.text:
        if not sys.stdin.isatty():
            stdin_text = sys.stdin.read().strip()
            if stdin_text:
                args.text = [stdin_text]

    if not args.text:
        parser.error(
            "No text provided. Pass text as argument or pipe via stdin.")

    args.text = " ".join(args.text)
    return args


def _print_result(result: dict[str, Any]) -> None:
    """Print pipeline result in a formatted way using Rich.

    Args:
        result: Pipeline result dictionary.
    """
    console.print(
        Panel(
            result["anonymized_text"][:500],
            title="[bold blue]Anonymized Text[/bold blue]",
            border_style="blue",
        ),
    )

    llm_response = result.get("llm_response", "")
    if llm_response:
        console.print(
            Panel(
                llm_response[:500],
                title="[bold green]LLM Response (Demasked)[/bold green]",
                border_style="green",
            ),
        )
    else:
        console.print(
            Panel(
                "[yellow]No LLM response (no API key or API error)[/yellow]",
                title="[bold yellow]LLM Response[/bold yellow]",
                border_style="yellow",
            ),
        )

    # Stats table
    stats = result.get("stats", {})
    if isinstance(stats, dict):
        table = Table(title="Session Statistics", border_style="cyan")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        for key, value in stats.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    table.add_row(f"  {sub_key}", str(sub_value))
            else:
                table.add_row(key.replace("_", " ").title(), str(value))

        console.print(table)


async def process(
    text: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Process text through the PrivacyGuard pipeline.

    This is the main public API method.

    Args:
        text: Input text that may contain PII.
        system_prompt: Optional system prompt for the LLM.

    Returns:
        Dictionary with anonymized_text, llm_response, and stats.

    Raises:
        TextTooLongError: If text exceeds maximum allowed length.
        PrivacyGuardError: On other pipeline errors.
    """
    global _pipeline

    _pipeline = PrivacyGuardPipeline()
    try:
        result = await _pipeline.process(text, system_prompt)
        return result
    except TextTooLongError:
        raise
    except PrivacyGuardError:
        raise
    except Exception as exc:
        raise PrivacyGuardError(f"Pipeline error: {exc}") from exc
    finally:
        await _pipeline.close()
        _pipeline = None


def main() -> None:
    """CLI entry point."""
    _setup_logging()

    # Register signal handlers
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, "SIGBREAK"):
        # type: ignore[attr-defined]
        signal.signal(signal.SIGBREAK, _handle_signal)

    # Parse CLI arguments
    args = _parse_args()

    console.print("[bold]PrivacyGuard Pipeline[/bold]")
    console.print(f"Processing text ({len(args.text)} chars)...\n")

    try:
        result = asyncio.run(process(args.text, args.system))
        _print_result(result)
    except TextTooLongError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    except PrivacyGuardError as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)


if __name__ == "__main__":
    main()
