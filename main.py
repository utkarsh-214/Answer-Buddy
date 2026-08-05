"""
main.py
-------
Entry point for Answer-Buddy.

Usage:
  python main.py --ingest          # Parse PDFs and store in Qdrant (run once)
  python main.py --ingest --fresh  # Wipe existing data and re-ingest
  python main.py --query           # Start interactive Q&A session
  python main.py --query -q "..."  # Ask a single question and exit
"""

import argparse
import sys
import textwrap

from src.rag_pipeline import RAGPipeline, Answer


# ── Output formatting ─────────────────────────────────────────────────────────

DIVIDER = "─" * 70


def _format_answer(answer: Answer) -> str:
    """
    Formats an Answer dataclass into a human-readable string block.

    Produces output like:
    ──────────────────────────────────────────
    Question: What is the leave policy?

    Answer:
    Employees are entitled to 24 annual leave days.

    Sources:
    [1] employee_handbook.pdf | Page 17
        "All full-time employees receive 24 days..."
    ──────────────────────────────────────────
    """
    lines = [
        DIVIDER,
        f"Question: {answer.question}",
        "",
        "Answer:",
        textwrap.fill(answer.answer_text, width=70),
        "",
    ]

    if answer.found_in_docs and answer.citations:
        lines.append("Sources:")
        for i, citation in enumerate(answer.citations, start=1):
            lines.append(
                f"  [{i}] {citation.doc_name}  |  Page {citation.page_number}"
                f"  (relevance: {citation.score})"
            )
            # Wrap the snippet with indentation
            snippet_lines = textwrap.wrap(
                f'"{citation.snippet.strip()}"', width=66
            )
            for snippet_line in snippet_lines:
                lines.append(f"      {snippet_line}")
            lines.append("")

    lines.append(DIVIDER)
    return "\n".join(lines)


# ── CLI argument parsing ──────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="answer-buddy",
        description="Answer-Buddy: RAG-powered Q&A over PDF documents.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py --ingest
              python main.py --ingest --fresh
              python main.py --query
              python main.py --query -q "What is the leave policy?"
        """),
    )

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--ingest",
        action="store_true",
        help="Parse PDFs and ingest them into Qdrant.",
    )
    mode.add_argument(
        "--query",
        action="store_true",
        help="Start an interactive Q&A session.",
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="(Only with --ingest) Delete existing collection and re-ingest from scratch.",
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        default=None,
        help="(Only with --query) Ask a single question and exit.",
    )

    return parser


# ── Modes ─────────────────────────────────────────────────────────────────────

def run_ingest(pipeline: RAGPipeline, fresh: bool) -> None:
    """Run the ingestion pipeline."""
    try:
        pipeline.ingest(recreate=fresh)
    except FileNotFoundError as exc:
        print(f"\n❌ Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"\n❌ Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except EnvironmentError as exc:
        print(f"\n❌ Configuration error: {exc}", file=sys.stderr)
        sys.exit(1)


def run_query(pipeline: RAGPipeline, single_question: str | None) -> None:
    """Run the interactive Q&A session (or answer a single question)."""
    print("\n💬 Answer-Buddy — Document Q&A")
    print("Type your question and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        if single_question is not None:
            question = single_question.strip()
        else:
            try:
                question = input("❓ Question: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n\nGoodbye! 👋")
                break

        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("\nGoodbye! 👋")
            break

        try:
            answer = pipeline.query(question)
            print(_format_answer(answer))
        except EnvironmentError as exc:
            print(f"\n❌ Configuration error: {exc}", file=sys.stderr)
            sys.exit(1)
        except RuntimeError as exc:
            print(f"\n❌ Error while generating answer: {exc}", file=sys.stderr)

        # Exit after single question
        if single_question is not None:
            break


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = RAGPipeline()

    if args.ingest:
        run_ingest(pipeline, fresh=args.fresh)
    elif args.query:
        run_query(pipeline, single_question=args.question)


if __name__ == "__main__":
    main()
