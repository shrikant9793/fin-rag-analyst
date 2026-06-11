"""
scripts/test_retrieval.py
=========================
Quick smoke test — verifies Qdrant is reachable, the collection exists,
and hybrid retrieval returns results.

Usage:
    python scripts/test_retrieval.py
    python scripts/test_retrieval.py --query "What is Apple gross margin?"
    python scripts/test_retrieval.py --query "Revenue growth" --ticker AAPL --top-n 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.rag_pipeline import RAGPipeline

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test hybrid RAG retrieval")
    parser.add_argument(
        "--query",
        default="What are the key financial highlights and revenue figures?",
        help="Test query string",
    )
    parser.add_argument("--ticker", default=None, help="Filter by ticker symbol")
    parser.add_argument("--top-n",  type=int, default=5, help="Number of chunks to retrieve")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    console.rule("[bold cyan]RAG Retrieval Smoke Test[/bold cyan]")

    pipeline = RAGPipeline()

    # Check collection stats
    try:
        stats = pipeline.collection_stats()
        console.print(f"\n[green]✓ Qdrant connected[/green]")
        console.print(f"  Collection : {stats['collection']}")
        console.print(f"  Points     : {stats['points_count']}")
        console.print(f"  Status     : {stats['status']}\n")
    except Exception as exc:
        console.print(f"[red]✗ Cannot connect to Qdrant: {exc}[/red]")
        console.print("  → Is Qdrant running? Try: docker compose up qdrant -d")
        sys.exit(1)

    if stats["points_count"] == 0:
        console.print("[yellow]⚠ Collection is empty — run scripts/ingest.py first[/yellow]")
        sys.exit(0)

    # Run retrieval
    filters = {"ticker": args.ticker.upper()} if args.ticker else None
    console.print(Panel(f"[bold]{args.query}[/bold]", title="Query", border_style="cyan"))

    results = pipeline.retrieve(query=args.query, top_n=args.top_n, filters=filters)

    if not results:
        console.print("[yellow]No results returned[/yellow]")
        sys.exit(0)

    # Display results
    table = Table(title=f"Top {len(results)} Retrieved Chunks", show_lines=True)
    table.add_column("#",        width=3,  justify="right")
    table.add_column("RRF Score", width=10, justify="right")
    table.add_column("Ticker",   width=8)
    table.add_column("Doc Type", width=20)
    table.add_column("Date",     width=12)
    table.add_column("Content Preview", no_wrap=False)

    for i, chunk in enumerate(results, 1):
        table.add_row(
            str(i),
            f"{chunk.score:.4f}",
            chunk.metadata.get("ticker", "-"),
            chunk.metadata.get("doc_type", "-"),
            chunk.metadata.get("filing_date", "-"),
            chunk.content[:200].replace("\n", " ") + "…",
        )

    console.print(table)
    console.print(f"\n[green]✓ Retrieval test passed — {len(results)} chunks returned[/green]")


if __name__ == "__main__":
    main()