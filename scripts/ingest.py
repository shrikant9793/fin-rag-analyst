"""
scripts/ingest.py
=================
CLI script for ingesting financial documents (SEC filings, earnings transcripts)
into Qdrant.

Usage:
    # Ingest all PDFs in a folder
    python scripts/ingest.py \
        --path data/raw/ \
        --ticker AAPL \
        --doc-type sec_filing \
        --filing-date 2024-10-31 \
        --company "Apple Inc."

    # Ingest a single file
    python scripts/ingest.py \
        --path data/raw/MSFT_earnings_Q4_2024.txt \
        --ticker MSFT \
        --doc-type earnings_transcript \
        --filing-date 2024-07-30 \
        --company "Microsoft Corporation"

    # Recreate collection from scratch
    python scripts/ingest.py --path data/raw/ --ticker AAPL --recreate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make project root importable when running as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loguru import logger
from rich.console import Console
from rich.table import Table

from src.rag_pipeline import RAGPipeline

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest financial documents into Qdrant vector DB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--path", required=True,
        help="File or folder path containing financial documents",
    )
    parser.add_argument("--ticker",       required=True,  help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("--doc-type",     required=True,
                        choices=["sec_filing", "earnings_transcript", "research_note"],
                        help="Document type classification")
    parser.add_argument("--filing-date",  required=True,  help="Filing date YYYY-MM-DD")
    parser.add_argument("--company",      default="",     help="Full company name")
    parser.add_argument("--recreate",     action="store_true",
                        help="Drop and recreate Qdrant collection before ingesting")
    parser.add_argument("--batch-size",   type=int, default=64, help="Upsert batch size")
    return parser.parse_args()


def collect_files(path_str: str, supported_exts: list[str]) -> list[Path]:
    """Collect all supported files from path (file or directory)."""
    p = Path(path_str)
    if p.is_file():
        return [p] if p.suffix.lower() in supported_exts else []
    if p.is_dir():
        files = []
        for ext in supported_exts:
            files.extend(p.glob(f"*{ext}"))
        return sorted(files)
    raise FileNotFoundError(f"Path does not exist: {p}")


def main() -> None:
    args = parse_args()

    console.rule("[bold cyan]Financial Document Ingestion[/bold cyan]")

    pipeline = RAGPipeline()
    supported_exts = pipeline.settings.ingestion_config["supported_extensions"]

    files = collect_files(args.path, supported_exts)
    if not files:
        console.print(f"[red]No supported files found at: {args.path}[/red]")
        console.print(f"Supported extensions: {supported_exts}")
        sys.exit(1)

    # Show files to be ingested
    table = Table(title="Files to Ingest", show_lines=True)
    table.add_column("File", style="cyan")
    table.add_column("Size (KB)", justify="right")
    for f in files:
        table.add_row(f.name, f"{f.stat().st_size / 1024:.1f}")
    console.print(table)

    doc_metadata = {
        "ticker":       args.ticker.upper(),
        "doc_type":     args.doc_type,
        "filing_date":  args.filing_date,
        "company_name": args.company,
    }
    console.print(f"\nMetadata: {doc_metadata}\n")

    if args.recreate:
        console.print("[yellow]⚠ Recreating collection — all existing data will be lost[/yellow]")
        pipeline.ensure_collection(recreate=True)
    else:
        pipeline.ensure_collection()

    total = pipeline.ingest_documents(
        file_paths=files,
        doc_metadata=doc_metadata,
        batch_size=args.batch_size,
    )

    stats = pipeline.collection_stats()
    console.print(f"\n[green]✓ Ingestion complete[/green]")
    console.print(f"  Total chunks ingested : [bold]{total}[/bold]")
    console.print(f"  Collection total points: [bold]{stats['points_count']}[/bold]")
    console.print(f"  Collection status      : [bold]{stats['status']}[/bold]")


if __name__ == "__main__":
    main()