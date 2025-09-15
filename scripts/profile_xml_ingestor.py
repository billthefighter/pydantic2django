"""
Profile the XML instance ingestion using PyInstrument.

Usage examples (from project root):

  - Using local schemas (pass absolute paths):
      poetry run pyinstrument scripts/profile_xml_ingestor.py \
        --schemas /ABS/PATH/TO/MTConnectDevices_2.5.xsd /ABS/PATH/TO/MTConnectStreams_1.7.xsd \
        --xml /ABS/PATH/TO/example.xml \
        --app-label tests

  - Auto-download MTConnect schemas (defaults to VERSION=1.7 to mirror user snippet):
      poetry run pyinstrument scripts/profile_xml_ingestor.py \
        --download-mtconnect --version 1.7 \
        --xml /ABS/PATH/TO/example.xml \
        --app-label tests

Notes:
- Passing --save will attempt to write instances to the DB. This requires concrete
  generated models to be imported by an installed app and migrated prior to run.
- The in-memory registration performed here is sufficient for instantiation (save=False).
"""

from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

BASE_SCHEMA_URL = "https://schemas.mtconnect.org/schemas/"
DEFAULT_VERSION = "1.7"
DEFAULT_SCHEMA_FILENAMES = [
    "MTConnectStreams_{VERSION}.xsd",
    "MTConnectDevices_{VERSION}.xsd",
    "MTConnectAssets_{VERSION}.xsd",
    "MTConnectError_{VERSION}.xsd",
]


def _ensure_django_ready(settings_module: str = "tests.settings") -> None:
    # Ensure project root is on sys.path so 'tests' and 'src' are importable
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    os.environ.setdefault("DJANGO_SETTINGS_MODULE", settings_module)
    import django  # Local import to avoid hard dependency for help text

    django.setup()


def _download_mtconnect_schemas(version: str, dest_dir: Path) -> list[str]:
    dest_dir.mkdir(parents=True, exist_ok=True)
    schema_paths: list[str] = []
    for template in DEFAULT_SCHEMA_FILENAMES:
        filename = template.format(VERSION=version)
        url = f"{BASE_SCHEMA_URL}{filename}"
        target = dest_dir / filename
        if not target.exists():
            try:
                urllib.request.urlretrieve(url, target.as_posix())
            except Exception as exc:
                print(f"Failed to download {url}: {exc}", file=sys.stderr)
                raise
        schema_paths.append(target.as_posix())
    return schema_paths


def _generate_models_in_memory(schema_files: list[str], app_label: str) -> None:
    # Generates and registers model classes into an in-memory registry
    # so the ingestor can resolve types without installing an app.
    from pydantic2django.xmlschema.generator import XmlSchemaDjangoModelGenerator

    generator = XmlSchemaDjangoModelGenerator(
        schema_files=[Path(p) for p in schema_files],
        app_label=app_label,
        output_path="__discard__.py",
        verbose=False,
    )
    # This builds carriers, registers classes into the xmlschema registry, and
    # returns the generated source text (unused here).
    _ = generator.generate_models_file()


def _run_ingestion(schema_files: list[str], xml_path: str, app_label: str, save: bool) -> None:
    from pydantic2django.xmlschema.ingestor import XmlInstanceIngestor

    ingestor = XmlInstanceIngestor(schema_files=[Path(p) for p in schema_files], app_label=app_label)
    instance = ingestor.ingest_from_file(xml_path, save=save)
    print(f"Root instance type: {type(instance).__name__}")
    print(f"Total created instances (incl. nested): {len(ingestor.created_instances)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile XML ingestion of a sample instance.")
    parser.add_argument("--xml", required=True, help="Absolute path to XML instance file")
    parser.add_argument(
        "--schemas",
        nargs="+",
        help="Absolute paths to one or more XSDs. Ignored when --download-mtconnect is used.",
    )
    parser.add_argument(
        "--download-mtconnect",
        action="store_true",
        help="Download MTConnect schemas (Streams/Devices/Assets/Error) from schemas.mtconnect.org",
    )
    parser.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help="MTConnect schema version string used with --download-mtconnect (e.g., 1.7 or 2.5)",
    )
    parser.add_argument("--app-label", default="tests", help="Target Django app label for generated models")
    parser.add_argument(
        "--save",
        action="store_true",
        help="Attempt to save created instances (requires real DB tables/migrations)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Number of times to run ingestion in a single process (default: 2)",
    )
    parser.add_argument(
        "--reuse-ingestor",
        action="store_true",
        help="Construct a single XmlInstanceIngestor and call ingest_from_file multiple times",
    )
    parser.add_argument(
        "--singleton",
        action="store_true",
        help="Use get_shared_ingestor/warmup for process-wide reuse (skips per-run generation)",
    )
    # Programmatic profiling mode (use this instead of wrapping with CLI 'pyinstrument')
    parser.add_argument(
        "--internal-profile",
        action="store_true",
        help="Profile inside this script using pyinstrument. Write per-run and/or combined HTML.",
    )
    parser.add_argument(
        "--timeline",
        action="store_true",
        help="When using --internal-profile, capture a timeline (preserve ordering, no condensing)",
    )
    parser.add_argument(
        "--per-run-html-dir",
        help="Directory to write one HTML profile per run (requires --internal-profile)",
    )
    parser.add_argument(
        "--combined-html-out",
        help="Write a single combined HTML profile across all runs (requires --internal-profile)",
    )
    args = parser.parse_args()

    # Optional: start a combined profiler BEFORE any setup so we capture Django
    # setup, schema warmup, and all ingestion runs in one timeline.
    early_combined_profiler = None
    if args.internal_profile and args.combined_html_out and not args.per_run_html_dir:
        from pyinstrument import Profiler  # local import

        early_combined_profiler = Profiler()
        early_combined_profiler.start()

    # Resolve schemas
    if args.download_mtconnect:
        # Store under tests/xmlschema/downloaded/<version>/ for visibility
        dest_dir = Path("tests/xmlschema/downloaded") / args.version
        schema_files = _download_mtconnect_schemas(args.version, dest_dir)
    else:
        if not args.schemas:
            parser.error("Either --schemas must be provided or use --download-mtconnect.")
        schema_files = [Path(p).as_posix() for p in args.schemas]

    # Boot Django
    _ensure_django_ready()

    # Prepare in-memory models for ingestion (ok for save=False paths)
    # Optionally pre-generate classes unless singleton mode is requested
    if not args.singleton:
        _generate_models_in_memory(schema_files=schema_files, app_label=args.app_label)

    # Helper to execute a single ingestion run given an ingestor factory
    def _execute_run(i: int, reuse: bool) -> None:
        if reuse:
            # Reused ingestor is closed over
            instance = shared_ingestor.ingest_from_file(args.xml, save=args.save)
            print(f"Run {i+1}: Root instance type: {type(instance).__name__}")
            print(f"Run {i+1}: Total created instances: {len(shared_ingestor.created_instances)}")
        else:
            _run_ingestion(schema_files=schema_files, xml_path=args.xml, app_label=args.app_label, save=args.save)

    # Internal profiling support
    if args.internal_profile:
        from pyinstrument import Profiler
        from pyinstrument.renderers import HTMLRenderer

        # Per-run HTML outputs
        per_run_dir = Path(args.per_run_html_dir) if args.per_run_html_dir else None
        if per_run_dir and args.combined_html_out:
            raise SystemExit("--per-run-html-dir and --combined-html-out are mutually exclusive. Pick one.")
        if per_run_dir:
            per_run_dir.mkdir(parents=True, exist_ok=True)

        # If reusing ingestor, build once
        shared_ingestor = None
        if args.reuse_ingestor:
            if args.singleton:
                from pydantic2django.xmlschema.ingestor import get_shared_ingestor

                shared_ingestor = get_shared_ingestor(
                    schema_files=[Path(p) for p in schema_files], app_label=args.app_label
                )
            else:
                from pydantic2django.xmlschema.ingestor import XmlInstanceIngestor

                shared_ingestor = XmlInstanceIngestor(
                    schema_files=[Path(p) for p in schema_files], app_label=args.app_label
                )

        # Combined profile across all runs if requested
        combined_profiler = None
        if args.combined_html_out and not per_run_dir:
            # If we started early (to include setup), reuse that session
            if early_combined_profiler is not None:
                combined_profiler = early_combined_profiler
            else:
                combined_profiler = Profiler()
                combined_profiler.start()

        for i in range(max(1, args.runs)):
            label = "reusing ingestor" if args.reuse_ingestor else "fresh ingestor"
            print(f"\n=== Ingestion run {i+1}/{args.runs} ({label}) ===")

            if per_run_dir:
                p = Profiler()
                p.start()
                if args.reuse_ingestor:
                    # mypy: shared_ingestor is set above in this branch
                    assert shared_ingestor is not None
                    instance = shared_ingestor.ingest_from_file(args.xml, save=args.save)
                    print(f"Run {i+1}: Root instance type: {type(instance).__name__}")
                    print(f"Run {i+1}: Total created instances: {len(shared_ingestor.created_instances)}")
                else:
                    _run_ingestion(
                        schema_files=schema_files, xml_path=args.xml, app_label=args.app_label, save=args.save
                    )
                p.stop()
                out_path = per_run_dir / f"profile_run_{i+1}.html"
                assert p.last_session is not None
                html = HTMLRenderer(timeline=args.timeline).render(p.last_session)
                out_path.write_text(html, encoding="utf-8")
                print(f"Wrote per-run profile: {out_path}")
            else:
                _execute_run(i, args.reuse_ingestor)

        if combined_profiler:
            combined_profiler.stop()
            assert combined_profiler.last_session is not None
            html = HTMLRenderer(timeline=args.timeline).render(combined_profiler.last_session)
            Path(args.combined_html_out).write_text(html, encoding="utf-8")
            print(f"Wrote combined profile: {args.combined_html_out}")

    else:
        # Ingest one or more times (expect external CLI pyinstrument to wrap the whole process)
        if args.reuse_ingestor:
            if args.singleton:
                from pydantic2django.xmlschema.ingestor import get_shared_ingestor

                shared_ingestor = get_shared_ingestor(
                    schema_files=[Path(p) for p in schema_files], app_label=args.app_label
                )
            else:
                from pydantic2django.xmlschema.ingestor import XmlInstanceIngestor

                shared_ingestor = XmlInstanceIngestor(
                    schema_files=[Path(p) for p in schema_files], app_label=args.app_label
                )
            for i in range(max(1, args.runs)):
                print(f"\n=== Ingestion run {i+1}/{args.runs} (reusing ingestor) ===")
                instance = shared_ingestor.ingest_from_file(args.xml, save=args.save)
                print(f"Run {i+1}: Root instance type: {type(instance).__name__}")
                print(f"Run {i+1}: Total created instances: {len(shared_ingestor.created_instances)}")
        else:
            for i in range(max(1, args.runs)):
                print(f"\n=== Ingestion run {i+1}/{args.runs} (fresh ingestor) ===")
                _run_ingestion(schema_files=schema_files, xml_path=args.xml, app_label=args.app_label, save=args.save)


if __name__ == "__main__":
    main()
