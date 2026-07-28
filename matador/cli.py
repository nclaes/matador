import logging
import sys
from pathlib import Path

import click
import yaml

from matador import __version__
from matador.backends.registry import list_backends

_DEFAULT_CONFIG = Path("/work/training_config.yaml")

_LOGGER = logging.getLogger(__name__)


@click.group(invoke_without_command=True)
@click.option("-v", "--verbose", is_eager=True, is_flag=True, default=False, help="Enable verbose logging.")
@click.pass_context
def main(ctx: click.Context, verbose: bool) -> None:
    """Matador — automated RTL Accelerator Generator for Tsetlin Machines."""
    if ctx.invoked_subcommand is None:
        from matador.splash import show_if_interactive
        if not show_if_interactive():
            click.echo(ctx.get_help())
        return
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)


@main.command("faena")
def faena() -> None:
    """Interactive guide: toolchain overview + next-step wizard."""
    _print_toolchain()

    has_model = click.confirm(
        "\n  Do you already have a trained Tsetlin Machine model (TMIR file)?",
        default=False,
    )

    if has_model:
        click.echo("")
        click.echo("  Skip straight to RTL generation. Pick a backend:")
        click.echo("    matador list-backends")
        click.echo("")
        click.echo("  Then, for the one you picked:")
        click.echo("    cp examples/<backend>.yaml /work/<backend>.yaml")
        click.echo("    matador generate --backend <backend> --config /work/<backend>.yaml")
        return

    has_boolean_data = click.confirm(
        "  Do you already have Boolean (0/1) training/test data files?",
        default=True,
    )

    if not has_boolean_data:
        has_raw_data = click.confirm(
            "  Do you have raw (non-Boolean) data ready to convert — locally or via a URL?",
            default=True,
        )

        if not has_raw_data:
            click.echo("")
            click.echo("  Step 0 — check if it's already registered:")
            click.echo("    matador list-datasets")
            click.echo("")
            click.echo("  Step 1 — fetch/materialize it:")
            click.echo("    matador ingest --dataset <name> --output-dir /work/raw   # if registered")
            click.echo("    -- or, if not registered --")
            click.echo("    cp examples/data_source_config.yaml /work/data_source_config.yaml")
            click.echo("    matador ingest --config /work/data_source_config.yaml --output-dir /work/raw")
            click.echo("    Outputs: /work/raw/<key>.npz")
            click.echo("")

        click.echo("  Step 2 — describe how to booleanize it:")
        click.echo("    If registered, inspect a recipe first (verified default, or an")
        click.echo("    unverified skeleton to edit if it doesn't have one yet):")
        click.echo("      matador booleanize --dataset <name> --show-recipe > /work/booleanisation_config.yaml")
        click.echo("    Otherwise, start from scratch:")
        click.echo("      cp examples/booleanisation_config.yaml /work/booleanisation_config.yaml")
        click.echo("    Either way: edit it freely, then point raw_npz at the ingest output above")
        click.echo("    (or your own x/y npz) if it isn't already set correctly.")
        click.echo("")
        click.echo("  Step 3 — booleanize:")
        click.echo("    matador booleanize --config /work/booleanisation_config.yaml")
        click.echo("    Outputs: <name>_train.txt / <name>_test.txt")
        click.echo("             (the exact format training_config.yaml's train_data/test_data expects)")
        click.echo("")

    click.echo("  Step 4 — edit the training config:")
    click.echo("    cp examples/training_config.yaml /work/training_config.yaml")
    if not has_boolean_data:
        click.echo("    (set train_data/test_data to the booleanize output above)")
    click.echo("")
    click.echo("  Step 5 — train:")
    click.echo("    matador train --config /work/training_config.yaml")
    click.echo("    Outputs: /work/TMIR/<model_name>/<model>.npz  +  .../validation_config.yaml")
    click.echo("             /work/TMIR/<model_name>/rom_inference.py  (standalone provenance script)")
    click.echo("    (<model_name> defaults to train_data's filename, e.g. digits_train.txt -> \"digits\")")
    click.echo("")

    if has_boolean_data and click.confirm("  Start training now using /work/training_config.yaml?", default=True):
        _run_training(_DEFAULT_CONFIG)


def _print_toolchain() -> None:
    """Print the Matador toolchain stages to stdout."""
    stages = [
        ("ingest",     "Fetch/materialize raw data (local or external)",
                       "data_source_config.yaml  →  raw arrays (npz/csv)"),
        ("booleanize", "Encode raw arrays into Boolean features",
                       "booleanisation_config.yaml  →  <name>_train.txt / _test.txt"),
        ("train",      "Train a TM model on Boolean feature data",
                       "training_config.yaml  →  TMIR (.npz / .yaml)"),
        ("validate",   "Check model accuracy on the held-out test set",
                       "TMIR + test data  →  accuracy report"),
        ("generate",   "Synthesise Verilog RTL (--backend NAME required)",
                       "TMIR + <backend>.yaml  →  <backend>/RTL/"),
        ("emulate",    "Cycle-accurate software emulator — no simulator needed",
                       "TMIR + config  →  InferenceTrace per sample"),
        ("simulate",   "Compile and run RTL testbenches",
                       "generated RTL  →  PASS / FAIL per testbench"),
        ("reprogram-suite", "Multi-model/dataset RTL testbenches (reprogrammable backends)",
                       "reprogram_config.yaml  →  tb_reprogram_suite.v + stimulus/expected .memh"),
        ("provenance", "ROM-based full-dataset inference report",
                       "TMIR + test data  →  JSON (fingerprint + accuracy)"),
    ]

    click.echo("")
    click.echo("  The Matador Toolchain")
    click.echo("  " + "─" * 60)
    for cmd, summary, detail in stages:
        label = f"matador {cmd}"
        pad   = " " * max(0, 24 - len(label))
        click.echo(f"  {click.style(label, bold=True)}{pad}  {summary}")
        click.echo(f"  {'':24}  {click.style(detail, dim=True)}")
        click.echo("")



@main.command("ingest")
@click.option(
    "--dataset",
    "dataset_name",
    default=None,
    help="Registered dataset key (see `matador list-datasets`). Alternative to --config.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to a data_source_config.yaml (standalone spec, or {catalog, key} pointer). "
         "Alternative to --dataset. Default: /work/data_source_config.yaml",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("/work/raw"),
    show_default=True,
    help="Where materialized raw arrays are written.",
)
def ingest(dataset_name: "str | None", config_path: "Path | None", output_dir: Path) -> None:
    """Fetch and materialize raw data — either a registered dataset
    (--dataset, see `matador list-datasets`) or a data_source_config.yaml
    (--config, for a dataset not in the shared catalog yet).

    \b
    Examples:
      matador ingest --dataset digits --output-dir /work/raw
      matador ingest --config /work/data_source_config.yaml --output-dir /work/raw

    \b
    Config template (for --config):
      cp examples/data_source_config.yaml /work/data_source_config.yaml
    """
    from matador.preprocessing import registry
    from matador.preprocessing.ingest import run_ingest
    from matador.preprocessing.sources import resolve_source_config

    if dataset_name and config_path:
        raise click.ClickException("Pass either --dataset or --config, not both.")
    if not dataset_name and not config_path:
        config_path = Path("/work/data_source_config.yaml")

    if dataset_name:
        try:
            spec, defaults = registry.get_dataset_and_defaults(dataset_name)
        except (KeyError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
    else:
        if not config_path.exists():
            raise click.ClickException(
                f"Config file not found: {config_path}\n"
                "  Create one based on examples/data_source_config.yaml, or use --dataset "
                "if this is already in the shared catalog (see `matador list-datasets`)."
            )
        try:
            spec, defaults = resolve_source_config(config_path)
        except Exception as exc:
            raise click.ClickException(f"Invalid config: {exc}") from exc

    click.echo(f"Ingesting {spec.key!r} ({spec.name or spec.key})")
    click.echo(f"  source: {spec.source.kind}")
    click.echo("")

    try:
        report = run_ingest(spec, output_dir, defaults=defaults)
    except Exception as exc:
        raise click.ClickException(f"Ingest failed: {exc}") from exc

    click.echo(f"  train samples: {report.x_train.shape[0]}  test samples: {report.x_test.shape[0]}")
    for w in report.warnings:
        click.echo(f"  warning: {w}")
    click.echo("")
    click.echo("Output:")
    for fmt, path in report.output_paths.items():
        click.echo(f"  {fmt}: {path}")
    click.echo("")
    click.echo("Next step — booleanize this into train/test files matador train can use:")
    if dataset_name:
        click.echo(f"  matador booleanize --dataset {dataset_name} --raw-dir {output_dir}")
    else:
        click.echo("  cp examples/booleanisation_config.yaml /work/booleanisation_config.yaml")
        click.echo("  matador booleanize --config /work/booleanisation_config.yaml")


def _format_encoder_spec(spec) -> str:
    """One-line human summary of a FeatureEncoderSpec, e.g.
    'thermometer(bits=8, range=[0.0, 16.0])'."""
    parts = []
    if spec.bits is not None:
        parts.append(f"bits={spec.bits}")
    if spec.range is not None:
        parts.append(f"range={list(spec.range)}")
    if spec.bins is not None:
        parts.append(f"bins={spec.bins}")
    if spec.quantile:
        parts.append("quantile=true")
    if spec.threshold is not None:
        parts.append(f"threshold={spec.threshold}")
    if spec.categories is not None:
        parts.append(f"categories={spec.categories}")
    inner = ", ".join(parts)
    return f"{spec.encoder}({inner})" if inner else spec.encoder


def _yaml_safe(obj):
    """Recursively convert tuples to lists — pydantic's tuple[float, float]
    (FeatureEncoderSpec.range) round-trips through model_dump() as a real
    tuple, which yaml.safe_dump can't represent."""
    if isinstance(obj, tuple):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_yaml_safe(v) for v in obj]
    return obj


@main.command("booleanize")
@click.option(
    "--dataset",
    "dataset_name",
    default=None,
    help="Registered dataset key with a default booleanization recipe. Alternative to --config.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to a booleanisation_config.yaml. Alternative to --dataset. "
         "Default: /work/booleanisation_config.yaml",
)
@click.option(
    "--raw-dir",
    "raw_dir",
    type=click.Path(path_type=Path),
    default=Path("/work/raw"),
    show_default=True,
    help="Only used with --dataset: directory containing <dataset>.npz (matador ingest's --output-dir).",
)
@click.option(
    "--output-dir",
    "output_dir",
    type=click.Path(path_type=Path),
    default=Path("/work/booleanised"),
    show_default=True,
    help="Only used with --dataset: where Boolean train/test files are written.",
)
@click.option(
    "--show-recipe",
    "show_recipe",
    is_flag=True,
    default=False,
    help="Print --dataset NAME's resolved booleanisation_config.yaml and exit, without running "
         "booleanization. Redirect to a file to inspect it or use it as a starting point for your own.",
)
def booleanize(
    dataset_name: "str | None", config_path: "Path | None",
    raw_dir: Path, output_dir: Path, show_recipe: bool,
) -> None:
    """Encode raw arrays into the Boolean train/test text files `matador
    train` consumes — either a registered dataset's default recipe
    (--dataset, run after `matador ingest --dataset ...`) or a fully custom
    booleanisation_config.yaml (--config).

    \b
    Examples:
      matador booleanize --dataset digits --raw-dir /work/raw
      matador booleanize --config /work/booleanisation_config.yaml
      matador booleanize --dataset digits --show-recipe > /work/booleanisation_config.yaml

    \b
    Config template (for --config, or to use a different encoding on a
    registered dataset than its default recipe):
      cp examples/booleanisation_config.yaml /work/booleanisation_config.yaml
    """
    from matador.config.schema import BooleanisationConfig
    from matador.preprocessing import registry
    from matador.preprocessing.booleanize import run_booleanize

    if dataset_name and config_path:
        raise click.ClickException("Pass either --dataset or --config, not both.")
    if show_recipe and not dataset_name:
        raise click.ClickException("--show-recipe requires --dataset NAME.")
    if not dataset_name and not config_path:
        config_path = Path("/work/booleanisation_config.yaml")

    if dataset_name:
        raw_npz = raw_dir / f"{dataset_name}.npz"

        if show_recipe:
            # Every registered dataset has SOMETHING to inspect and suggest
            # edits to here — a verified default if one exists, otherwise a
            # generic unvalidated skeleton to start from. Only the actual
            # (non---show-recipe) run path below requires a verified default.
            try:
                recipe, is_verified = registry.get_recipe_for_inspection(dataset_name)
            except (KeyError, FileNotFoundError) as exc:
                raise click.ClickException(str(exc)) from exc

            payload: dict = {"raw_npz": str(raw_npz), "name": dataset_name, "output_dir": str(output_dir)}
            if recipe.features:
                payload["features"] = [f.model_dump(exclude_none=True) for f in recipe.features]
            if recipe.default_encoder is not None:
                payload["default_encoder"] = recipe.default_encoder.model_dump(exclude_none=True)

            ds = registry.get_dataset(dataset_name)
            if is_verified:
                click.echo(f"# Verified default booleanization recipe for {dataset_name!r} ({ds.name or dataset_name}).")
                click.echo("# Reproduces the documented Boolean shape exactly (see data/Raw_Data_Bank.yaml).")
            else:
                click.echo(f"# UNVERIFIED starting skeleton for {dataset_name!r} ({ds.name or dataset_name}).")
                click.echo("# No catalog-recorded default exists for this dataset -- the encoder below is a")
                click.echo("# generic numeric guess (quantile-fit thermometer), not validated against any")
                click.echo("# known original encoding. Read the notes below before trusting it as-is:")
                if ds.notes:
                    for note_line in ds.notes.strip().splitlines():
                        click.echo(f"#   {note_line}")
            click.echo("#")
            click.echo("# Save as a booleanisation_config.yaml (edit freely — this is a starting")
            click.echo("# point you're free to suggest changes to, not a locked-in default), then:")
            click.echo("#   matador booleanize --config <path>")
            if not raw_npz.exists():
                click.echo(f"# NOTE: {raw_npz} does not exist yet — run this first:")
                click.echo(f"#   matador ingest --dataset {dataset_name} --output-dir {raw_dir}")
            click.echo(yaml.safe_dump(_yaml_safe(payload), sort_keys=False))
            return

        try:
            recipe = registry.get_default_booleanization(dataset_name)
        except (KeyError, FileNotFoundError) as exc:
            raise click.ClickException(str(exc)) from exc
        if recipe is None:
            raise click.ClickException(
                f"Dataset {dataset_name!r} has no VERIFIED default booleanization recipe "
                "(see its own notes in data/Raw_Data_Bank.yaml for why — usually a "
                "feature-engineering step this pipeline doesn't automate), so matador won't "
                "silently apply an unverified guess. Inspect a starting skeleton to edit and "
                "verify yourself:\n"
                f"  matador booleanize --dataset {dataset_name} --show-recipe > /work/booleanisation_config.yaml\n"
                "Then run it explicitly once you're happy with it:\n"
                "  matador booleanize --config /work/booleanisation_config.yaml"
            )

        if not raw_npz.exists():
            raise click.ClickException(
                f"{raw_npz} not found — run this first:\n"
                f"  matador ingest --dataset {dataset_name} --output-dir {raw_dir}"
            )
        config = BooleanisationConfig(
            raw_npz=raw_npz, name=dataset_name, output_dir=output_dir,
            features=recipe.features, default_encoder=recipe.default_encoder,
        )
    else:
        if not config_path.exists():
            raise click.ClickException(
                f"Config file not found: {config_path}\n"
                "  Create one based on examples/booleanisation_config.yaml, or use --dataset "
                "if this dataset has a default recipe (see `matador list-datasets`)."
            )
        try:
            raw = yaml.safe_load(config_path.read_text())
            config = BooleanisationConfig.model_validate(raw)
        except Exception as exc:
            raise click.ClickException(f"Invalid config: {exc}") from exc

    click.echo(f"Booleanizing {config.raw_npz} -> {config.name}_{{train,test}}.txt")
    click.echo("")

    try:
        report = run_booleanize(config)
    except Exception as exc:
        raise click.ClickException(f"Booleanization failed: {exc}") from exc

    click.echo(f"  train samples: {report.n_train}  test samples: {report.n_test}")
    click.echo(f"  {report.n_features_raw} raw features -> {report.n_features_bool} Boolean bits")
    click.echo("")
    click.echo("Output:")
    for name, path in report.output_paths.items():
        click.echo(f"  {name}: {path}")
    click.echo("")
    from matador.splash import _training_config_target
    existing_target = _training_config_target(_DEFAULT_CONFIG) if _DEFAULT_CONFIG.exists() else None

    if existing_target == config.name:
        # Same dataset re-booleanized -- the existing config already trains
        # (or is meant to train) this one, safe to keep pointing at it.
        target_cfg = _DEFAULT_CONFIG
        click.echo(f"Next step — edit {target_cfg} and set:")
    elif _DEFAULT_CONFIG.exists():
        # _DEFAULT_CONFIG already trains a DIFFERENT model -- don't suggest
        # silently repurposing it. A workspace can hold several training
        # configs, one per model, matching matador train's own per-model
        # TMIR/<model_name>/ output namespacing.
        target_cfg = _DEFAULT_CONFIG.parent / f"{config.name}_training_config.yaml"
        click.echo(f"Next step — {_DEFAULT_CONFIG} already trains {existing_target or 'a different model'};")
        click.echo(f"copy a separate config for this one:")
        click.echo(f"  cp examples/training_config.yaml {target_cfg}")
        click.echo(f"Edit {target_cfg} and set:")
    else:
        target_cfg = _DEFAULT_CONFIG
        click.echo("Next step — copy a training config template, then fill it out:")
        click.echo(f"  cp examples/training_config.yaml {target_cfg}")
        click.echo(f"Edit {target_cfg} and set:")
    click.echo(f"  train_data: {report.output_paths['train']}")
    click.echo(f"  test_data:  {report.output_paths['test']}")
    click.echo(f"  features:   {report.n_features_bool}")
    click.echo(f"  classes:    {report.n_classes}")
    click.echo("Also review the remaining hyperparameters (clauses, s, T, epochs,")
    click.echo("max_included_literals, seed) — the copied defaults are a starting point,")
    click.echo("not tuned for this dataset.")
    click.echo("")
    click.echo(f"  matador train --config {target_cfg}")


@main.command("train")
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_CONFIG,
    show_default=True,
    help="Path to training_config.yaml.",
)
def train(config_path: Path) -> None:
    """Train a Tsetlin Machine model from a YAML config file."""
    _run_training(config_path)


def _run_training(config_path: Path) -> None:
    from matador.config.schema import TrainingConfig
    from matador.models.trainer import export_tmir, load_data, train_model

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "  Create one based on examples/training_config.yaml and mount your\n"
            "  work directory: make shell WORK_DIR=/path/to/your/data"
        )

    try:
        raw = yaml.safe_load(config_path.read_text())
        config = TrainingConfig.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    click.echo(f"Config loaded from {config_path}")
    click.echo(f"  TM type   : {config.tm_type.value}")
    click.echo(f"  Clauses   : {config.clauses}")
    click.echo(f"  Classes   : {config.classes}")
    click.echo(f"  Features  : {config.features}")
    click.echo(f"  s / T     : {config.s} / {config.T}")
    click.echo(f"  Epochs    : {config.epochs}")
    click.echo("")

    click.echo("Loading data…")
    try:
        data = load_data(config)
    except Exception as exc:
        raise click.ClickException(f"Failed to load data: {exc}") from exc

    click.echo(
        f"  train samples: {len(data['x_train'])}  "
        f"test samples: {len(data['x_test'])}"
    )
    click.echo("")

    click.echo(f"Training for {config.epochs} epoch(s)…")
    tm = train_model(config, data)

    click.echo("")
    click.echo("Exporting TMIR…")
    yaml_out, npz_out, val_cfg_out = export_tmir(tm, config)
    tmir_dir = yaml_out.parent
    click.echo(f"Done. TMIR written to:\n  {yaml_out}\n  {npz_out}")
    click.echo(f"\nProvenance script (numpy-only, shareable):")
    click.echo(f"  python3 {tmir_dir}/rom_inference.py --test-data <path>")
    click.echo(f"\nTo validate the trained model run:")
    click.echo(f"  matador validate --config {val_cfg_out}")


_DEFAULT_VALIDATION_CONFIG = Path("/work/validation_config.yaml")


@main.command("validate")
@click.option(
    "--backend", "backend_name",
    default="vanilla_tiled",
    show_default=True,
    type=click.Choice(list_backends()),
    help="Backend (used with --mode rtl to locate the generated RTL directory).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=_DEFAULT_VALIDATION_CONFIG,
    show_default=True,
    help="Path to validation_config.yaml.",
)
@click.option(
    "--mode",
    type=click.Choice(["model", "rtl"]),
    default="model",
    show_default=True,
    help="model: software validation against TMIR; rtl: validate synthesised RTL via simulation.",
)
def validate(backend_name: str, config_path: Path, mode: str) -> None:
    """Validate a TMIR model or generated RTL."""
    from matador.models.validator import validate_model, validate_rtl

    if not config_path.exists():
        if mode == "rtl":
            raise click.ClickException(
                f"Config file not found: {config_path}\n"
                "  Pass your accelerator_config.yaml:\n"
                "    matador validate --mode rtl --config /path/to/accelerator_config.yaml"
            )
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            "  Create a validation_config.yaml with at minimum:\n"
            "    model_path: /path/to/model.yaml\n"
            "  Optionally add:\n"
            "    test_data:  /path/to/test.txt"
        )

    try:
        raw = yaml.safe_load(config_path.read_text())
    except Exception as exc:
        raise click.ClickException(f"Failed to read config: {exc}") from exc

    if mode == "rtl":
        from matador.backends.registry import get as get_backend
        backend = get_backend(backend_name)()
        try:
            config = backend.config_class.model_validate(raw)
        except Exception as exc:
            raise click.ClickException(f"Invalid accelerator config: {exc}") from exc

        # RTL lives under <output_dir>/<backend>/
        config = config.model_copy(update={"output_dir": config.output_dir / backend_name})
        click.echo(f"Backend       : {backend_name}")
        click.echo(f"RTL directory : {config.output_dir / 'RTL'}")
        click.echo("")

        try:
            report = validate_rtl(config)
        except Exception as exc:
            raise click.ClickException(str(exc)) from exc

        for r in report.results:
            label = click.style("PASS", fg="green") if r.passed else click.style("FAIL", fg="red")
            click.echo(f"  {r.name:<20}  {label}   {r.summary}")
            for line in r.failures:
                click.echo(f"      {line}")

        click.echo("")
        click.echo(f"{report.n_passed} passed, {report.n_failed} failed")
        click.echo("")
        if report.all_passed:
            click.echo(click.style("RTL validation passed.", fg="green"))
        else:
            click.echo(click.style("RTL validation FAILED.", fg="red"))
            raise SystemExit(1)
        return

    from matador.config.schema import ValidationConfig
    try:
        config = ValidationConfig.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    click.echo(f"Validating model: {config.model_path}")
    click.echo("")

    try:
        report = validate_model(config)
    except Exception as exc:
        raise click.ClickException(f"Failed to load model: {exc}") from exc

    # ── Schema ────────────────────────────────────────────────────────────────
    schema_label = click.style("PASS", fg="green") if report.schema_ok else click.style("FAIL", fg="red")
    click.echo(f"  Schema validation   {schema_label}")
    if not report.schema_ok:
        click.echo(f"    {report.schema_error}")

    # ── Test vectors ──────────────────────────────────────────────────────────
    if report.vectors_ran:
        v_ok = report.vectors_passed == report.vectors_total
        v_label = click.style("PASS", fg="green") if v_ok else click.style("FAIL", fg="red")
        click.echo(f"  Test vectors        {v_label}  ({report.vectors_passed}/{report.vectors_total})")

    # ── Accuracy ──────────────────────────────────────────────────────────────
    if report.accuracy_pct is not None:
        click.echo(
            f"  Test accuracy       {report.accuracy_pct:.2f}%"
            f"  ({report.accuracy_correct}/{report.accuracy_total} correct)"
        )

    click.echo("")
    if report.all_passed:
        click.echo(click.style("Validation passed.", fg="green"))
    else:
        click.echo(click.style("Validation FAILED.", fg="red"))
        raise SystemExit(1)


@main.command("generate")
@click.option(
    "--backend", "backend_name",
    required=True,
    type=click.Choice(list_backends()),
    help=(
        "Accelerator architecture to generate.  "
        "Each backend uses its own config file: /work/vanilla_tiled.yaml or /work/vanilla_hardwired.yaml."
    ),
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to backend config YAML.  Default: /work/<backend>.yaml",
)
def generate(backend_name: str, config_path: Path | None) -> None:
    """Generate RTL for a TM accelerator backend.

    Each backend requires its own dedicated config file.

    \b
    Examples:
      matador generate --backend vanilla_tiled     --config /work/vanilla_tiled.yaml
      matador generate --backend vanilla_hardwired --config /work/vanilla_hardwired.yaml

    \b
    Config templates:
      cp examples/vanilla_tiled.yaml     /work/vanilla_tiled.yaml
      cp examples/vanilla_hardwired.yaml /work/vanilla_hardwired.yaml
    """
    from matador.backends.registry import get as get_backend
    from matador.ir.tm_ir import TMIR

    # Default config name is /work/<backend>.yaml
    if config_path is None:
        config_path = Path(f"/work/{backend_name}.yaml")

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            f"  Copy the template:  cp examples/{backend_name}.yaml {config_path}"
        )

    try:
        raw = yaml.safe_load(config_path.read_text())
    except Exception as exc:
        raise click.ClickException(f"Failed to read config: {exc}") from exc

    targets = [backend_name]

    backend    = get_backend(backend_name)()
    base_config = None
    try:
        base_config = backend.config_class.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    click.echo(f"Loading model: {base_config.model_path}")
    try:
        suffix = base_config.model_path.suffix.lower()
        tmir   = TMIR.from_yaml(base_config.model_path) if suffix in {".yaml", ".yml"} \
                 else TMIR.from_npz(base_config.model_path)
    except Exception as exc:
        raise click.ClickException(f"Failed to load TMIR: {exc}") from exc

    click.echo(
        f"  {tmir.variant} TM  |  "
        f"{tmir.architecture.n_features} features  |  "
        f"{tmir.architecture.n_classes} classes  |  "
        f"{tmir.architecture.n_clauses_total} clauses"
    )
    click.echo(f"  Generating [{backend_name}]…")
    click.echo("")

    # Backend generates into its own named subdirectory: <output_dir>/<backend>/RTL/
    namespaced_config = base_config.model_copy(
        update={"output_dir": base_config.output_dir / backend_name}
    )

    try:
        artifacts = backend.generate(tmir, namespaced_config)
    except NotImplementedError as exc:
        raise click.ClickException(str(exc)) from exc
    except Exception as exc:
        raise click.ClickException(f"RTL generation failed: {exc}") from exc

    rtl_dir = artifacts.rtl_dir
    click.echo(f"RTL written to: {rtl_dir}")
    click.echo("")
    click.echo("Next steps:")
    click.echo(f"  matador simulate --backend {backend_name} --config {config_path}")
    click.echo(f"  matador emulate  --backend {backend_name} --config {config_path} --verify")


@main.command("simulate")
@click.option(
    "--backend", "backend_name",
    required=True,
    type=click.Choice(list_backends()),
    help="Backend to simulate.  Default config: /work/<backend>.yaml",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to backend config YAML.  Default: /work/<backend>.yaml",
)
@click.option(
    "--sim",
    type=click.Choice(["verilator", "iverilog", "auto"]),
    default="auto",
    show_default=True,
    help="Simulator to use. auto: Verilator if available, else iverilog.",
)
@click.option("--tb", default=None, help="Run only this testbench (e.g. tb_system).")
@click.option("--waves", "open_waves", is_flag=True, default=False,
              help="Open GTKWave after simulation.")
def simulate(backend_name: str, config_path: Path, sim: str, tb: str, open_waves: bool) -> None:
    """Compile and run RTL testbenches; optionally open GTKWave."""
    from matador.backends.registry import get as get_backend
    from matador.models.validator import validate_rtl

    if config_path is None:
        config_path = Path(f"/work/{backend_name}.yaml")

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            f"  Run 'matador generate --backend {backend_name}' first."
        )

    backend = get_backend(backend_name)()
    try:
        raw    = yaml.safe_load(config_path.read_text())
        config = backend.config_class.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    # RTL lives under <output_dir>/<backend>/
    config = config.model_copy(update={"output_dir": config.output_dir / backend_name})

    click.echo("Running RTL simulation…")
    click.echo("")

    try:
        report = validate_rtl(config)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    for r in report.results:
        if tb and r.name != tb and r.name != f"tb_{tb}":
            continue
        label = click.style("PASS", fg="green") if r.passed else click.style("FAIL", fg="red")
        click.echo(f"  {r.name:<25}  {label}   {r.summary}")
        for line in r.failures:
            click.echo(f"      {line}")

    click.echo("")
    click.echo(f"{report.n_passed} passed, {report.n_failed} failed")
    click.echo("")

    if open_waves:
        import subprocess
        sim_dir  = config.output_dir / "RTL" / "sim"
        waves_sh = sim_dir / "waves.sh"
        tb_name  = tb or "tb_system"
        if waves_sh.exists():
            click.echo(f"Opening GTKWave for {tb_name}…")
            subprocess.Popen(["bash", str(waves_sh), tb_name])
        else:
            click.echo(f"waves.sh not found at {waves_sh} — re-run 'matador generate'")

    if not report.all_passed:
        raise SystemExit(1)


@main.command("reprogram-suite")
@click.option(
    "--backend", "backend_name",
    required=True,
    type=click.Choice(list_backends()),
    help="Backend to target — must support runtime reprogramming (currently: vanilla_gp_tiled).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to the backend's own accelerator config YAML (same file used for "
         "`matador generate`/`matador simulate`).  Default: /work/<backend>.yaml",
)
@click.option(
    "--reprogram-config",
    "reprogram_config_path",
    type=click.Path(path_type=Path),
    default=Path("/work/reprogram_config.yaml"),
    show_default=True,
    help="Path to a reprogram_config.yaml (the ordered {model, dataset} step list).",
)
def reprogram_suite(backend_name: str, config_path: "Path | None", reprogram_config_path: Path) -> None:
    """Build a testbench + stimulus that reprograms an already-generated
    bundle across multiple models/datasets in one continuous run — the
    concrete, checkable proof that a runtime-reprogrammable backend really
    does reprogram with YOUR models, not just one at a time.

    \b
    Requires `matador generate` to have already run for --backend/--config.

    \b
    Examples:
      matador reprogram-suite --backend vanilla_gp_tiled \\
          --config /work/vanilla_gp_tiled.yaml \\
          --reprogram-config /work/reprogram_config.yaml

    \b
    Config template:
      cp examples/reprogram_config.yaml /work/reprogram_config.yaml
    """
    from matador.backends.base import ReprogramStep
    from matador.backends.gp_tiled.reprogram_config import ReprogramSuiteConfig
    from matador.backends.registry import get as get_backend

    if config_path is None:
        config_path = Path(f"/work/{backend_name}.yaml")

    backend = get_backend(backend_name)()
    if not backend.supports_reprogramming:
        supported = [n for n in list_backends() if get_backend(n)().supports_reprogramming]
        raise click.ClickException(
            f"Backend {backend_name!r} does not support runtime reprogramming. "
            f"Backends that do: {', '.join(supported) or '(none registered)'}"
        )

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            f"  This is the same config used for 'matador generate --backend {backend_name}'."
        )
    if not reprogram_config_path.exists():
        raise click.ClickException(
            f"Config file not found: {reprogram_config_path}\n"
            "  Create one based on examples/reprogram_config.yaml."
        )

    try:
        raw = yaml.safe_load(config_path.read_text())
        config = backend.config_class.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc
    config = config.model_copy(update={"output_dir": config.output_dir / backend_name})

    try:
        raw_reprogram = yaml.safe_load(reprogram_config_path.read_text())
        reprogram_config = ReprogramSuiteConfig.model_validate(raw_reprogram)
    except Exception as exc:
        raise click.ClickException(f"Invalid reprogram config: {exc}") from exc

    rtl_dir = config.output_dir / "RTL"
    if not rtl_dir.exists():
        raise click.ClickException(
            f"{rtl_dir} does not exist — run 'matador generate --backend {backend_name} "
            f"--config {config_path}' first."
        )

    steps = [
        ReprogramStep(
            tmir_path=s.model, dataset_path=s.dataset, vectors_path=s.vectors,
            n_samples=s.n_samples, seed=s.seed, name=s.name,
        )
        for s in reprogram_config.steps
    ]

    click.echo(f"Building reprogram suite for {backend_name} at {rtl_dir} ({len(steps)} step(s))")
    click.echo("")

    try:
        artifacts = backend.build_reprogram_suite(rtl_dir, steps, config)
    except Exception as exc:
        raise click.ClickException(f"Reprogram-suite build failed: {exc}") from exc

    click.echo("Wrote:")
    for p in [*artifacts.testbenches, *artifacts.sim_scripts]:
        click.echo(f"  {p}")
    click.echo("")
    sim_dir = rtl_dir / "sim"
    tb_name = artifacts.testbenches[0].stem
    src_dir = rtl_dir / "src"
    srcs = " ".join(str(src_dir / f) for f in (
        "axis_fifo.v", "clause_eval.v", "tile_mem.v", "score_acc_rt.v", "argmax_rt.v", "tm_accel_gp.v",
    ))
    out_bin = sim_dir / tb_name
    click.echo("Next step — compile and run it:")
    click.echo(f"  iverilog -g2001 -Wall -Wno-timescale -o {out_bin} {srcs} {artifacts.testbenches[0]} && "
               f"(cd {sim_dir} && vvp {out_bin})")


@main.command("waves")
@click.option(
    "--backend", "backend_name",
    required=True,
    type=click.Choice(list_backends()),
    help="Backend whose RTL bundle to open waveforms for.  Default config: /work/<backend>.yaml",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to backend config YAML.  Default: /work/<backend>.yaml",
)
@click.option("--tb", default="tb_system", show_default=True,
              help="Which testbench waveform to open.")
def waves(backend_name: str, config_path: "Path | None", tb: str) -> None:
    """Open GTKWave for a simulated testbench waveform."""
    import subprocess

    from matador.backends.registry import get as get_backend

    if config_path is None:
        config_path = Path(f"/work/{backend_name}.yaml")

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            f"  Run 'matador generate --backend {backend_name}' first."
        )

    backend = get_backend(backend_name)()
    try:
        raw    = yaml.safe_load(config_path.read_text())
        config = backend.config_class.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    # RTL lives under <output_dir>/<backend>/ — same namespacing generate/simulate use.
    config = config.model_copy(update={"output_dir": config.output_dir / backend_name})

    sim_dir  = config.output_dir / "RTL" / "sim"
    waves_sh = sim_dir / "waves.sh"

    if not waves_sh.exists():
        raise click.ClickException(
            f"waves.sh not found at {waves_sh}\n"
            f"  Run 'matador generate --backend {backend_name}' first, then simulate to produce a waveform."
        )

    click.echo(f"Opening GTKWave for {tb}…")
    result = subprocess.run(["bash", str(waves_sh), tb])
    if result.returncode != 0:
        raise SystemExit(result.returncode)


@main.command("emulate")
@click.option(
    "--backend", "backend_name",
    required=True,
    type=click.Choice(list_backends()),
    help="Backend to emulate.  Default config: /work/<backend>.yaml",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to backend config YAML.  Default: /work/<backend>.yaml",
)
@click.option("--trace", "trace_path", type=click.Path(path_type=Path),
              default=None, help="Write InferenceTrace JSON to this path.")
@click.option("--verbose", is_flag=True, default=False,
              help="Print per-vector trace summary to stdout.")
@click.option("--verify", is_flag=True, default=False,
              help="Also cross-check emulator against reference inference.")
def emulate(backend_name: str, config_path: Path, trace_path: Path, verbose: bool, verify: bool) -> None:
    """Run the software emulator on embedded TMIR test vectors."""
    import json
    from matador.backends.registry import get as get_backend
    from matador.ir.tm_ir import TMIR

    if config_path is None:
        config_path = Path(f"/work/{backend_name}.yaml")

    if not config_path.exists():
        raise click.ClickException(
            f"Config file not found: {config_path}\n"
            f"  Run 'matador generate --backend {backend_name}' first or point to a config YAML."
        )

    backend = get_backend(backend_name)()
    try:
        raw    = yaml.safe_load(config_path.read_text())
        config = backend.config_class.model_validate(raw)
    except Exception as exc:
        raise click.ClickException(f"Invalid config: {exc}") from exc

    # Model lives in the namespaced output dir written by generate
    config = config.model_copy(update={"output_dir": config.output_dir / backend_name})

    click.echo(f"Loading model: {config.model_path}")
    try:
        suffix = config.model_path.suffix.lower()
        tmir   = TMIR.from_yaml(config.model_path) if suffix in {".yaml", ".yml"} else TMIR.from_npz(config.model_path)
    except Exception as exc:
        raise click.ClickException(f"Failed to load TMIR: {exc}") from exc

    if not (tmir.verification and tmir.verification.test_vectors):
        raise click.ClickException(
            "No test vectors embedded in TMIR.\n"
            "  Re-train with 'matador train' — test vectors are embedded automatically."
        )

    vectors = tmir.verification.test_vectors
    click.echo(f"Running emulator on {len(vectors)} test vector(s)…")
    click.echo("")

    emulator_cls = backend.emulator_class
    if emulator_cls is None:
        raise click.ClickException(
            f"The '{backend_name}' backend does not have a software emulator."
        )
    emul = emulator_cls(tmir, config)

    all_traces = []
    fail_cnt   = 0
    pass_cnt   = 0

    for idx, vec in enumerate(vectors):
        trace = emul.run(vec.input)
        predicted = trace.predicted_class
        expected  = vec.expected_class
        correct   = (predicted == expected)

        if correct:
            label = click.style("PASS", fg="green")
            pass_cnt += 1
        else:
            label = click.style("FAIL", fg="red")
            fail_cnt += 1

        argmax  = getattr(trace, "argmax_event", None)
        scores_str = ", ".join(str(s) for s in (argmax.scores if argmax else []))
        click.echo(f"  [{idx:3d}] {label}  predicted={predicted}  expected={expected}  scores=[{scores_str}]")

        if verbose:
            # Tiled-specific counters
            if hasattr(trace, "fsm_events"):
                click.echo(f"         FSM transitions: {len(trace.fsm_events)}")
            if hasattr(trace, "rom_events"):
                click.echo(f"         ROM reads:        {len(trace.rom_events)}")
            if hasattr(trace, "clause_partial_events"):
                click.echo(f"         Clause evals:     {len(trace.clause_partial_events)}")
            if hasattr(trace, "score_vote_events"):
                click.echo(f"         Score votes:      {len(trace.score_vote_events)}")
            # Hardwired-specific counters
            if hasattr(trace, "clause_eval_events"):
                click.echo(f"         Clause evals:     {len(trace.clause_eval_events)}")
            if hasattr(trace, "pipeline_stages_waited"):
                click.echo(f"         Pipeline stages:  {trace.pipeline_stages_waited}")

        all_traces.append(trace)

    click.echo("")
    click.echo(f"{pass_cnt} passed, {fail_cnt} failed")

    if verify:
        click.echo("")
        click.echo("Cross-checking emulator against reference inference…")
        from matador.verification.compare import compare_emulator_to_reference
        report = compare_emulator_to_reference(tmir, config, emulator_cls=emulator_cls)
        if report.all_passed:
            click.echo(click.style(
                f"  Emulator matches reference on all {report.n_vectors} vector(s).", fg="green"
            ))
        else:
            click.echo(click.style(
                f"  Emulator/reference MISMATCH on {report.n_failed}/{report.n_vectors} vector(s).",
                fg="red",
            ))
            first = report.first_failure()
            if first:
                click.echo(f"  First mismatch: vector {first.index}")
                click.echo(f"    emulator  class={first.emulator_class}  scores={first.emulator_scores}")
                click.echo(f"    reference class={first.reference_class} scores={first.reference_scores}")
            fail_cnt += report.n_failed

    if trace_path is not None:
        import dataclasses
        def _serialise(obj):
            if dataclasses.is_dataclass(obj):
                return dataclasses.asdict(obj)
            if isinstance(obj, list):
                return [_serialise(x) for x in obj]
            return obj

        records = [_serialise(t) for t in all_traces]
        trace_path.write_text(json.dumps(records, indent=2))
        click.echo(f"Trace written to: {trace_path}")

    click.echo("")
    if fail_cnt == 0:
        click.echo(click.style("Emulation passed.", fg="green"))
    else:
        click.echo(click.style("Emulation FAILED.", fg="red"))
        raise SystemExit(1)


@main.command("provenance")
@click.option(
    "--model",
    "model_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to TMIR model file (.yaml or .npz).",
)
@click.option(
    "--test-data",
    "test_data_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to space-separated Boolean test dataset (last column = label).",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=None,
    help="Write JSON report to this path (default: <model_dir>/provenance_report.json).",
)
def provenance(model_path: Path, test_data_path: Path, output_path: Path) -> None:
    """Run full-dataset ROM-based inference and write a provenance report.

    Loads the Include-action matrix from the TMIR file (identical to what
    the RTL tile ROM stores) and runs inference through the pure-NumPy
    reference engine.  No tmu C extension required.

    The JSON report includes: model fingerprint, dataset SHA-256, accuracy,
    per-class accuracy, and confusion matrix — suitable for sharing with
    hardware collaborators as a golden reference.
    """
    from matador.inference.provenance import generate

    if not model_path.exists():
        raise click.ClickException(f"Model file not found: {model_path}")
    if not test_data_path.exists():
        raise click.ClickException(f"Test dataset not found: {test_data_path}")

    if output_path is None:
        output_path = model_path.parent / "provenance_report.json"

    click.echo(f"Model      : {model_path}")
    click.echo(f"Test data  : {test_data_path}")
    click.echo(f"Inference  : ROM-equivalent (pure NumPy, matador.inference.reference)")
    click.echo("")

    try:
        report = generate(model_path, test_data_path)
    except Exception as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo(f"  Samples  : {report.n_samples}")
    click.echo(f"  Features : {report.n_features}")
    click.echo(f"  Classes  : {report.n_classes}")
    click.echo(f"  Clauses  : {report.n_clauses_total}")
    click.echo("")
    click.echo(f"  Accuracy : {report.accuracy_pct:.2f}%"
               f"  ({report.accuracy_correct}/{report.accuracy_total})")
    click.echo("")
    click.echo("  Per-class accuracy:")
    for cls, res in report.per_class.items():
        click.echo(f"    class {cls:>2}  {res['accuracy_pct']:6.2f}%"
                   f"  ({res['correct']}/{res['total']})")
    click.echo("")
    click.echo(f"  Model fingerprint : {report.model_fingerprint}")
    click.echo(f"  Dataset SHA-256   : {report.test_data_sha256}")
    click.echo("")

    report.write(output_path)
    click.echo(f"Provenance report written to:\n  {output_path}")


@main.command("status")
def status() -> None:
    """Show workspace status and available next actions.

    Reads /work/ and reports which pipeline stages are complete,
    then lists every command that is applicable right now.

    Works in non-interactive mode (scripts, CI, piped output).
    """
    from matador.splash import _dm, _workspace_state
    state = _workspace_state()
    for line in _dm(state):
        # Strip ANSI codes when output is not a TTY
        if not sys.stdout.isatty():
            import re as _re
            line = _re.sub(r"\033\[[0-9;]*m", "", line)
        click.echo(line)


# Directories matador itself writes into a work dir at their default paths.
# Backend RTL dirs (named after each registry.list_backends() entry) are
# added dynamically in _clean_targets() -- not hardcoded here, so a new
# backend is covered automatically.
_GENERATED_DIRS = ("TMIR", "raw", "booleanised", "_cache", "_extracted")

# Config YAMLs a user typically authors by hand (cp'd from examples/ and
# edited) -- kept by default even when cleaning generated output, since
# removing someone's edited recipe is a different, bigger ask than removing
# what matador derived from it. <backend>.yaml entries are added dynamically.
_CONFIG_FILENAMES = (
    "training_config.yaml", "data_source_config.yaml", "booleanisation_config.yaml",
    "reprogram_config.yaml", "validation_config.yaml",
)


def _clean_targets(work_dir: Path, include_configs: bool) -> list[Path]:
    from matador.backends.registry import list_backends

    targets = [work_dir / d for d in _GENERATED_DIRS if (work_dir / d).exists()]
    targets += [work_dir / bk for bk in list_backends() if (work_dir / bk).exists()]
    if include_configs:
        names = list(_CONFIG_FILENAMES) + [f"{bk}.yaml" for bk in list_backends()]
        targets += [work_dir / n for n in names if (work_dir / n).exists()]
    return sorted(set(targets))


def _refuse_if_dangerous(work_dir: Path) -> None:
    import matador

    resolved = work_dir.resolve()
    dangerous = {Path("/"), Path.home(), Path(matador.__file__).resolve().parent.parent}
    if resolved in dangerous:
        raise click.ClickException(
            f"Refusing to clean {resolved} — this looks like a root, home, or the matador "
            "repo's own directory, not a work directory. Pass --work-dir explicitly if this "
            "is really where you keep matador's generated output."
        )


@main.command("clean")
@click.option(
    "--work-dir",
    type=click.Path(path_type=Path),
    default=Path("/work"),
    show_default=True,
    help="Work directory to clean.",
)
@click.option(
    "--configs", "include_configs", is_flag=True, default=False,
    help="Also remove config YAMLs you authored (training_config.yaml, <backend>.yaml, etc.), "
         "not just generated output.",
)
@click.option(
    "--all", "clean_all", is_flag=True, default=False,
    help="Remove EVERYTHING directly under --work-dir, including files matador doesn't "
         "recognize. Use when you want a genuinely empty work directory, not just a reset "
         "of matador's own output.",
)
@click.option("--dry-run", is_flag=True, default=False, help="Show what would be removed; delete nothing.")
@click.option("-y", "--yes", is_flag=True, default=False, help="Skip the confirmation prompt.")
def clean(work_dir: Path, include_configs: bool, clean_all: bool, dry_run: bool, yes: bool) -> None:
    """Remove generated output from a work directory so you can start fresh.

    By default, only removes directories matador itself writes (TMIR/, raw/,
    booleanised/, per-backend RTL/, ingest's internal cache dirs) — config
    YAMLs you edited by hand are left alone. Always previews what will be
    removed and asks for confirmation unless --yes is given.

    \b
    Examples:
      matador clean --dry-run                # preview only, nothing removed
      matador clean                          # remove generated output, keep your configs
      matador clean --configs                # also remove your config YAMLs
      matador clean --all -y                 # wipe --work-dir entirely, no prompt
    """
    import shutil

    _refuse_if_dangerous(work_dir)

    if not work_dir.exists():
        click.echo(f"{work_dir} does not exist — nothing to clean.")
        return

    if clean_all:
        targets = sorted(work_dir.iterdir())
    else:
        targets = _clean_targets(work_dir, include_configs)

    if not targets:
        click.echo(f"Nothing to clean in {work_dir}.")
        return

    click.echo(f"The following will be removed from {work_dir}:")
    for t in targets:
        kind = "dir " if t.is_dir() else "file"
        click.echo(f"  [{kind}] {t.relative_to(work_dir)}")
    click.echo("")

    if dry_run:
        click.echo("(dry run — nothing deleted)")
        return

    if clean_all:
        prompt = (
            f"Remove ALL {len(targets)} item(s) directly under {work_dir}, including anything "
            "matador didn't create?"
        )
    else:
        prompt = f"Remove {len(targets)} item(s) from {work_dir}?"

    if not yes and not click.confirm(prompt, default=False):
        click.echo("Aborted — nothing removed.")
        return

    for t in targets:
        if t.is_dir():
            shutil.rmtree(t)
        else:
            t.unlink()

    click.echo(f"Removed {len(targets)} item(s). {work_dir} is ready for a fresh start.")


@main.command()
def version() -> None:
    """Print the Matador version."""
    click.echo(f"matador {__version__}")


def _print_backends_registry() -> None:
    from matador.backends.registry import describe, list_backends as _list
    click.echo("Registered accelerator backends  (--backend flag):")
    click.echo("")
    for name in _list():
        click.echo(f"  {click.style(name, bold=True)}")
        click.echo(f"    {describe(name)}")
        click.echo(f"    Config template:  examples/{name}.yaml")
        click.echo("")
    click.echo("Synthesis backends (Vivado):")
    click.echo("  vivado      Xilinx Vivado (synthesis + implementation)")


def _print_datasets_registry() -> None:
    from matador.preprocessing import registry

    try:
        names = registry.list_datasets()
    except FileNotFoundError as exc:
        raise click.ClickException(str(exc)) from exc

    click.echo("Registered datasets  (--dataset flag on ingest/booleanize):")
    click.echo(f"  catalog: {registry.DEFAULT_CATALOG}")
    click.echo("")
    for name in names:
        ds = registry.get_dataset(name)
        click.echo(f"  {click.style(name, bold=True)}")
        click.echo(f"    {ds.name or name}")
        if ds.booleanization is not None:
            if ds.booleanization.default_encoder is not None:
                click.echo(f"    default recipe: {_format_encoder_spec(ds.booleanization.default_encoder)}")
            if ds.booleanization.features:
                click.echo(f"    + {len(ds.booleanization.features)} column-specific override(s)")
            click.echo(f"    inspect:  matador booleanize --dataset {name} --show-recipe")
        else:
            click.echo("    no verified default (see its own notes) — but you can still inspect an")
            click.echo(f"    unverified starting skeleton:  matador booleanize --dataset {name} --show-recipe")
        click.echo("")


@main.command("registry")
def registry_cmd() -> None:
    """Show everything registered: accelerator backends and datasets.

    One place to see the whole plugin catalog — equivalent to running
    `matador list-backends` followed by `matador list-datasets`, which
    remain available individually for scripting.
    """
    _print_backends_registry()
    click.echo("")
    click.echo("─" * 60)
    click.echo("")
    _print_datasets_registry()


@main.command("list-backends")
def list_backends_cmd() -> None:
    """List available RTL accelerator backends from the plugin registry."""
    _print_backends_registry()


@main.command("list-datasets")
def list_datasets_cmd() -> None:
    """List registered datasets from the shared catalog (--dataset flag)."""
    _print_datasets_registry()


@main.command("list-simulators")
def list_simulators() -> None:
    """List available RTL simulators."""
    click.echo("Available simulators:")
    click.echo("  verilator   Open-source SystemVerilog simulator")
    click.echo("  iverilog    Icarus Verilog simulator")
