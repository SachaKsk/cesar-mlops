"""CESAR CLI: one entry point for all commands.

All command-line interfaces (typer) live in the cli/ folder.
Each command file only handles arguments and printing; the real logic
lives in runtime/, model_acceptance_tests/, etc.
"""

import typer

from cli.batch import batch_app
from cli.predict_one import predict_one_app
from cli.acceptance_tests import acceptance_tests_app
from cli.experiment_analysis import app as experiment_analysis_app

app = typer.Typer(
    name="cesar",
    help="CESAR: batch and single-record property value prediction.",
)
app.add_typer(batch_app, name="batch")
app.add_typer(predict_one_app, name="predict-one")
app.add_typer(acceptance_tests_app, name="acceptance-tests")
app.add_typer(experiment_analysis_app, name="experiment-analysis")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
