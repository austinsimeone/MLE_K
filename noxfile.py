import nox
from nox.sessions import Session


@nox.session(python="3.9", reuse_venv=True)
def black(session: Session) -> None:
    """Run black code formatter."""
    session.install("black")
    session.run("black", "--line-length", "158", ".")


@nox.session(python="3.9", reuse_venv=True)
def lint(session: Session) -> None:
    """Lint using flake8."""
    session.install("flake8")
    session.run("flake8", "src")
    session.run("flake8", "scripts")


@nox.session(python="3.9", reuse_venv=True)
def typing(session: Session) -> None:
    """Run mypy static type checker."""
    session.install("mypy")
    session.install("types-PyYAML")
    session.run("mypy", "src")
    session.run("mypy", "scripts")


@nox.session(python="3.9", reuse_venv=True)
def tests(session: Session) -> None:
    """Run the test suite."""
    session.install(".")
    session.install("-r", "requirements.txt")
    session.run("pytest", "tests")
