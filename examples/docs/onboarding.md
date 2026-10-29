# Onboarding

New joiners get access to the repository, the CI system and the team chat on day one.
Ask in the team channel if any of the three is missing after lunch.

# Local setup

Install uv, then run `uv sync` in the repository root. The test suite runs with `uv run pytest`.
Integration tests need Docker running; they are skipped otherwise.

# Releases

We release from the main branch every Tuesday. A release needs one approval and a green build.
Hotfixes skip the Tuesday train but still need the approval.
