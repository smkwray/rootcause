# Security Policy

## Reporting a Vulnerability

If you discover a security issue in this repository, do not open a public issue with exploit details.

Send a private report to the maintainer with:

- a description of the issue
- affected files, commands, or workflows
- reproduction steps
- any suggested remediation

I will acknowledge receipt, assess impact, and work on a fix before public disclosure.

## Scope

This project is primarily a research/data pipeline repository. The highest-priority reports are issues that could lead to:

- credential or API key exposure
- unsafe handling of downloaded data or generated artifacts
- command injection or unsafe shell execution
- accidental publication of sensitive local paths or secrets
