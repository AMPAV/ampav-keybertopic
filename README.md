# ampav-keybert

Synchronous KeyBERT keyword and keyphrase extraction tooling for the AMPAV
environment.

The public extraction API is under development.

## Development

Use the shared AMPAV virtual environment and install the package in editable
mode:

```bash
python -m pip install -e .
```

Run the unit tests without loading a real model:

```bash
python -m unittest discover -s tests
```
