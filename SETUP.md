# GeoBot Setup

When you **open this repo on a new device** (or after a fresh clone):

1. **Run from project root** – `cd` into the GeoBot folder before running commands.
2. **Install dependencies** – run setup once:

### Windows
```bat
setup.bat
```

### Mac / Linux
```sh
chmod +x setup.sh && ./setup.sh
```

### Or manually
```bash
pip install -r requirements.txt
```

Then start the app:
```bash
streamlit run app.py
```
