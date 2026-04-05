# 🚀 Quick Start Guide

## Installation

```bash
pip install -r requirements.txt
```

## Running the App

### Web Interface (Recommended)
```bash
cd frontend
python app.py
```
Then open: **http://localhost:5000**

### Command Line
```bash
python main.py
```

## Using the Web Interface

1. Enter a disease name (e.g., "lung cancer", "diabetes")
2. Click "Discover Drugs"
3. Watch the animated workflow
4. Try the Molecule Analyzer at the bottom

## Test Different Diseases

- **lung cancer** → EGFR + cancer drugs
- **diabetes** → INS + diabetes drugs
- **Alzheimer's disease** → APP + cognitive drugs
- **heart disease** → ACE + cardiac drugs
- **asthma** → ADRB2 + respiratory drugs

## Optional: Enable AI Features

Add to `.env` file:
```
OPENAI_API_KEY=your-key-here
```

## Files You Need

```
README.md           - Project overview
GUIDE.md            - Detailed concepts
LEARNING_PATH.md    - Learning roadmap
main.py             - CLI version
frontend/           - Web interface
  ├── app.py        - Flask server
  ├── templates/    - HTML
  └── static/       - CSS/JS
src/                - Source code
```

## Troubleshooting

**Port in use?** Change port in `frontend/app.py`:
```python
app.run(debug=True, port=5001)
```

**Module errors?** Reinstall:
```bash
pip install -r requirements.txt
```

---

For detailed learning, see **LEARNING_PATH.md**
