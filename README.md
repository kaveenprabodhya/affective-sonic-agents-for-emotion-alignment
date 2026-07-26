# A Multi-Agent Approach to Explore How Aligned Brand-Intended and Audience-Perceived Emotional Responses using Synthetic Data. 

## An Application to Sonic Identity Evaluation as Expressed Through Radio Advertising

---

# 1. create the virtual environment
python -m venv .venv

# 2. activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\Activate.ps1     # Windows PowerShell
# .venv\Scripts\activate.bat     # Windows cmd

# 3. install
pip install --upgrade pip
pip install -r requirements.txt

# 4. verify
python -c "import librosa, soundfile, sklearn, pandas, matplotlib; print('ok', librosa.__version__)"

# 5. once it works, freeze an exact lock for reproducibility
pip freeze > requirements.lock.txt