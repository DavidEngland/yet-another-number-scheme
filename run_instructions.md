# Running YANS Scripts on macOS

## Installing Dependencies

Before running the scripts, you need to install the required dependencies:

```bash
# Install sympy and other dependencies
python3 -m pip install sympy mpmath numpy

# Or install from requirements.txt
python3 -m pip install -r requirements.txt
```

## Common Python Command Names

On macOS, Python is often accessible through different command names:

```bash
# Try one of these commands
python3 /Users/davidengland/Documents/GitHub/yet-another-number-scheme/test_bernoulli.py

# Or if you have a specific Python version
python3.9 /Users/davidengland/Documents/GitHub/yet-another-number-scheme/test_bernoulli.py
python3.10 /Users/davidengland/Documents/GitHub/yet-another-number-scheme/test_bernoulli.py

# Using the full path to Python (find with "which python3")
/usr/bin/python3 /Users/davidengland/Documents/GitHub/yet-another-number-scheme/test_bernoulli.py
/usr/local/bin/python3 /Users/davidengland/Documents/GitHub/yet-another-number-scheme/test_bernoulli.py
```

## Running Scripts with the Shebang Line

You can also add a shebang line at the top of your script and make it executable:

1. Add this as the first line of your script:
   ```python
   #!/usr/bin/env python3
   ```

2. Make the script executable:
   ```bash
   chmod +x /Users/davidengland/Documents/GitHub/yet-another-number-scheme/test_bernoulli.py
   ```

3. Run it directly:
   ```bash
   ./test_bernoulli.py
   ```

## Using Homebrew Python

If you installed Python via Homebrew, try:

```bash
brew list python  # List Python installations
/opt/homebrew/bin/python3 /Users/davidengland/Documents/GitHub/yet-another-number-scheme/test_bernoulli.py
```

## Running through an IDE

If you have VS Code, PyCharm, or another IDE installed, you can open and run the script directly from there.
