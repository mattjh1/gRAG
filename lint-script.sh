#!/bin/bash

echo "🧹 Cleaning up Python project..."

# 1. Install tools if needed
echo "📦 Installing cleanup tools..."
pip install black isort autopep8 pylint

# 2. Auto-format code
echo "🎨 Formatting with black..."
black --line-length=100 --exclude='(\.venv|venv|__pycache__|\.git)' .

echo "📚 Organizing imports..."
isort --profile=black --skip=venv --skip=.venv --skip=__pycache__ .

# 3. Fix basic style issues
echo "🔧 Auto-fixing with autopep8..."
autopep8 --in-place --aggressive --aggressive --exclude=venv,__pycache__,.venv --recursive .

# 4. Create pylint config
echo "⚙️ Creating pylint config..."
cat > .pylintrc << 'EOF'
[MESSAGES CONTROL]
disable=missing-module-docstring,
        missing-class-docstring,
        missing-function-docstring,
        line-too-long,
        trailing-whitespace,
        import-outside-toplevel,
        too-few-public-methods,
        too-many-locals,
        too-many-branches,
        too-many-statements,
        broad-exception-caught,
        unused-import,
        wrong-import-order,
        trailing-newlines,
        pointless-string-statement,
        invalid-name

[FORMAT]
max-line-length=100

[BASIC]
good-names=i,j,k,ex,Run,_,e,id

[DESIGN]
max-args=10
max-locals=20
max-statements=60
EOF

# 5. Lint only Python source files
echo "🔍 Running focused pylint check..."
git ls-files '*.py' | \
    grep -v __pycache__ | \
    grep -v .venv | \
    grep -v venv | \
    grep -E '^src/' | \
    head -10 | \
    xargs pylint --rcfile=.pylintrc

echo "✅ Basic cleanup complete!"
echo "Now manually fix the remaining import errors and undefined variables."
