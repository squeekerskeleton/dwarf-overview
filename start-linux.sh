#!/bin/bash
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  echo "First run: setting up, this takes a minute..."
  python3 -m venv .venv && .venv/bin/pip install --upgrade pip >/dev/null
fi
while true; do
  .venv/bin/pip install -q -r requirements.txt
  .venv/bin/python -m dwarf_overview
  [ $? -eq 3 ] || break  # 3 = an update was installed, so start again
  export DWARF_OVERVIEW_NO_BROWSER=1
done
