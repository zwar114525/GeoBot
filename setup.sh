#!/bin/sh
# GeoBot - One-time setup when opening repo on a new device
echo "Installing GeoBot dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
  echo ""
  echo "Setup complete. Run: streamlit run app.py"
else
  echo "Setup failed. Check your Python/pip installation."
  exit 1
fi
