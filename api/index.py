"""Vercel WSGI entry point — exposes the Dash app's Flask server."""

import sys
import os

# Ensure the vercel_app directory is on the path so `app` can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app import server as app
