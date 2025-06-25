# gunicorn.conf.py
import os

# Basic configuration
bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"
workers = 1
worker_class = "uvicorn.workers.UvicornWorker"

# INCREASE TIMEOUTS
timeout = 120  # 2 minutes 
keepalive = 5
max_requests = 1000
max_requests_jitter = 100

# Logging
loglevel = "info"
accesslog = "-"
errorlog = "-"

# Performance
preload_app = True
worker_tmp_dir = "/dev/shm"