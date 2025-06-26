"""
Timing middleware for FastAPI request logging
"""
import time
from typing import Callable
from datetime import datetime
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.types import ASGIApp

class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to log request timing and performance metrics"""
    
    def __init__(self, app: ASGIApp):
        super().__init__(app)
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Start timing
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        # Get timestamp for logging
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Get route name if available
        route = request.url.path
        if hasattr(request.state, 'endpoint') and request.state.endpoint:
            route = f"{route} -> {request.state.endpoint.__name__}"
        
        # Log request details with timing
        print(
            f"[{timestamp}] ⏱️ {request.method} {route} - "
            f"Status: {response.status_code} - "
            f"Duration: {duration:.3f}s"
        )
        
        return response 