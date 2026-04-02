"""Health check endpoint."""

def GET(req):
    """GET /api/health - Health check."""
    import json
    return {
        "statusCode": 200,
        "body": json.dumps({"status": "ok", "version": "0.1.0", "platform": "vercel"}),
        "headers": {"Content-Type": "application/json"}
    }
