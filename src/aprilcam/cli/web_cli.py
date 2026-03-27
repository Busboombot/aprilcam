"""CLI subcommand: aprilcam web — Start the HTTP/WebSocket server."""

import argparse


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="aprilcam web",
        description="Start the AprilCam HTTP server with REST API, MCP SSE, and WebSocket streaming",
    )
    parser.add_argument("--port", type=int, default=17439, help="Port to listen on (default: 17439)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    args = parser.parse_args(argv)

    import uvicorn
    from aprilcam.web_server import create_app

    app = create_app()

    print(f"Starting AprilCam web server on {args.host}:{args.port}")
    print(f"  REST API:    http://{args.host}:{args.port}/api/")
    print(f"  MCP SSE:     http://{args.host}:{args.port}/mcp/sse")
    print(f"  WebSocket:   ws://{args.host}:{args.port}/ws/tags/<source_id>")
    print(f"  API docs:    http://{args.host}:{args.port}/")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info", workers=1)
    return 0
