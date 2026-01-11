FROM python:3.11-slim

WORKDIR /app

# Install uv for package management
RUN pip install --no-cache-dir uv

# Copy local source code
COPY . /app

# Install from local source (not PyPI)
RUN uv pip install --system --no-cache-dir .

# Apply FastMCP fix for Annotated[str, ...] JSON parsing bug
# FastMCP incorrectly parses str params as JSON when FASTMCP_TOOL_ATTEMPT_PARSE_JSON_ARGS=true
# because it checks `annotation in (str,)` but pydantic uses Annotated[str, Field(...)]
RUN FASTMCP_TOOL_FILE=$(python -c "import fastmcp.tools.tool as t; print(t.__file__)") && \
    sed -i 's/from typing import TYPE_CHECKING, Any/from typing import TYPE_CHECKING, Any, Annotated, get_origin, get_args/' "$FASTMCP_TOOL_FILE" && \
    python << 'EOF'
import sys
fastmcp_path = sys.argv[1] if len(sys.argv) > 1 else None
if not fastmcp_path:
    import fastmcp.tools.tool as t
    fastmcp_path = t.__file__
with open(fastmcp_path, 'r') as f:
    content = f.read()
old = '''                # skip if the type is a simple type (int, float, bool)
                if signature.parameters[param_name].annotation in (
                    int,
                    float,
                    bool,
                ):
                    continue'''
new = '''                # skip if the type is a simple type (int, float, bool, str)
                # Also handle Annotated[str, ...] by unwrapping
                annotation = signature.parameters[param_name].annotation
                base_type = annotation
                if get_origin(annotation) is Annotated:
                    base_type = get_args(annotation)[0]
                if base_type in (int, float, bool, str):
                    continue'''
if old in content:
    content = content.replace(old, new)
    with open(fastmcp_path, 'w') as f:
        f.write(content)
    print("FastMCP str fix applied")
else:
    print("FastMCP already patched or structure changed")
EOF

# Expose the default port for SSE transport (not used for STDIO but kept for compatibility)
EXPOSE 8000

# Set environment variables with defaults that can be overridden at runtime
ENV QDRANT_URL=""
ENV QDRANT_API_KEY=""
# NOTE: COLLECTION_NAME intentionally left unset to expose collection_name parameter in tool schema
# If set, make_partial_function strips collection_name from schema, preventing multi-collection use
# ENV COLLECTION_NAME=""
ENV EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"

# Run the already-installed package (STDIO mode by default)
# MCPO will communicate via stdin/stdout
CMD ["mcp-server-qdrant"]
