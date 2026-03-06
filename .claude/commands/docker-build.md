---
description: Build Docker image using docker compose (CPU-only torch, non-root trader). Usage: /project:docker-build
---

Build the SwingRL Docker image using docker compose.

```bash
docker compose build $ARGUMENTS
```

Common patterns:
- Fresh build (no cache): `/project:docker-build` then `docker compose build --no-cache`
- Run tests in container: `docker compose run --rm swingrl pytest tests/ -v`
- Interactive shell: `docker compose run --rm swingrl bash`
- Check image size: `docker images swingrl`

Note: Uses CPU-only PyTorch (pytorch-cpu index). MPS is for M1 Mac native runs only.
Production deployment uses the same image with TRADING_MODE=paper in .env.
