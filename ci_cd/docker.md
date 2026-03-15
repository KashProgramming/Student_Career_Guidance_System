# Docker

Build images locally:

```bash
docker build -f docker/Dockerfile.backend -t student-career-backend:latest .
docker build -f docker/Dockerfile.frontend -t student-career-frontend:latest .
```

Run locally:

```bash
docker-compose up --build
```
