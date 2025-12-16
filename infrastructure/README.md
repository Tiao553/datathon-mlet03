# Infrastructure

This directory contains all infrastructure-as-code and deployment configurations.

## Structure

### `local/`
Local development environment using Docker Compose.
- `docker-compose.yml`: Full stack (API, MLflow, Elasticsearch, Kibana, Langfuse, Airflow)
- `Dockerfile`: API service container definition

### `cloud/`
Cloud deployment configurations (Terraform, Kubernetes, etc.)
- **Status:** To be implemented in future phases
- **Planned:** AWS/GCP infrastructure definitions

## Quick Start (Local)

```bash
cd infrastructure/local
docker-compose up -d
```

**Services:**
- API: http://localhost:8000
- MLflow: http://localhost:5000
- Kibana: http://localhost:5601
- Langfuse: http://localhost:3000
- Airflow: http://localhost:8080
