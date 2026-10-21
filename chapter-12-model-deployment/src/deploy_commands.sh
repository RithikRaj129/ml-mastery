# deploy_commands.sh - Build, test, and deploy

# ── Build Docker image ────────────────────────────────────────────
docker build -t ml-cancer-classifier:v1.0 .
docker build -t ml-cancer-classifier:v1.0 --no-cache .  # Force rebuild

# ── Run locally ───────────────────────────────────────────────────
docker run -p 8000:8000 \
  -e API_KEY=my-secret-key \
  -e MODEL_PATH=/app/model_pipeline.joblib \
  ml-cancer-classifier:v1.0

# ── Test the container ────────────────────────────────────────────
curl http://localhost:8000/health

# ── Push to Docker Hub ────────────────────────────────────────────
docker tag ml-cancer-classifier:v1.0 yourusername/ml-api:v1.0
docker push yourusername/ml-api:v1.0

# ── Deploy to AWS ECS (Elastic Container Service) ─────────────────
# 1. Create ECR repository
aws ecr create-repository --repository-name ml-cancer-classifier

# 2. Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin \
  123456789.dkr.ecr.us-east-1.amazonaws.com

# 3. Push to ECR
docker tag ml-cancer-classifier:v1.0 \
  123456789.dkr.ecr.us-east-1.amazonaws.com/ml-cancer-classifier:v1.0
docker push 123456789.dkr.ecr.us-east-1.amazonaws.com/ml-cancer-classifier:v1.0

# 4. Deploy to ECS Fargate (serverless containers)
aws ecs update-service \
  --cluster ml-cluster \
  --service ml-api-service \
  --force-new-deployment

# ── Deploy to Google Cloud Run (simplest cloud option) ────────────
gcloud run deploy ml-api \
  --image gcr.io/PROJECT_ID/ml-cancer-classifier:v1.0 \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 100 \
  --set-env-vars API_KEY=my-secret-key

# ── Deploy to Azure Container Instances ───────────────────────────
az container create \
  --resource-group ml-rg \
  --name ml-api \
  --image yourusername/ml-api:v1.0 \
  --cpu 2 --memory 4 \
  --ports 8000 \
  --environment-variables API_KEY=my-secret-key
