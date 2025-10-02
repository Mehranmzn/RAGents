# Kubernetes Deployment for RAGents

Production-grade Kubernetes deployment with auto-scaling, load balancing, and high availability.

## Architecture

```
┌─────────────────┐
│  Load Balancer  │
└────────┬────────┘
         │
    ┌────▼────────────────────┐
    │  Ingress Controller     │
    └────┬────────────────────┘
         │
    ┌────▼─────────────────┐
    │  RAGents Service     │
    │  (Load Balanced)     │
    └──┬──┬──┬──┬──┬───────┘
       │  │  │  │  │
    ┌──▼──▼──▼──▼──▼──┐
    │  RAGents Pods   │ (3-20 replicas)
    │  - Auto-scaled  │
    │  - Health Checks│
    │  - PDB: min 2   │
    └──┬──────────┬───┘
       │          │
    ┌──▼──┐    ┌─▼────────┐
    │Redis│    │PostgreSQL│
    │Cache│    │Checkpts  │
    └─────┘    └──────────┘
```

## Prerequisites

1. **Kubernetes Cluster** (v1.24+)
   - minikube (local)
   - EKS (AWS)
   - GKE (Google Cloud)
   - AKS (Azure)

2. **kubectl** configured

3. **Container Registry**
   - Docker Hub
   - ECR
   - GCR
   - ACR

4. **Metrics Server** (for HPA)
   ```bash
   kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
   ```

## Quick Start

### 1. Build and Push Docker Image

```bash
# Using BentoML
cd /path/to/ragents
bentoml build
bentoml containerize ragents:latest -t your-registry/ragents:latest
docker push your-registry/ragents:latest
```

### 2. Create Namespace

```bash
kubectl create namespace production
```

### 3. Configure Secrets

```bash
# Copy example secrets
cp k8s/secrets.yaml.example k8s/secrets.yaml

# Edit with your actual credentials
vim k8s/secrets.yaml

# Apply secrets
kubectl apply -f k8s/secrets.yaml
```

### 4. Deploy Redis

```bash
kubectl apply -f k8s/redis.yaml
```

### 5. Deploy RAGents

```bash
# Update image in deployment.yaml to your registry
kubectl apply -f k8s/deployment.yaml
```

### 6. Verify Deployment

```bash
# Check pods
kubectl get pods -n production

# Check HPA
kubectl get hpa -n production

# Check service
kubectl get svc -n production

# Check logs
kubectl logs -f deployment/ragents-service -n production
```

## Configuration

### Horizontal Pod Autoscaler (HPA)

The HPA automatically scales RAGents pods based on:

- **CPU Utilization**: Scale when >70%
- **Memory Utilization**: Scale when >80%
- **HTTP Requests**: Scale when >100 RPS per pod

**Scaling Behavior:**
- **Scale Up**: 50% increase or +2 pods (whichever is larger), every 60s
- **Scale Down**: 10% decrease or -1 pod (whichever is smaller), every 60s
- **Min Replicas**: 3
- **Max Replicas**: 20

### Resource Requests/Limits

**Per Pod:**
- Requests: 2 CPU, 4Gi memory
- Limits: 4 CPU, 8Gi memory

**Adjust based on your workload:**
```yaml
resources:
  requests:
    memory: "4Gi"
    cpu: "2000m"
  limits:
    memory: "8Gi"
    cpu: "4000m"
```

### Pod Disruption Budget (PDB)

Maintains minimum 2 pods during:
- Voluntary disruptions (node drains, updates)
- Cluster maintenance
- Rolling updates

## Monitoring

### Health Checks

**Liveness Probe:**
- Checks if pod is alive
- Restarts pod on failure
- Endpoint: `/health`

**Readiness Probe:**
- Checks if pod is ready to serve traffic
- Removes from service on failure
- Endpoint: `/health`

**Startup Probe:**
- Allows slow startup (up to 5 minutes)
- Prevents premature restarts

### Metrics

**Prometheus Integration:**
```yaml
annotations:
  prometheus.io/scrape: "true"
  prometheus.io/port: "3000"
  prometheus.io/path: "/metrics"
```

**Key Metrics:**
- Request rate (RPS)
- Response latency (P50, P95, P99)
- Error rate
- Cache hit rate
- Queue depth
- Active threads

## Scaling Strategies

### Vertical Scaling

Increase resources per pod:

```bash
kubectl patch deployment ragents-service -n production -p '
{
  "spec": {
    "template": {
      "spec": {
        "containers": [{
          "name": "ragents",
          "resources": {
            "requests": {"cpu": "4000m", "memory": "8Gi"},
            "limits": {"cpu": "8000m", "memory": "16Gi"}
          }
        }]
      }
    }
  }
}'
```

### Horizontal Scaling

Adjust HPA limits:

```bash
kubectl patch hpa ragents-hpa -n production -p '
{
  "spec": {
    "minReplicas": 5,
    "maxReplicas": 50
  }
}'
```

### Manual Scaling

Override HPA temporarily:

```bash
# Scale to specific replica count
kubectl scale deployment ragents-service -n production --replicas=10

# Delete HPA to prevent auto-scaling
kubectl delete hpa ragents-hpa -n production
```

## High Availability

### Multi-Zone Deployment

Distribute pods across availability zones:

```yaml
spec:
  template:
    spec:
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: app
                  operator: In
                  values:
                  - ragents
              topologyKey: topology.kubernetes.io/zone
```

### Rolling Updates

Zero-downtime deployments:

```bash
kubectl set image deployment/ragents-service ragents=your-registry/ragents:v2 -n production
kubectl rollout status deployment/ragents-service -n production
```

## Troubleshooting

### Pods Not Starting

```bash
# Check pod status
kubectl describe pod <pod-name> -n production

# Check logs
kubectl logs <pod-name> -n production

# Check events
kubectl get events -n production --sort-by='.lastTimestamp'
```

### HPA Not Scaling

```bash
# Check HPA status
kubectl describe hpa ragents-hpa -n production

# Verify metrics server
kubectl get deployment metrics-server -n kube-system

# Check current metrics
kubectl top pods -n production
```

### High Latency

```bash
# Check pod resources
kubectl top pods -n production

# Check Redis connection
kubectl exec -it <pod-name> -n production -- redis-cli -h redis-service ping

# Check cache stats
curl http://<service-ip>/cache/stats
```

## Cost Optimization

### Cluster Autoscaler

Enable node auto-scaling:

```yaml
# For EKS
eksctl create cluster --managed --asg-access
```

### Spot Instances

Use spot/preemptible instances for cost savings:

```yaml
spec:
  template:
    spec:
      nodeSelector:
        karpenter.sh/capacity-type: spot
```

### Resource Optimization

Monitor and adjust based on actual usage:

```bash
# Get resource usage over time
kubectl top pods -n production --containers

# Analyze with VPA (Vertical Pod Autoscaler)
kubectl get vpa ragents-vpa -n production
```

## Security

### Network Policies

Restrict traffic between pods:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ragents-netpol
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: ragents
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: ingress-nginx
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: redis
```

### RBAC

Limit permissions:

```bash
kubectl create serviceaccount ragents-sa -n production
kubectl create rolebinding ragents-rb --serviceaccount=production:ragents-sa --clusterrole=view -n production
```

## Backup and Disaster Recovery

### Redis Backup

```bash
# Manual backup
kubectl exec -it redis-0 -n production -- redis-cli BGSAVE

# Automated with CronJob
kubectl apply -f k8s/redis-backup-cronjob.yaml
```

### PostgreSQL Backup

```bash
# Using pg_dump
kubectl exec -it postgres-0 -n production -- pg_dump -U postgres ragents > backup.sql
```

## Load Testing

### Using Apache Bench

```bash
ab -n 1000 -c 10 -p request.json -T application/json http://<service-ip>/query
```

### Using k6

```javascript
import http from 'k6/http';

export default function () {
  const payload = JSON.stringify({
    query: 'What is RAGents?',
    agent_type: 'simple'
  });
  http.post('http://<service-ip>/query', payload, {
    headers: { 'Content-Type': 'application/json' }
  });
}
```

Run test:
```bash
k6 run --vus 100 --duration 5m loadtest.js
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/your-org/ragents/issues
- Documentation: https://docs.ragents.ai
