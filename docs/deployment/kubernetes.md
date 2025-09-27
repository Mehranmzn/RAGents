# Kubernetes Deployment

Deploy RAGents applications on Kubernetes for scalable, production-ready implementations.

## Overview

RAGents supports Kubernetes deployment with:
- **Scalable Agent Services** - Auto-scaling based on demand
- **Vector Store Integration** - Persistent storage for embeddings
- **Load Balancing** - Distribute requests across agent instances
- **Health Monitoring** - Built-in health checks and monitoring
- **Configuration Management** - ConfigMaps and Secrets integration

## Quick Start

### Prerequisites

```bash
# Ensure you have:
# - Kubernetes cluster (v1.19+)
# - kubectl configured
# - Docker registry access
# - Helm (optional but recommended)

kubectl cluster-info
```

### Basic Deployment

```yaml
# ragents-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ragents-app
  labels:
    app: ragents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ragents
  template:
    metadata:
      labels:
        app: ragents
    spec:
      containers:
      - name: ragents
        image: ragents:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENAI_API_KEY
          valueFrom:
            secretKeyRef:
              name: ragents-secrets
              key: openai-api-key
        - name: RAGENTS_VECTOR_STORE_TYPE
          value: "weaviate"
        - name: RAGENTS_WEAVIATE_URL
          value: "http://weaviate-service:8080"
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: ragents-service
spec:
  selector:
    app: ragents
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Deploy to Kubernetes

```bash
# Create namespace
kubectl create namespace ragents

# Create secrets
kubectl create secret generic ragents-secrets \
  --from-literal=openai-api-key=your-openai-key \
  --namespace=ragents

# Deploy application
kubectl apply -f ragents-deployment.yaml -n ragents

# Check deployment
kubectl get pods -n ragents
kubectl get services -n ragents
```

## Complete Production Setup

### 1. Configuration Management

```yaml
# ragents-configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: ragents-config
  namespace: ragents
data:
  RAGENTS_LLM_PROVIDER: "openai"
  RAGENTS_CHUNK_SIZE: "1000"
  RAGENTS_TOP_K: "5"
  RAGENTS_ENABLE_CACHING: "true"
  RAGENTS_VECTOR_STORE_TYPE: "weaviate"
  RAGENTS_ENABLE_TRACING: "true"
  RAGENTS_LOG_LEVEL: "info"

---
apiVersion: v1
kind: Secret
metadata:
  name: ragents-secrets
  namespace: ragents
type: Opaque
stringData:
  openai-api-key: "your-openai-api-key"
  anthropic-api-key: "your-anthropic-api-key"
  weaviate-api-key: "your-weaviate-api-key"
```

### 2. Vector Store Setup (Weaviate)

```yaml
# weaviate-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: weaviate
  namespace: ragents
spec:
  replicas: 1
  selector:
    matchLabels:
      app: weaviate
  template:
    metadata:
      labels:
        app: weaviate
    spec:
      containers:
      - name: weaviate
        image: semitechnologies/weaviate:latest
        ports:
        - containerPort: 8080
        env:
        - name: QUERY_DEFAULTS_LIMIT
          value: "25"
        - name: AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED
          value: "true"
        - name: PERSISTENCE_DATA_PATH
          value: "/var/lib/weaviate"
        - name: DEFAULT_VECTORIZER_MODULE
          value: "none"
        - name: ENABLE_MODULES
          value: "text2vec-openai,generative-openai"
        volumeMounts:
        - name: weaviate-data
          mountPath: /var/lib/weaviate
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
      volumes:
      - name: weaviate-data
        persistentVolumeClaim:
          claimName: weaviate-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: weaviate-service
  namespace: ragents
spec:
  selector:
    app: weaviate
  ports:
  - port: 8080
    targetPort: 8080

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: weaviate-pvc
  namespace: ragents
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 20Gi
```

### 3. RAGents Application Deployment

```yaml
# ragents-production.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ragents-app
  namespace: ragents
spec:
  replicas: 5
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 2
      maxUnavailable: 1
  selector:
    matchLabels:
      app: ragents
  template:
    metadata:
      labels:
        app: ragents
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8000"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: ragents
        image: ragents:v1.0.0
        ports:
        - containerPort: 8000
          name: http
        envFrom:
        - configMapRef:
            name: ragents-config
        - secretRef:
            name: ragents-secrets
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
          timeoutSeconds: 10
          failureThreshold: 3
        readinessProbe:
          httpGet:
            path: /ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        volumeMounts:
        - name: app-data
          mountPath: /app/data
        - name: cache-volume
          mountPath: /app/cache
      volumes:
      - name: app-data
        persistentVolumeClaim:
          claimName: ragents-data-pvc
      - name: cache-volume
        emptyDir:
          sizeLimit: 1Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: ragents-data-pvc
  namespace: ragents
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
```

## Auto-Scaling Configuration

### Horizontal Pod Autoscaler

```yaml
# ragents-hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ragents-hpa
  namespace: ragents
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ragents-app
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
      - type: Percent
        value: 10
        periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
      - type: Percent
        value: 50
        periodSeconds: 60
```

### Vertical Pod Autoscaler

```yaml
# ragents-vpa.yaml
apiVersion: autoscaling.k8s.io/v1
kind: VerticalPodAutoscaler
metadata:
  name: ragents-vpa
  namespace: ragents
spec:
  targetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ragents-app
  updatePolicy:
    updateMode: "Auto"
  resourcePolicy:
    containerPolicies:
    - containerName: ragents
      maxAllowed:
        cpu: 4
        memory: 8Gi
      minAllowed:
        cpu: 100m
        memory: 256Mi
```

## Monitoring and Observability

### Service Monitor for Prometheus

```yaml
# ragents-monitoring.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: ragents-metrics
  namespace: ragents
spec:
  selector:
    matchLabels:
      app: ragents
  endpoints:
  - port: http
    path: /metrics
    interval: 30s

---
apiVersion: v1
kind: Service
metadata:
  name: ragents-metrics
  namespace: ragents
  labels:
    app: ragents
spec:
  selector:
    app: ragents
  ports:
  - name: http
    port: 8000
    targetPort: 8000
```

### Logging with Fluentd

```yaml
# ragents-logging.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: fluentd-config
  namespace: ragents
data:
  fluent.conf: |
    <source>
      @type tail
      path /var/log/containers/ragents-*.log
      pos_file /var/log/fluentd-ragents.log.pos
      tag ragents.*
      format json
    </source>

    <match ragents.**>
      @type elasticsearch
      host elasticsearch.logging.svc.cluster.local
      port 9200
      index_name ragents-logs
      type_name ragents
    </match>

---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: fluentd-ragents
  namespace: ragents
spec:
  selector:
    matchLabels:
      name: fluentd-ragents
  template:
    metadata:
      labels:
        name: fluentd-ragents
    spec:
      containers:
      - name: fluentd
        image: fluent/fluentd-kubernetes-daemonset:v1-debian-elasticsearch
        volumeMounts:
        - name: config-volume
          mountPath: /fluentd/etc/fluent.conf
          subPath: fluent.conf
        - name: varlog
          mountPath: /var/log
      volumes:
      - name: config-volume
        configMap:
          name: fluentd-config
      - name: varlog
        hostPath:
          path: /var/log
```

## Ingress Configuration

### NGINX Ingress

```yaml
# ragents-ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: ragents-ingress
  namespace: ragents
  annotations:
    kubernetes.io/ingress.class: "nginx"
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "10m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
  - hosts:
    - ragents.yourdomain.com
    secretName: ragents-tls
  rules:
  - host: ragents.yourdomain.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: ragents-service
            port:
              number: 80
```

## Helm Chart Deployment

### Helm Values

```yaml
# values.yaml
replicaCount: 3

image:
  repository: ragents
  tag: latest
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
  hosts:
    - host: ragents.yourdomain.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: ragents-tls
      hosts:
        - ragents.yourdomain.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 20
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

config:
  llmProvider: "openai"
  vectorStoreType: "weaviate"
  chunkSize: 1000
  topK: 5
  enableCaching: true

secrets:
  openaiApiKey: "your-openai-key"
  anthropicApiKey: "your-anthropic-key"

weaviate:
  enabled: true
  persistence:
    size: 20Gi

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
```

### Deploy with Helm

```bash
# Add RAGents Helm repository
helm repo add ragents https://charts.ragents.com
helm repo update

# Deploy with custom values
helm install ragents ragents/ragents \
  --namespace ragents \
  --create-namespace \
  --values values.yaml

# Upgrade deployment
helm upgrade ragents ragents/ragents \
  --namespace ragents \
  --values values.yaml
```

## Security Configuration

### Network Policies

```yaml
# ragents-network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: ragents-network-policy
  namespace: ragents
spec:
  podSelector:
    matchLabels:
      app: ragents
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - namespaceSelector:
        matchLabels:
          name: ingress-nginx
    ports:
    - protocol: TCP
      port: 8000
  egress:
  - to:
    - podSelector:
        matchLabels:
          app: weaviate
    ports:
    - protocol: TCP
      port: 8080
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS for external APIs
```

### Pod Security Policy

```yaml
# ragents-psp.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: ragents-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Best Practices

### Resource Management

1. **Set Resource Limits** - Prevent resource starvation
2. **Use Multi-Zone Deployment** - For high availability
3. **Configure Persistent Storage** - For vector databases
4. **Monitor Resource Usage** - Optimize based on metrics

### Security

1. **Use Secrets for API Keys** - Never expose in ConfigMaps
2. **Enable Network Policies** - Restrict pod-to-pod communication
3. **Run as Non-Root User** - Follow security best practices
4. **Regular Security Updates** - Keep images updated

### Monitoring

1. **Health Checks** - Implement proper liveness and readiness probes
2. **Metrics Collection** - Use Prometheus for monitoring
3. **Log Aggregation** - Centralize logs with ELK stack
4. **Alert Configuration** - Set up alerts for critical issues

### Performance

1. **Horizontal Scaling** - Scale based on demand
2. **Cache Configuration** - Use persistent caching layers
3. **Load Balancing** - Distribute traffic evenly
4. **Database Optimization** - Tune vector store performance

## Troubleshooting

### Common Issues

**Pod Startup Failures:**
```bash
# Check pod status
kubectl describe pod -l app=ragents -n ragents

# Check logs
kubectl logs -l app=ragents -n ragents --tail=100

# Check events
kubectl get events -n ragents --sort-by='.lastTimestamp'
```

**Connection Issues:**
```bash
# Test service connectivity
kubectl run test-pod --rm -i --tty --image=busybox -- /bin/sh
# Inside pod:
# wget -qO- http://ragents-service/health

# Check DNS resolution
nslookup ragents-service.ragents.svc.cluster.local
```

**Resource Issues:**
```bash
# Check resource usage
kubectl top pods -n ragents
kubectl top nodes

# Check resource limits
kubectl describe pod -l app=ragents -n ragents | grep -A 5 "Limits"
```

## Next Steps

- **[Kubeflow Integration](kubeflow.md)** - ML pipeline deployment
- **[Production Deployment](production.md)** - Production considerations
- **[Docker Setup](docker.md)** - Container configuration
- **[Monitoring](../advanced/observability.md)** - Detailed monitoring setup