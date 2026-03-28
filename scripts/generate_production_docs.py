import os

# Synthetic production-level knowledge base
PRODUCTION_DOCS = [
    # API Documentation (20 docs)
    """REST API v2.1 Authentication: All API endpoints require an X-API-Key header with a valid API key. Generate API keys in the dashboard under Settings > API Keys. API keys expire after 90 days of inactivity and must be rotated annually for security compliance.""",
    
    """Rate Limiting: Standard tier allows 1000 requests per minute. Premium tier allows 10000 requests per minute. Enterprise tier has custom limits. Rate limits reset at UTC midnight daily. Exceeding limits returns 429 Too Many Requests status.""",
    
    """Error Codes: 400 Bad Request (invalid params), 401 Unauthorized (missing/invalid auth), 403 Forbidden (insufficient permissions), 404 Not Found (resource missing), 429 Too Many Requests (rate limited), 500 Internal Server Error (backend failure), 503 Service Unavailable (maintenance).""",
    
    """Webhook Integration: Webhooks deliver real-time events to your application. Configure webhooks in Settings > Webhooks. Retry failed deliveries up to 5 times exponentially. Verify webhook signatures using HMAC-SHA256 with your webhook secret.""",
    
    """Database Connections: Connect to PostgreSQL 12+, MySQL 8+, or MongoDB 4.4+. Connection pooling is recommended for production. SSL/TLS required for external connections. Connection strings use format: driver://user:password@host:port/database.""",
    
    # System Architecture (20 docs)
    """Microservices Architecture: The platform consists of 7 independent microservices: Auth Service, API Gateway, Data Processing, Storage Service, Cache Layer, Search Service, and Notification Service. Services communicate via gRPC with service discovery via Consul.""",
    
    """High Availability: All services run on Kubernetes with minimum 3 replicas per pod. Database uses primary-replica replication with automatic failover. Redis cluster provides distributed caching. Load balancers route traffic across availability zones.""",
    
    """Disaster Recovery: RPO (Recovery Point Objective) is 5 minutes. RTO (Recovery Time Objective) is 15 minutes. Daily backups stored in multiple regions. Automated failover tested monthly. Disaster recovery drills conducted quarterly.""",
    
    """Security Compliance: SOC 2 Type II certified. GDPR, HIPAA, and PCI-DSS compliant. Data encrypted at-rest using AES-256 and in-transit using TLS 1.3. Penetration testing performed quarterly by third-party auditors.""",
    
    """Deployment Pipeline: Changes go through develop → staging → production. Each stage has automated tests, security scanning, and performance benchmarking. Canary deployments reduce blast radius. Rollback available within 60 seconds.""",
    
    # Feature Documentation (20 docs)
    """Real-time Data Streaming: Stream data using WebSocket connections or Server-Sent Events. Guaranteed delivery with at-least-once semantics. Batch processing available for non-real-time use cases. Maximum message size 10MB.""",
    
    """Data Export: Export to CSV, JSON, Parquet, or Arrow formats. Scheduled exports available daily, weekly, or monthly. Export data to AWS S3, Google Cloud Storage, or Azure Blob Storage. Encrypted exports with customer-managed keys supported.""",
    
    """Search Capabilities: Full-text search across all document types. Advanced query syntax with AND, OR, NOT operators. Faceted search with filters. Elasticsearch backend ensures sub-100ms query latency. Custom analyzers for domain-specific terms.""",
    
    """Collaboration Features: Real-time collaboration on documents with conflict resolution. Comments and mentions for team communication. Role-based access control with granular permissions. Activity audit logs track all changes.""",
    
    """Custom Integrations: Zapier integration connects to 5000+ apps. Webhooks enable custom workflows. IFTTT support for simple automations. Custom API endpoints for enterprise customers.""",
    
    # Troubleshooting & FAQs (20 docs)
    """Connection Timeout Issues: Check internet connectivity first. Verify API endpoint is reachable via curl or ping. Ensure firewall allows outbound HTTPS on port 443. Check API key validity. Contact support with request ID from logs if problem persists.""",
    
    """Performance Degradation: Check dashboard status page for service incidents. Verify personal API rate limits in Settings. Reduce query complexity if fetching large datasets. Enable caching headers. Contact support for dedicated investigation.""",
    
    """Data Sync Failures: Enable automatic retry with exponential backoff. Check source data format matches schema. Verify authentication credentials haven't expired. Review data validation rules. Check storage quota in Settings.""",
    
    """Missing Historical Data: Data retention depends on plan tier. Free tier: 7 days, Standard: 30 days, Premium: 90 days, Enterprise: 1 year. Export data before retention expiry. Archive to cold storage. Contact support to extend retention.""",
    
    """Account & Billing Questions: Upgrade plans in Settings > Billing. Billing cycle: monthly or annual. No setup fees. Cancel anytime with no penalty. Usage-based overage charges apply only above plan tier limits. Tax ID required for business accounts.""",
    
    # Best Practices (20 docs)
    """Query Optimization: Use indexes on frequently filtered columns. Avoid SELECT * in large tables. Use pagination with limit/offset. Batch operations when possible. Cache results for repeated queries. Monitor slow query logs.""",
    
    """Security Best Practices: Rotate API keys quarterly. Never commit credentials to version control. Use environment variables for secrets. Enable MFA on user accounts. Restrict IP addresses for API access. Use VPC endpoints for private connectivity.""",
    
    """Scalability Guidelines: Design stateless services for horizontal scaling. Use message queues for async processing. Implement circuit breakers. Cache aggressively. Shard data when table exceeds 100GB. Monitor resource utilization trends.""",
    
    """Testing Strategy: Unit tests for all business logic (>80% coverage). Integration tests for API endpoints. Performance tests under load (1000+ concurrent users). Security tests for SQL injection and XSS. Staging environment must mirror production.""",
    
    """Monitoring & Alerts: Set alerts for CPU >80%, memory >85%, disk >90%. Monitor API latency percentiles (p50, p95, p99). Track error rates and alert when >1% of requests fail. Log all authentication failures. Review metrics daily.""",
    
    # Release Notes & Changelog (10 docs)
    """v2.1.0 Release (Jan 2025): Performance improvements reduce query latency by 35%. New WebSocket API for real-time subscriptions. Breaking change: deprecated /v1/legacy endpoint. New: support for GraphQL queries. Fixed: memory leak in cache layer.""",
    
    """v2.0.0 Release (Dec 2024): Complete API redesign with cleaner REST conventions. Kubernetes-native deployment. Multi-region failover. Webhook retry logic enhanced. Deprecation period: 6 months for v1 endpoints.""",
    
    """Roadmap Q1 2025: GraphQL subscriptions. Machine learning-powered recommendations. Enhanced analytics dashboard. API versioning strategy. Mobile app public beta.""",
    
    """Known Issues v2.1.0: Occasional sync delays (5-30s) when processing >10k events/sec. Export to older Excel versions may truncate long fields. WebSocket connections timeout after 24 hours idle.""",
    
    """Deprecation Notice: Scheduled removal of OAuth 1.0 support (June 2025). SOAP endpoints deprecated (Sept 2025). XML response format deprecated (Dec 2025). Customers notified 6 months before removal.""",
    
    # Enterprise Features (10 docs)
    """Single Sign-On (SSO): SAML 2.0 and OAuth 2.0 supported. Just-in-time provisioning for new users. Automatic group sync from identity provider. Audit logs for access control changes.""",
    
    """Data Residency: Data stored exclusively in selected region (US, EU, APAC, or single country). No cross-border data transfers. Compliant with GDPR, LGPD, and local data protection laws.""",
    
    """Advanced Permissioning: Row-level security (RLS) for data isolation. Column-level encryption for sensitive fields. Attribute-based access control (ABAC). Time-based access restrictions. IP whitelist/blacklist enforcement.""",
    
    """Custom SLA: Enterprise plans include guaranteed 99.99% uptime SLA. Priority support with <1 hour response time. Dedicated infrastructure option. Custom rate limits. On-premise deployment available.""",
    
    """Compliance Reporting: Automated compliance reports for SOC 2, HIPAA, PCI-DSS. Real-time audit logs exported to SIEM. Data classification and sensitivity labeling. Compliance dashboard with policy enforcement.""",
]

def generate_production_docs():
    docs_path = "data/rag_docs"
    os.makedirs(docs_path, exist_ok=True)
    
    for idx, doc in enumerate(PRODUCTION_DOCS):
        filename = f"{docs_path}/doc_{idx:03d}.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(doc)
    
    print(f"✅ Generated {len(PRODUCTION_DOCS)} production-level documents")

if __name__ == "__main__":
    generate_production_docs()
