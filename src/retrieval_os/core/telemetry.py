"""OpenTelemetry tracer provider setup.

Traces are exported to Jaeger (or any OTLP-compatible backend) via gRPC.
Metrics are served separately via prometheus-client at /metrics.
"""

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

_provider: TracerProvider | None = None


def setup_telemetry(
    *,
    app_name: str,
    environment: str,
    otel_endpoint: str,
    enabled: bool = True,
) -> None:
    global _provider

    resource = Resource.create(
        {
            "service.name": app_name,
            "service.environment": environment,
            "deployment.environment": environment,
        }
    )

    provider = TracerProvider(resource=resource)

    if enabled and otel_endpoint:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=otel_endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
    else:
        provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    trace.set_tracer_provider(provider)
    _provider = provider


def get_tracer(name: str) -> trace.Tracer:
    return trace.get_tracer(name)


def shutdown_telemetry() -> None:
    global _provider
    if _provider:
        _provider.shutdown()
        _provider = None
