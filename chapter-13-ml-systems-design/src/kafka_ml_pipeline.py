# kafka_ml_pipeline.py - Kafka-based streaming ML inference
# Install: pip install confluent-kafka kafka-python
# Requires running Kafka broker

from confluent_kafka import Producer, Consumer, KafkaError
from confluent_kafka.admin import AdminClient, NewTopic
import json
import numpy as np
import joblib
import time
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

KAFKA_CONFIG = {
    'bootstrap.servers': 'localhost:9092',
    'client.id': 'ml-pipeline',
}

# ── Topic management ──────────────────────────────────────────────
def create_topics(topic_names: list, num_partitions: int = 3,
                   replication_factor: int = 1) -> None:
    admin = AdminClient(KAFKA_CONFIG)
    topics = [
        NewTopic(name, num_partitions=num_partitions,
                 replication_factor=replication_factor)
        for name in topic_names
    ]
    futures = admin.create_topics(topics)
    for topic, future in futures.items():
        try:
            future.result()  # Block until topic is created
            print(f'Topic created: {topic}')
        except Exception as e:
            if 'already exists' not in str(e):
                logger.error(f'Failed to create topic {topic}: {e}')


# ── ML Event Producer ─────────────────────────────────────────────
class MLEventProducer:
    '''Publishes ML events (features, predictions) to Kafka topics.'''

    def __init__(self, config: dict = KAFKA_CONFIG):
        self.producer = Producer(config)

    def publish(self, topic: str, key: str, value: Dict[str, Any],
                timestamp: Optional[datetime] = None) -> None:
        message = {
            'key':       key,
            'value':     value,
            'timestamp': (timestamp or datetime.utcnow()).isoformat(),
        }
        self.producer.produce(
            topic=topic,
            key=key.encode('utf-8'),
            value=json.dumps(message).encode('utf-8'),
            on_delivery=self._delivery_callback
        )
        self.producer.poll(0)  # Trigger delivery callbacks

    def _delivery_callback(self, err, msg):
        if err:
            logger.error(f'Message delivery failed: {err}')

    def flush(self, timeout: float = 10.0) -> None:
        self.producer.flush(timeout=timeout)

    def __enter__(self): return self
    def __exit__(self, *args): self.flush()


# ── ML Inference Consumer ─────────────────────────────────────────
class MLInferenceConsumer:
    '''
    Consumes feature events from Kafka, runs ML inference,
    publishes predictions back to Kafka.
    '''

    def __init__(self, model_path: str, input_topic: str,
                  output_topic: str, group_id: str,
                  feature_names: list):
        self.model        = joblib.load(model_path)
        self.input_topic  = input_topic
        self.output_topic = output_topic
        self.feature_names= feature_names
        self.running      = False

        consumer_config = {**KAFKA_CONFIG,
            'group.id':           group_id,
            'auto.offset.reset':  'earliest',
            'enable.auto.commit': False,  # Manual commit for exactly-once
        }
        self.consumer = Consumer(consumer_config)
        self.producer = MLEventProducer()
        self.stats    = {'processed': 0, 'errors': 0, 'latencies': []}

    def process_event(self, event: Dict) -> Dict:
        '''Transform event into prediction.'''
        features = event['value'].get('features', {})
        X = np.array([[features.get(f, 0) for f in self.feature_names]])

        pred  = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]

        return {
            'entity_id':   event['key'],
            'prediction':  int(pred),
            'probability': float(max(proba)),
            'model':       type(self.model).__name__,
            'predicted_at':datetime.utcnow().isoformat(),
        }

    def run(self, max_messages: Optional[int] = None) -> None:
        self.consumer.subscribe([self.input_topic])
        self.running = True
        n_processed  = 0

        print(f'Consumer started: {self.input_topic} → {self.output_topic}')

        try:
            while self.running:
                msg = self.consumer.poll(timeout=1.0)

                if msg is None: continue
                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF: continue
                    logger.error(f'Consumer error: {msg.error()}')
                    self.stats['errors'] += 1
                    continue

                t0    = time.perf_counter()
                event = json.loads(msg.value().decode('utf-8'))

                try:
                    prediction = self.process_event(event)
                    self.producer.publish(self.output_topic,
                                          key=event['key'],
                                          value=prediction)
                    self.consumer.commit(asynchronous=False)  # Commit after processing
                    latency_ms = (time.perf_counter()-t0)*1000
                    self.stats['latencies'].append(latency_ms)
                    self.stats['processed'] += 1

                except Exception as e:
                    logger.error(f'Processing error: {e}')
                    self.stats['errors'] += 1

                n_processed += 1
                if max_messages and n_processed >= max_messages:
                    break

        finally:
            self.consumer.close()
            self.print_stats()

    def print_stats(self) -> None:
        lats = self.stats['latencies']
        if lats:
            print(f'Processed: {self.stats["processed"]}  Errors: {self.stats["errors"]}')
            print(f'Latency: p50={np.percentile(lats,50):.1f}ms  p95={np.percentile(lats,95):.1f}ms p99={np.percentile(lats,99):.1f}ms')

    def stop(self) -> None:
        self.running = False


# ── Simulated Kafka ML pattern (without running Kafka) ────────────
print('=== Kafka ML Pattern (Simulated) ===')
print('In production:')
print('  1. Transactions → Kafka topic: raw-events')
print('  2. Feature computation service → Kafka topic: enriched-features')
print('  3. ML inference consumer → reads enriched-features')
print('  4. ML inference consumer → writes to: fraud-predictions')
print('  5. Downstream systems → consume fraud-predictions')
print('  6. Model monitoring → reads both inputs and predictions')
print()
print('Key Kafka parameters for ML:')
params = {
    'num.partitions':         '= num inference workers (for parallelism)',
    'retention.ms':           '604800000 = 7 days (for replay/retraining)',
    'max.poll.interval.ms':   '30000 = max time between polls (ML may be slow)',
    'fetch.max.bytes':        'Increase for batching multiple events per poll',
    'enable.auto.commit':     'False - commit AFTER successful inference',
    'isolation.level':        'read_committed - for exactly-once semantics',
}
for param, desc in params.items():
    print(f'  {param:<28}: {desc}')
