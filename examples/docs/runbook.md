# Runbook: queue backlog alert

The alert fires when the oldest message in the work queue is more than ten minutes old.

## First checks

Look at the consumer error rate first. A backlog with a high error rate means messages are
failing and being retried, not that consumers are too slow.

## Error E1042

E1042 in the consumer log means the upload token expired before the worker picked the
message up. Raise the token lifetime above the queue's visibility timeout.

## Scaling

If the error rate is normal, raise the consumer concurrency. Do not raise it above the
database connection limit divided by the number of consumer instances.
