# Decision: retries

Clients retry only on 5xx responses and network errors, never on 4xx.
Backoff starts at one second and doubles up to thirty seconds, five attempts in total.

# Decision: feature flags

New features ship behind a flag that defaults to on in staging and off in production
for the first week. The flag is removed within a month of the full rollout.
