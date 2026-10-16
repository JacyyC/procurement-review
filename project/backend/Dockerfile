# Backend delivery image for the V1-04 Compose workbench.
#
# Build context is `project/` (see ../docker-compose.yml) so the image can hold both the
# locked backend and the shared `scripts/` bootstrap.  Every base image is pinned by
# digest: the tag is documentation, the digest is the contract (Spec §5.2).
#
# Two stages: `builder` resolves the locked uv environment once, `runtime` only carries
# the resolved virtual environment plus the sources — no compiler, no uv cache, no
# credentials, no `.local/` state.

FROM ghcr.io/astral-sh/uv:0.11.1@sha256:fc93e9ecd7218e9ec8fba117af89348eef8fd2463c50c13347478769aaedd0ce AS uv


FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254 AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv

COPY --from=uv /uv /uvx /usr/local/bin/

WORKDIR /app/backend

# Dependency layer first: it only changes when uv.lock or pyproject.toml change.
COPY backend/pyproject.toml backend/uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

# Project layer: the locked sources are installed from the same lock file.
COPY backend/src ./src
RUN uv sync --locked --no-dev

# The delivered image must also be able to prove its own lock is intact.
RUN uv lock --check --offline


FROM python:3.12-slim-bookworm@sha256:782412e85d0f0984994c290652577d4018aff08145c85b262bb63dc0c7522254 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin

# A fixed unprivileged identity without pulling in extra packages: the slim base has no
# `useradd`, and the delivery must not need an apt layer it cannot pin.
RUN printf '%s\n' 'app:x:10001:10001::/nonexistent:/usr/sbin/nologin' >> /etc/passwd \
    && printf '%s\n' 'app:x:10001:' >> /etc/group

COPY --from=uv /uv /uvx /usr/local/bin/
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app/backend
COPY backend/pyproject.toml backend/uv.lock ./
COPY backend/src ./src
COPY scripts /app/scripts
# The checkout may carry restrictive umask modes (git tracks only the executable bit),
# so the unprivileged `app` user must be able to read everything that was copied in.
RUN chmod -R a+rX /app /opt/venv \
    && chmod 0755 /app/scripts/*.sh /app/scripts/*.py

# The API and web processes run unprivileged; the one-shot `init` service overrides this
# to root because it alone must write the protected credential file onto the host path.
USER app

EXPOSE 8000

ENTRYPOINT ["/app/scripts/container_entrypoint.sh"]
CMD ["uvicorn", "procurement_review.api:app", "--host", "0.0.0.0", "--port", "8000"]
