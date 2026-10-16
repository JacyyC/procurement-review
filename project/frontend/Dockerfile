# Frontend delivery image for the V1-04 Compose workbench.
#
# Build context is `project/frontend/`.  `npm ci` installs strictly from
# `package-lock.json` (a mismatch between package.json and the lock fails the build),
# and the preview server keeps the same-origin `/api` proxy the e2e suite already uses.

FROM node:22-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS builder

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

COPY tsconfig.json tsconfig.app.json tsconfig.node.json vite.config.ts index.html ./
COPY src ./src
RUN npm run build


FROM node:22-slim@sha256:83f487e0a63425e5b4d146fb5e5be574bcbe1b7b843d3ebafdd95eaf7767a7e5 AS runtime

ENV NODE_ENV=production

RUN printf '%s\n' 'app:x:10001:10001::/nonexistent:/usr/sbin/nologin' >> /etc/passwd \
    && printf '%s\n' 'app:x:10001:' >> /etc/group

WORKDIR /app

# `vite preview` is part of the frozen toolchain that produced the bundle, so the server
# and the build share exactly the locked dependency tree.
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package.json ./package.json
COPY --from=builder /app/package-lock.json ./package-lock.json
COPY --from=builder /app/vite.config.ts ./vite.config.ts

# The checkout may carry restrictive umask modes, so make the delivered tree readable
# for the unprivileged `app` user.  Vite bundles its own config into
# `node_modules/.vite-temp` at startup, so those cache directories must be writable.
RUN chmod -R a+rX /app \
    && mkdir -p /app/node_modules/.vite-temp /app/node_modules/.vite \
    && chown -R app:app /app/node_modules/.vite-temp /app/node_modules/.vite

USER app

EXPOSE 5173

CMD ["node", "node_modules/vite/bin/vite.js", "preview", "--host", "0.0.0.0", "--port", "5173"]
