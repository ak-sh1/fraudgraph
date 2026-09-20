# Multi-stage build for FraudGraph

# Stage 1: Python pipeline (data generation)
FROM python:3.11-slim AS python-builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY data/ data/

# Generate demo data (optional - can be pre-generated)
# RUN python -m src.generate.synthetic && \
#     python -m src.features.build && \
#     python -m src.models.train && \
#     python -m src.eval.evaluate && \
#     python -m src.export_demo

# Stage 2: Next.js app
FROM node:22-alpine

WORKDIR /app

COPY package*.json ./
RUN npm ci --production

COPY . .
COPY --from=python-builder /app/data/demo/ /app/public/data/

RUN npm run build

ENV PORT=8080
EXPOSE 8080

CMD ["npm", "start"]
