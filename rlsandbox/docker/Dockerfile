# Runner image for grading. python:3.12-slim plus pytest, nothing else.
# Built once with network access; grading containers then run with --network none.
FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.5
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /work
