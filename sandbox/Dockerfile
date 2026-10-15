FROM python:3.12.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 HOME=/work TMPDIR=/work
COPY requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir --only-binary=:all: -r /opt/requirements.txt
COPY inspect_data.py /opt/inspect_data.py
USER 10001:10001
WORKDIR /work
CMD ["python", "-c", "import time; time.sleep(86400)"]
