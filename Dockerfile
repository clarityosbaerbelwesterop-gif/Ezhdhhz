# ROUGE TITAN on your own environment (workstation, server, cloud VM).
#   docker build -t titan --build-arg TORCH=cuda .      # TORCH = none | cpu | cuda
#   docker run --gpus all -v /mnt/data:/data -v /mnt/runs:/runs titan env --path /runs
#   docker run --gpus all -v /mnt/data:/data -v /mnt/runs:/runs titan quickstart --data /data/corpus.bin \
#       --params 1b --backend auto --out /runs/pretrain-1b
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git make && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/titan
COPY . .
ARG TORCH=cpu
RUN pip install --no-cache-dir -e ".[test]" && \
    if [ "$TORCH" = "cpu" ]; then pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu; \
    elif [ "$TORCH" = "cuda" ]; then pip install --no-cache-dir torch; fi
VOLUME ["/data", "/runs"]
ENTRYPOINT ["python", "-m", "titan"]
CMD ["env"]
