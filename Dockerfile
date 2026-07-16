# openSUSE BCI Python base, per standing container policy.
# Justification: this job needs only CPython plus Pillow (all HTTP is stdlib
# urllib), so the minimal BCI Python image is a clean, policy-compliant fit.
# No GPU libraries are needed here - the model runs in your Ollama server; this
# container is just the orchestration client that talks to it over HTTP.
FROM registry.suse.com/bci/python:3.13

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ocr_books.py .

# Input books mount at /data; Ollama host is passed at runtime.
# Built/run with nerdctl under Lima (containerd) - see README for host reachability.
# Example:
#   nerdctl run --rm -v "$PWD:/data" -e OLLAMA_HOST=http://<ollama-host>:11434 \
#       book-ocr --input /data
ENTRYPOINT ["python", "ocr_books.py"]
CMD ["--input", "/data"]
