FROM python:3.8.12
WORKDIR /app
COPY requirements.txt .
COPY . .
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt stopwords

CMD ["python", "./api.py"]

EXPOSE 5000
EXPOSE 443