FROM python:3.11-slim

WORKDIR /app

COPY flask_app/ /app/

COPY ml-pipeline-imdb-movies-review/models/vectorizer.pkl /app/models/vectorizer.pkl

RUN pip install -r requirements.txt

RUN python -m nltk.downloader stopwords wordnet omw-1.4

EXPOSE 5000

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]