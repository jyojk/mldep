FROM python:3.9.7

COPY . /app

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python3", "app.py", "--host=0.0.0.0", "--port=5000", "gunicorn"]