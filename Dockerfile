FROM python:3.12.2

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

# CMD ["streamlit","run", "app.py"]
EXPOSE 80

CMD ["streamlit", "run", "app.py", "--server.port", "80", "--server.enableCORS", "false"]
