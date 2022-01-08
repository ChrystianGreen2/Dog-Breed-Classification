FROM python:3.7

EXPOSE 8080

COPY class_names class_names
COPY model model
COPY haarcascade_frontalface_alt.xml haarcascade_frontalface_alt.xml

ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . .

RUN pip install -r requirements.txt

ENTRYPOINT [“streamlit”, “run”, “app.py”, “–server.port=8080”, “–server.address=0.0.0.0”]

CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 app:app
