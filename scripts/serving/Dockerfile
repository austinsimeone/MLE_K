FROM python:3.9

ADD src /app/src
ADD requirements.txt /app/requirements.txt
ADD setup.py /app/setup.py
ADD setup.cfg /app/setup.cfg

WORKDIR /app
RUN pip install -r requirements.txt

CMD ["uvicorn", "scripts.serving.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
