FROM python:3.10-slim-buster

WORKDIR /StockTrain

COPY . .

RUN pip install pipenv
RUN pipenv sync

CMD ["pipenv", "run", "python3", "-m" , "flask", "run", "--host=0.0.0.0"]