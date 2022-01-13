# P7_TdB
PROJET 7 tableau de bord pour équipe relation client

executer localement le TdB

$ git clone https://github.com/heroku/python-getting-started.git
$ cd python-getting-started

$ python3 -m venv getting-started
$ pip install -r requirements.txt

$ createdb python_getting_started

$ python manage.py migrate
$ python manage.py collectstatic

$ heroku local

## déployer par Heroku

$ heroku open
$ heroku create
$ git push heroku main

$ heroku run python P7_App_test.py
$ heroku open
