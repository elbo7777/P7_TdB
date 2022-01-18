# P7_TdB
PROJET 7 tableau de bord pour équipe relation client

## executer localement le TdB
```sh
$ git clone https://github.com/elbo7777/P7_TdB.git
cd P7_TdB
$ python3 -m venv P7_TdB
$ pip install -r requirements.txt
$ python P7_App_test.py
$ heroku local
```

## déployer par Heroku
```sh
$ heroku open
$ heroku create
$ git push heroku main
$ heroku run python P7_App_test.py
$ heroku open
```
## éxécution sur le web en production via heroku : 
- pour accéder au tableau de bord : https://p7tdb.herokuapp.com
