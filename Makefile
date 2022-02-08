# ----------------------------------
#          INSTALL & TEST
# ----------------------------------
install_requirements:
	@pip install -r requirements.txt

check_code:
	@flake8 scripts/* DR_MB/*.py

black:
	@black scripts/* DR_MB/*.py

test:
	@coverage run -m pytest tests/*.py
	@coverage report -m --omit="${VIRTUAL_ENV}/lib/python*"

ftest:
	@Write me

clean:
	@rm -f */version.txt
	@rm -f .coverage
	@rm -fr */__pycache__ */*.pyc __pycache__
	@rm -fr build dist
	@rm -fr DR_MB-*.dist-info
	@rm -fr DR_MB.egg-info

install:
	@pip install . -U

all: clean install test black check_code

count_lines:
	@find ./ -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./scripts -name '*-*' -exec  wc -l {} \; | sort -n| awk \
		        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''
	@find ./tests -name '*.py' -exec  wc -l {} \; | sort -n| awk \
        '{printf "%4s %s\n", $$1, $$2}{s+=$$0}END{print s}'
	@echo ''



run_api:
	uvicorn	api.fast:app --reload
#run_api3:
#	uvicorn	api.fast3:app3 --reload
# ----------------------------------

streamlit:
	-@streamlit run app.py
#streamlit3:
#	-@streamlit run app3.py

heroku_login:
	-@heroku login

heroku_upload_public_key:
	-@heroku keys:add ~/.ssh/id_ed25519.pub

#NN: APP_NAME = diabetic-retinopathy-nn
heroku_create_app:
	-@heroku create --ssh-git ${dr-martin-v1}

deploy_heroku:
	-@git push heroku master
	-@heroku ps:scale web=1
