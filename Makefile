install:
	pip install -r requirements.txt
convert:
	jupyter nbconvert --to python Text.ipynb
lint:
	pylint --disable=R,C Text.py
test:
	python -m pytest -vv --cov Text test.py
format:
	black Text.py
