.PHONY: mlflow exercise-1 exercise-2 exercise-3a exercise-3b exercise-4 exercise-5 exercise-6 exercise-7 exercise-7b reset

# --- MLflow UI ---
mlflow:
	mlflow ui

# --- Exercises ---
exercise-1:
	python exercises/exercise-1.py

exercise-2:
	python exercises/exercise-2.py

exercise-3a:
	python exercises/exercise-3a.py

exercise-3b:
	python exercises/exercise-3b.py

exercise-4:
	python exercises/exercise-4.py

exercise-5:
	python exercises/exercise-5.py

exercise-6:
	python exercises/exercise-6.py

exercise-7:
	python exercises/exercise-7.py

exercise-7b:
	python exercises/exercise-7b.py

# --- Reset MLflow data ---
reset:
	rm -rf mlruns/
	@echo "MLflow data cleared. Restart with: make mlflow"
