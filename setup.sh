# Create a virtual environment
python -m venv cifar10_classifiers_venv

# Activate the virtual environment
source ./cifar10_classifiers_venv/Scripts/activate

# Install requirements
python -m pip install --upgrade pip
python -m pip install -r ./src/requirements.txt

# deactivate
deactivate

#rm -rf cifar10_classifiers_venv