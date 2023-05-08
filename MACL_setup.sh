# Create a virtual environment
python3 -m venv cifar10_classifiers_venv

# Activate the virtual environment
source ./cifar10_classifiers_venv/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt

# deactivate
deactivate

#rm -rf cifar10_classifiers_venv