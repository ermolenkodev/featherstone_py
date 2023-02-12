# featherstone_py

This project is a Python implementation of rigid body dynamics algorithms, based on Roy Featherstone's book,
"Modern Robotics: Mechanics, Planning, and Control", and the MEE5114 course.
This project was created primarily for self-education but can also help those who going to enter the field
of computational rigid body dynamics and prefer the Python ecosystem over MATLAB/Julia or C++.

The examples in this project can help to clarify the differences in notation between Featherstone's book and the "Modern Robotics" book,
which can be quite confusing.
Please check out the example models and test cases for more details.Please check out the
[example models](featherstone_py/example_models.py) and [test cases](test/test_inverse_dynamics.py) for more details.

### Installation
```bash
git clone <this repo>
cd featherstone_py
pip install -r requirements.txt
pip install -e .
```

### Run tests
```bash
pip install -r test/requirements.txt
python -m pytest test
```

### Prepare environment for example notebooks
*NOTE: Only tested on Ubuntu*
```bash
pip install -r examples/requirements.txt
```

Install drake from apt using [official instructions](https://drake.mit.edu/apt.html#stable-releases)

### Note about the License
This project closely follows the API and notation used in spatial v2, and as a result,
it is considered a derivative work of this library and is licensed with GPL v3.