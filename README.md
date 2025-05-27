# Installation

Install [gp](https://github.com/ajjimeno/gp) and the [fast evolutionary evaluation](https://github.com/ajjimeno/fast-evolutionary-evaluation).

Ensure that the `PYTHONPATH` variable points to the `gp` tool by running:

```
export PYTHONPATH="/path/to/gp:$PYTHONPATH"
```

or more permanently

```
echo 'export PYTHONPATH="/path/to/gp:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

Install packages required for llm-gp
```
pip install -r requirements.txt
```

# Running

Set up the tool's environment variables in `.env` file.
The environment variables set the running parameters.
Update the `DATA_FOLDER` variable with the location of the data set.
An example data set is availble [here](https://github.com/ajjimeno/list-data).

```
python gp.py
```

# Citation

If you use this software, cite as follow:

```
@article{yepes2025evolutionary,
  title={Evolutionary thoughts: integration of large language models and evolutionary algorithms},
  author={Yepes, Antonio Jimeno and Barnard, Pieter},
  journal={arXiv preprint arXiv:2505.05756},
  year={2025}
}
```
