# DWave48

This project uses a rust backend for all the heavy computation, you will first need to [install rust](https://www.rust-lang.org/tools/install), then install [maturin](https://github.com/PyO3/maturin).
The backend can now be installed into the appropriate [virtual environment](https://virtualenv.pypa.io/en/latest/) using `make release` or entering the `MonteCarloGen` directory and typing `maturin develop --release`
Then you will need to install the python dependencies, listed in the requirements file (I need to add this): `pip install -r requirements.txt`

Tutorial to be added later, but calling main is typically enough.
