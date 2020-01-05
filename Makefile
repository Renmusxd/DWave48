
debug:
	cd MonteCarloGen && activate dwave && maturin develop

release:
	cd MonteCarloGen && activate dwave && maturin develop --release

install: release

install-debug: debug
