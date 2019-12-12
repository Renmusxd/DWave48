
build:
	cd MonteCarloGen && cargo +nightly build --release
	cp MonteCarloGen/target/release/libmonte_carlo.so monte_carlo.so

