
debug:
	cd MonteCarloGen && cargo +nightly build

release:
	cd MonteCarloGen && cargo +nightly build --release

copy:
	rm monte_carlo.so || :
	cp MonteCarloGen/target/release/libmonte_carlo.so monte_carlo.so || :
	cp MonteCarloGen/target/release/libmonte_carlo.dylib monte_carlo.so || :

install: release copy


install-debug: debug copy
