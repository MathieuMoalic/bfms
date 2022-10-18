default:
	. ./.env/bin/activate && maturin develop && pytest -rP
