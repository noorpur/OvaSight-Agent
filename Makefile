CONFIG=configs/mmotu_unet_m2.yaml

install:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt

audit:
	python -m ovasight_agent.agent audit --config $(CONFIG)

prepare:
	python -m ovasight_agent.agent prepare --config $(CONFIG)

train:
	python -m ovasight_agent.agent train --config $(CONFIG)

tune:
	python -m ovasight_agent.agent tune --config $(CONFIG)

evaluate:
	python -m ovasight_agent.agent evaluate --config $(CONFIG)

report:
	python -m ovasight_agent.agent report --config $(CONFIG)

run:
	python -m ovasight_agent.agent run --config $(CONFIG)
