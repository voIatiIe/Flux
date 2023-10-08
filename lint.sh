#!/bin/bash
set -e
set -x
poetry run autoflake --in-place --recursive --remove-all-unused-imports --remove-unused-variables --remove-duplicate-keys --ignore-init-module-imports flux
poetry run isort flux
poetry run black flux
# poetry run mypy flux
