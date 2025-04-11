#!/usr/bin/env bash

cargo build --release && cp target/release/rnoise.dll rnoise.pyd && uv run main.py
