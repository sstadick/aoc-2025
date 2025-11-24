#!/usr/bin/env bash

day="$1"
cp -r ./template "./${day}"
sed -i "s/template/${day}/" "./${day}/pixi.toml"
rm "./${day}/pixi.lock" || true
