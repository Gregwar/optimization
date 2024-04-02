#!/bin/bash

killall -2 php
php -S localhost:8080 &

# npx @marp-team/marp-cli@latest -w --html linear.md
# npx @marp-team/marp-cli@latest -w --html gradient.md
npx @marp-team/marp-cli@latest -w --html newton.md
