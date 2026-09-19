#!/usr/bin/env bash
# Забрать новые тикеты и проверки из шаблона проекта: make update.
#
# Шаблон np-gradient публичный: в нём нет решений, только тикеты, данные и проверки.
# Обновляются tickets/, checks/, data/, README.md, Makefile и workflow проверок.
# Ваш код (gradient/), pyproject.toml и uv.lock не трогаются.
set -euo pipefail
cd "$(dirname "$0")/.."
URL=${TEMPLATE_URL:-https://codeload.github.com/numpy-course-2026/np-gradient/tar.gz/refs/heads/main}

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL "$URL" | tar xz -C "$tmp" --strip-components=1

for d in tickets checks data; do
  rm -rf "$d"
  cp -R "$tmp/$d" "$d"
done
cp "$tmp/README.md" "$tmp/Makefile" .
mkdir -p .github/workflows
cp "$tmp/.github/workflows/checks.yml" .github/workflows/checks.yml

echo "Готово: тикетов в tickets/ — $(ls tickets | wc -l | tr -d ' ')."
if git rev-parse --git-dir >/dev/null 2>&1; then
  git status --short -- tickets checks data README.md Makefile .github
  echo "Закоммитьте обновление отдельным коммитом: git add -A tickets checks data README.md Makefile .github && git commit -m 'Новые тикеты из шаблона'"
fi
