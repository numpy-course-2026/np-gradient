#!/usr/bin/env bash
# Проверки проекта «Градиент» курса «NumPy: от массива до нейросети».
#
#   bash checks/run.sh           — проверить все тикеты
#   bash checks/run.sh grd2      — проверить один тикет
#   bash checks/run.sh --score   — только итог: «заработано максимум»
#   bash checks/run.sh --report  — для GitHub Actions: подробный вывод, сводка и баллы
#
# То же самое делают `make check` и `make check-grd2`.
#
# Проверка каждого тикета — небольшой скрипт checks/grdN/check.sh: его можно
# и нужно читать, он вызывает обычный pytest. Не изменяйте каталог checks/:
# работа с изменёнными проверками не засчитывается.

set -u
cd "$(dirname "$0")/.." || exit 2
export IN_CI=${GITHUB_ACTIONS:-}

# Python берём из окружения uv (`uv sync`), а без uv — python3, если в нём есть numpy и pytest.
if command -v uv >/dev/null 2>&1; then
  PY="uv run --quiet --frozen python"
elif python3 -c 'import numpy, pytest' >/dev/null 2>&1; then
  PY=python3
else
  echo "Не найден uv: установите его — https://docs.astral.sh/uv/ — и выполните uv sync"; exit 2
fi
export PY

if [ -t 1 ]; then
  RED=$(printf '\033[31m'); GREEN=$(printf '\033[32m')
  BOLD=$(printf '\033[1m'); DIM=$(printf '\033[2m'); OFF=$(printf '\033[0m')
else
  RED=; GREEN=; BOLD=; DIM=; OFF=
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export TMP

# ---- Функции для checks/grdN/check.sh ---------------------------------------
# Скрипт проверки пишет пояснения в поток 3 (say/fail); вывод pytest
# попадает в отчёт только при неудаче.

say() { printf '%s\n' "$*" >&3; }                      # пояснение в отчёт
fail() { say "  ✗ $*"; exit 1; }                        # проверка не пройдена
note() { say "  · $*"; }                                # заметка без вердикта
ok() { say "  ✓ $*"; }

quiet() { # название команда… -> выполнить; при неудаче показать хвост вывода и провалить проверку
  local title=$1 log; shift
  log=$(mktemp "$TMP/cmd.XXXXXX")
  if "$@" >"$log" 2>&1; then ok "$title"; return 0; fi
  say "  ✗ $title"
  tail -n 40 "$log" | sed 's/^/    /' >&3
  exit 1
}
pytest_run() { # название аргументы-pytest… -> pytest; при неудаче — хвост отчёта
  local title=$1; shift
  # shellcheck disable=SC2086
  quiet "$title" $PY -m pytest --tb=short "$@"
}
no_loops() { # файл функция[:N]… -> в функциях нет Python-циклов по элементам
  local file=$1; shift
  [ -f "$file" ] || fail "нет файла $file"
  # shellcheck disable=SC2086
  quiet "без циклов по элементам: $*" $PY checks/loops.py "$file" "$@"
}
answer() { # файл ключ -> значение строки «ключ: значение»
  tr -d '\r' <"$1" | sed -n "s/^$2:[[:space:]]*//p" | head -n1 | sed 's/[[:space:]`]*$//; s/^`//'
}
sha256() { # stdin -> sha256 (shasum в macOS, sha256sum в Linux)
  if command -v sha256sum >/dev/null 2>&1; then sha256sum | cut -d' ' -f1; else shasum -a 256 | cut -d' ' -f1; fi
}
sha() { printf '%s' "$1" | sha256; }
answer_is() { # файл ключ sha256-правильного-ответа -> ответ совпал (регистр и все пробелы не важны)
  local a; a=$(answer "$1" "$2" | tr 'A-Z' 'a-z' | tr -d ' ')
  [ -n "$a" ] || { say "  ✗ в $1 не заполнен ответ «$2»"; return 1; }
  [ "$(sha "$a")" = "$3" ] || { say "  ✗ ответ «$2» неверный"; return 1; }
  say "  ✓ ответ «$2»"
}
unchanged() { # файл sha256 -> файл не изменён (для файлов, которые трогать нельзя)
  [ "$(sha256 <"$1")" = "$2" ] || fail "файл $1 изменён — верните его: git checkout -- $1"
}
export -f say fail note ok quiet pytest_run no_loops answer sha256 sha answer_is unchanged

# ---- Запуск проверок --------------------------------------------------------
read_tasks() {
  grep -v '^#' checks/tasks.txt | grep -v '^[[:space:]]*$'
}

run_task() { # task -> 0 пройдено, 1 нет
  local task=$1 log="$TMP/$1.log"
  if [ ! -f "checks/$task/check.sh" ]; then echo "  нет checks/$task/check.sh" >"$log"; return 1; fi
  bash "checks/$task/check.sh" >"$log" 2>&1 3>&1 </dev/null
}

MODE=${1:-}
ONLY=
case "$MODE" in --score|--report) ;; "") ;; *) ONLY=$MODE; MODE=;; esac

earned=0; max=0; rows=""
while read -r task points title; do
  [ -n "$ONLY" ] && [ "$ONLY" != "$task" ] && continue
  max=$((max + points))
  run_task "$task"; rc=$?
  [ "$MODE" != "--score" ] && echo "${BOLD}$title${OFF}"
  [ "$MODE" != "--score" ] && cat "$TMP/$task.log"
  if [ $rc -eq 0 ]; then
    earned=$((earned + points)); rows="$rows| ✅ | $title | $points из $points |
"
    [ "$MODE" != "--score" ] && echo "  ${GREEN}$points из $points баллов${OFF}"
  else
    rows="$rows| ❌ | $title | 0 из $points |
"
    [ "$MODE" != "--score" ] && echo "  ${RED}0 из $points баллов${OFF}"
  fi
done < <(read_tasks)

if [ "$MODE" = "--score" ]; then
  echo "$earned $max"
  exit 0
fi
echo
echo "${BOLD}Итого: $earned из $max баллов${OFF}"
if [ -n "$IN_CI" ]; then
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ]; then
    { echo "### Автотесты: $earned из $max баллов"; echo
      echo "| | Задание | Баллы |"; echo "|---|---|---|"; printf '%s' "$rows"; } >>"$GITHUB_STEP_SUMMARY"
  fi
  if [ -n "${GITHUB_OUTPUT:-}" ]; then
    { echo "earned=$earned"; echo "max=$max"; } >>"$GITHUB_OUTPUT"
  fi
else
  echo "${DIM}То же самое GitHub Actions запустит после git push.${OFF}"
fi
[ "$earned" = "$max" ]
