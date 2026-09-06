#!/usr/bin/env bash
# InterServer VPS setup for Algo_Trading (see docs/vps_setup.md).
#
# Run on the VPS as root:
#   curl -fsSL ... | bash   # or copy repo and run:
#   sudo bash scripts/vps_setup.sh
#
# Examples:
#   bash scripts/vps_setup.sh                  # full setup
#   bash scripts/vps_setup.sh --mysql-only      # only ensure MySQL db/user
#   bash scripts/vps_setup.sh --check-mysql    # verify db/user/password only
#   bash scripts/vps_setup.sh --skip-clone --skip-systemd
#
# Override defaults via env:
#   REPO_DIR=/root/Algo_Trading MYSQL_ROOT_PASSWORD=secret bash scripts/vps_setup.sh

set -euo pipefail

# --- Config (override with env vars) ---
REPO_DIR="${REPO_DIR:-${HOME}/Algo_Trading}"
GITHUB_REPO="${GITHUB_REPO:-git@github.com:yyeshetch/Algo_Trading.git}"
GIT_BRANCH="${GIT_BRANCH:-main}"

MYSQL_DATABASE="${MYSQL_DATABASE:-stocks_analysis}"
MYSQL_USER="${MYSQL_USER:-algo}"
# Single-quoted default avoids bash expanding $89 in the password.
readonly _DEFAULT_MYSQL_PASSWORD='Round#Sky$89'
MYSQL_PASSWORD="${MYSQL_PASSWORD:-${_DEFAULT_MYSQL_PASSWORD}}"

DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
MYSQL_ROOT_PASSWORD="${MYSQL_ROOT_PASSWORD:-}"

# --- Flags ---
DO_MYSQL=1
DO_PACKAGES=1
DO_CLONE=1
DO_VENV=1
DO_ENV=1
DO_INIT_DB=1
DO_SYSTEMD=1
DO_FIREWALL=0
CHECK_ONLY=0
MYSQL_ONLY=0

usage() {
  cat <<'EOF'
Usage: vps_setup.sh [OPTIONS]

Automates docs/vps_setup.md on an Ubuntu/Debian VPS (run as root).

Options:
  --all              Full setup (default)
  --mysql-only       Install/start MySQL + ensure database/user only
  --check-mysql      Check database/user/login; create if missing (no apt/install)
  --skip-mysql       Skip MySQL steps
  --skip-clone       Skip git clone/pull
  --skip-venv        Skip Python venv + pip install
  --skip-env         Skip .env MySQL block update
  --skip-init-db     Skip --init-db / schema load
  --skip-systemd     Skip systemd unit install
  --with-firewall    Open ports 22 and 8000 via ufw
  -h, --help         Show this help

Environment:
  REPO_DIR, GITHUB_REPO, GIT_BRANCH
  MYSQL_DATABASE, MYSQL_USER, MYSQL_PASSWORD
  MYSQL_ROOT_PASSWORD   (if root needs password; else uses socket auth)
EOF
}

log()  { printf '[vps-setup] %s\n' "$*"; }
warn() { printf '[vps-setup] WARN: %s\n' "$*" >&2; }
die()  { printf '[vps-setup] ERROR: %s\n' "$*" >&2; exit 1; }

parse_args() {
  local any=0
  for arg in "$@"; do
    case "$arg" in
      --all) any=1 ;;
      --mysql-only)
        any=1
        MYSQL_ONLY=1
        DO_PACKAGES=0
        DO_CLONE=0
        DO_VENV=0
        DO_ENV=0
        DO_INIT_DB=0
        DO_SYSTEMD=0
        ;;
      --check-mysql)
        any=1
        CHECK_ONLY=1
        DO_PACKAGES=0
        DO_CLONE=0
        DO_VENV=0
        DO_ENV=0
        DO_INIT_DB=0
        DO_SYSTEMD=0
        ;;
      --skip-mysql) DO_MYSQL=0 ;;
      --skip-clone) DO_CLONE=0 ;;
      --skip-venv) DO_VENV=0 ;;
      --skip-env) DO_ENV=0 ;;
      --skip-init-db) DO_INIT_DB=0 ;;
      --skip-systemd) DO_SYSTEMD=0 ;;
      --with-firewall) DO_FIREWALL=1 ;;
      -h|--help) usage; exit 0 ;;
      *) die "Unknown option: $arg (try --help)" ;;
    esac
  done
  if [[ "$any" -eq 0 && "$#" -gt 0 ]]; then
    : # explicit skip flags only — still run remaining steps
  fi
}

require_root() {
  if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
    die "Run as root on the VPS (e.g. sudo bash scripts/vps_setup.sh)"
  fi
}

require_apt() {
  command -v apt-get >/dev/null 2>&1 || die "apt-get not found — this script targets Ubuntu/Debian VPS"
}

# Escape single quotes for SQL string literals.
sql_escape() {
  printf "%s" "$1" | sed "s/'/''/g"
}

# Run mysql as root (socket, sudo, or password).
run_mysql_root() {
  local sql="$1"
  if [[ -n "$MYSQL_ROOT_PASSWORD" ]]; then
    mysql -u root --password="${MYSQL_ROOT_PASSWORD}" --batch --skip-column-names -e "$sql"
  elif mysql -u root --batch --skip-column-names -e "SELECT 1" >/dev/null 2>&1; then
    mysql -u root --batch --skip-column-names -e "$sql"
  elif sudo mysql -u root --batch --skip-column-names -e "SELECT 1" >/dev/null 2>&1; then
    sudo mysql -u root --batch --skip-column-names -e "$sql"
  else
    die "Cannot connect as MySQL root. Set MYSQL_ROOT_PASSWORD or run mysql_secure_installation first."
  fi
}

run_mysql_root_file() {
  local file="$1"
  if [[ -n "$MYSQL_ROOT_PASSWORD" ]]; then
    mysql -u root --password="${MYSQL_ROOT_PASSWORD}" --batch < "$file"
  elif mysql -u root --batch -e "SELECT 1" >/dev/null 2>&1; then
    mysql -u root --batch < "$file"
  elif sudo mysql -u root --batch -e "SELECT 1" >/dev/null 2>&1; then
    sudo mysql -u root --batch < "$file"
  else
    die "Cannot connect as MySQL root."
  fi
}

mysql_database_exists() {
  local count
  count="$(run_mysql_root "SELECT COUNT(*) FROM information_schema.SCHEMATA WHERE SCHEMA_NAME = '$(sql_escape "$MYSQL_DATABASE")';")"
  [[ "${count:-0}" -gt 0 ]]
}

mysql_user_exists() {
  local count
  count="$(run_mysql_root "SELECT COUNT(*) FROM mysql.user WHERE user = '$(sql_escape "$MYSQL_USER")' AND host = 'localhost';")"
  [[ "${count:-0}" -gt 0 ]]
}

mysql_app_login_ok() {
  mysql -u "$MYSQL_USER" --password="${MYSQL_PASSWORD}" "$MYSQL_DATABASE" \
    --batch --skip-column-names -e "SELECT 1" >/dev/null 2>&1
}

install_mysql_server() {
  require_apt
  log "Installing MySQL server…"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y mysql-server
  systemctl enable mysql
  systemctl start mysql
  systemctl is-active --quiet mysql || die "MySQL failed to start"
  log "MySQL is running"
}

ensure_mysql_database_and_user() {
  local pw_sql
  pw_sql="$(sql_escape "$MYSQL_PASSWORD")"

  if mysql_database_exists; then
    log "Database '${MYSQL_DATABASE}' already exists"
  else
    log "Creating database '${MYSQL_DATABASE}'…"
  fi

  if mysql_user_exists; then
    log "User '${MYSQL_USER}'@'localhost' already exists"
    if mysql_app_login_ok; then
      log "App user login verified"
    else
      log "Updating password for '${MYSQL_USER}'@'localhost'…"
    fi
  else
    log "Creating user '${MYSQL_USER}'@'localhost'…"
  fi

  local tmp
  tmp="$(mktemp)"
  cat >"$tmp" <<SQL
CREATE DATABASE IF NOT EXISTS \`${MYSQL_DATABASE}\`
  CHARACTER SET utf8mb4
  COLLATE utf8mb4_unicode_ci;

CREATE USER IF NOT EXISTS '${MYSQL_USER}'@'localhost' IDENTIFIED BY '${pw_sql}';
ALTER USER '${MYSQL_USER}'@'localhost' IDENTIFIED BY '${pw_sql}';

GRANT ALL PRIVILEGES ON \`${MYSQL_DATABASE}\`.* TO '${MYSQL_USER}'@'localhost';
FLUSH PRIVILEGES;
SQL
  run_mysql_root_file "$tmp"
  rm -f "$tmp"

  if mysql_app_login_ok; then
    log "MySQL ready: database='${MYSQL_DATABASE}' user='${MYSQL_USER}' (login OK)"
  else
    die "Database/user created but login failed for '${MYSQL_USER}' — check password"
  fi

  run_mysql_root "SHOW DATABASES LIKE '${MYSQL_DATABASE}';"
  run_mysql_root "SELECT user, host FROM mysql.user WHERE user = '${MYSQL_USER}';"
}

check_mysql_status() {
  log "Checking MySQL database, user, and password…"
  if ! command -v mysql >/dev/null 2>&1; then
    warn "mysql client not installed"
    return 1
  fi
  if ! systemctl is-active --quiet mysql 2>/dev/null; then
    warn "MySQL service is not running"
    return 1
  fi

  local db_ok=0 user_ok=0 login_ok=0
  mysql_database_exists && db_ok=1
  mysql_user_exists && user_ok=1
  mysql_app_login_ok && login_ok=1

  printf '  database %-20s %s\n' "'${MYSQL_DATABASE}'" "$([[ "$db_ok" -eq 1 ]] && echo OK || echo MISSING)"
  printf '  user     %-20s %s\n' "'${MYSQL_USER}'@'localhost'" "$([[ "$user_ok" -eq 1 ]] && echo OK || echo MISSING)"
  printf '  login    %-20s %s\n' "(password check)" "$([[ "$login_ok" -eq 1 ]] && echo OK || echo FAIL)"

  [[ "$db_ok" -eq 1 && "$user_ok" -eq 1 && "$login_ok" -eq 1 ]]
}

install_base_packages() {
  require_apt
  log "Installing base packages (openssh-server, git, python3)…"
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    openssh-server git python3 python3-venv python3-pip curl
  systemctl enable ssh >/dev/null 2>&1 || true
  systemctl start ssh >/dev/null 2>&1 || true
}

clone_or_update_repo() {
  if [[ -d "${REPO_DIR}/.git" ]]; then
    log "Updating repo at ${REPO_DIR}…"
    git -C "$REPO_DIR" fetch origin
    git -C "$REPO_DIR" checkout "$GIT_BRANCH"
    git -C "$REPO_DIR" pull --ff-only origin "$GIT_BRANCH" || git -C "$REPO_DIR" pull origin "$GIT_BRANCH"
  else
    log "Cloning ${GITHUB_REPO} → ${REPO_DIR}…"
    git clone "$GITHUB_REPO" "$REPO_DIR"
    git -C "$REPO_DIR" checkout "$GIT_BRANCH"
  fi
}

setup_python_venv() {
  log "Creating Python venv and installing requirements…"
  cd "$REPO_DIR"
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
}

ensure_env_mysql_block() {
  local env_file="${REPO_DIR}/.env"
  log "Ensuring MySQL settings in ${env_file}…"
  touch "$env_file"
  chmod 600 "$env_file"

  set_env_kv() {
    local key="$1" val="$2" file="$3" tmp="${file}.tmp.$$"
    touch "$file"
    if grep -q "^${key}=" "$file" 2>/dev/null; then
      KEY="$key" VAL="$val" awk -F= '
        BEGIN { OFS = "=" }
        $1 == ENVIRON["KEY"] { print ENVIRON["KEY"], ENVIRON["VAL"]; next }
        { print }
      ' "$file" >"$tmp"
    else
      cp "$file" "$tmp"
      printf '%s=%s\n' "$key" "$val" >>"$tmp"
    fi
    mv "$tmp" "$file"
  }

  set_env_kv "MYSQL_HOST" "localhost" "$env_file"
  set_env_kv "MYSQL_PORT" "3306" "$env_file"
  set_env_kv "MYSQL_DATABASE" "$MYSQL_DATABASE" "$env_file"
  set_env_kv "MYSQL_USER" "$MYSQL_USER" "$env_file"
  set_env_kv "MYSQL_PASSWORD" "$MYSQL_PASSWORD" "$env_file"
  set_env_kv "STORAGE_BACKEND" "write_to_db" "$env_file"
  set_env_kv "DATA_DIR" "data" "$env_file"

  if ! grep -q "^KITE_API_KEY=" "$env_file"; then
    warn ".env missing KITE_API_KEY — copy credentials from your Mac (see docs/vps_setup.md §5)"
  fi
}

run_init_db() {
  log "Running --init-db to create application tables…"
  cd "$REPO_DIR"
  # shellcheck disable=SC1091
  source .venv/bin/activate
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
  PYTHONPATH=src python -m intraday_engine.cli.main --init-db
  mysql -u "$MYSQL_USER" --password="${MYSQL_PASSWORD}" "$MYSQL_DATABASE" -e "SHOW TABLES;"
}

install_systemd_services() {
  log "Installing systemd units (algo-scheduler, algo-dashboard)…"
  local py="${REPO_DIR}/.venv/bin/python"
  [[ -x "$py" ]] || die "Python venv not found at ${py} — run without --skip-venv first"

  tee /etc/systemd/system/algo-scheduler.service >/dev/null <<EOF
[Unit]
Description=Algo Trading session scheduler (writes to MySQL)
After=network-online.target mysql.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${REPO_DIR}
Environment=PYTHONPATH=src
EnvironmentFile=${REPO_DIR}/.env
ExecStart=${py} -m intraday_engine.cli.main --session-scheduler --storage write_to_db
Restart=always
RestartSec=15
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  tee /etc/systemd/system/algo-dashboard.service >/dev/null <<EOF
[Unit]
Description=Algo Trading dashboard (read-only, MySQL)
After=network-online.target algo-scheduler.service
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=${REPO_DIR}
Environment=PYTHONPATH=src
EnvironmentFile=${REPO_DIR}/.env
ExecStart=${py} -m intraday_engine.cli.main --dashboard --read-only --host 0.0.0.0 --port ${DASHBOARD_PORT}
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  systemctl enable algo-scheduler algo-dashboard
  systemctl restart algo-scheduler algo-dashboard || warn "Services installed but start failed — check journalctl"
  systemctl status algo-scheduler --no-pager -l || true
  systemctl status algo-dashboard --no-pager -l || true
}

configure_firewall() {
  command -v ufw >/dev/null 2>&1 || { warn "ufw not installed — skipping firewall"; return; }
  log "Configuring UFW (22, ${DASHBOARD_PORT})…"
  ufw allow 22/tcp
  ufw allow "${DASHBOARD_PORT}"/tcp
  ufw --force enable
  ufw status
}

print_next_steps() {
  cat <<EOF

[vps-setup] Done.

Next steps (if not automated above):
  1. Add GitHub deploy key on VPS:  ssh-keygen … && cat ~/.ssh/id_ed25519.pub
  2. Fill Kite credentials in ${REPO_DIR}/.env
  3. Dashboard: http://$(hostname -I 2>/dev/null | awk '{print $1}'):${DASHBOARD_PORT}
  4. Logs: journalctl -u algo-scheduler -f

MySQL:
  mysql -u ${MYSQL_USER} -p ${MYSQL_DATABASE}
  (password is set in this script / .env MYSQL_PASSWORD)

EOF
}

main() {
  parse_args "$@"
  require_root

  log "Algo_Trading VPS setup — repo=${REPO_DIR}"

  if [[ "$CHECK_ONLY" -eq 1 ]]; then
    if check_mysql_status; then
      log "All MySQL checks passed"
      exit 0
    fi
    log "Some checks failed — attempting to create/fix…"
    ensure_mysql_database_and_user
    check_mysql_status
    exit 0
  fi

  if [[ "$DO_PACKAGES" -eq 1 && "$MYSQL_ONLY" -eq 0 ]]; then
    install_base_packages
  fi

  if [[ "$DO_MYSQL" -eq 1 ]]; then
    if ! command -v mysql >/dev/null 2>&1; then
      install_mysql_server
    else
      systemctl enable mysql 2>/dev/null || true
      systemctl start mysql 2>/dev/null || true
    fi
    check_mysql_status || true
    ensure_mysql_database_and_user
    check_mysql_status
  fi

  if [[ "$DO_CLONE" -eq 1 && "$MYSQL_ONLY" -eq 0 ]]; then
    clone_or_update_repo
  fi

  if [[ "$DO_VENV" -eq 1 && "$MYSQL_ONLY" -eq 0 ]]; then
    [[ -f "${REPO_DIR}/requirements.txt" ]] || die "Repo not found at ${REPO_DIR} — clone first or set REPO_DIR"
    setup_python_venv
  fi

  if [[ "$DO_ENV" -eq 1 && "$MYSQL_ONLY" -eq 0 ]]; then
    ensure_env_mysql_block
  fi

  if [[ "$DO_INIT_DB" -eq 1 && "$MYSQL_ONLY" -eq 0 ]]; then
    run_init_db
  fi

  if [[ "$DO_SYSTEMD" -eq 1 && "$MYSQL_ONLY" -eq 0 ]]; then
    install_systemd_services
  fi

  if [[ "$DO_FIREWALL" -eq 1 ]]; then
    configure_firewall
  fi

  print_next_steps
}

main "$@"
