#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
set -eo pipefail
cd /root/maniskill_myws
pty bash scripts/pld/setup_libero.sh 2>&1
