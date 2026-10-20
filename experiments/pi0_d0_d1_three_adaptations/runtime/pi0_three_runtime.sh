#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
while pgrep -f '^uv sync --frozen --no-dev --python 3.11$' >/dev/null; do sleep 5; done
exec /bin/bash /workspace/pi0_experiment/scripts/adaptation/install_runtime.sh
