#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
while kill -0 5856 2>/dev/null; do sleep 5; done
exec /bin/bash /workspace/pi0_experiment/scripts/adaptation/install_runtime.sh
