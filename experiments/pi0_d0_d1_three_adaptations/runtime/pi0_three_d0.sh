#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
exec /bin/bash /workspace/pi0_experiment/scripts/adaptation/start_d0.sh
