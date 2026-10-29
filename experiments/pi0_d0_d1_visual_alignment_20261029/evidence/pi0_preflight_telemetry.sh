#!/bin/bash
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
exec nvidia-smi --query-gpu=timestamp,name,memory.total,memory.used,memory.free,temperature.gpu,temperature.memory,power.draw,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total --format=csv -l 5
