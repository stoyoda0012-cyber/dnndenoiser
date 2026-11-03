"""Print the AI-agent user guide of the installed version.

Written as UTF-8 whatever the console's encoding: the guide is not ASCII, and a
redirected stdout on a Japanese-locale Windows is cp932, which cannot hold it.
"""

import sys

from dnndenoiser.guides import read_guide

sys.stdout.buffer.write(read_guide().encode("utf-8"))
sys.stdout.flush()
