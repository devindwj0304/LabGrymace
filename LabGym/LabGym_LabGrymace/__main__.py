'''
Copyright (C)
This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with this program. If not, see https://tldrlegal.com/license/gnu-general-public-license-v3-(gpl-3)#fulltext. 

For license issues, please contact:

Dr. Bing Ye
Life Sciences Institute
University of Michigan
210 Washtenaw Avenue, Room 5403
Ann Arbor, MI 48109-2216
USA

Email: bingye@umich.edu
'''

# !New Update from Wenjin -- this file differs from upstream LabGym 2.9.1:
# Removed the startup PyPI version check. It contacted pypi.org on every launch,
# which stalls behind a restricted network, and it told users to upgrade to a
# release that LabGrymace cannot read.
# Every change is marked with the same tag below.

# Configure the logging system.
# Log the load of this module (by the module loader, on first import).
#
# These statements are intentionally positioned before this module's
# other imports (against the guidance of PEP 8), to log the load of this
# module before other import statements are executed and produce their
# own log messages.
import inspect
import logging
logrecords = [logging.LogRecord(lineno=inspect.stack()[0].lineno,
    level=logging.DEBUG, exc_info=None, name=__name__, pathname=__file__,
    msg='%s', args=(f'loading {__file__}',),
    )]

import LabGym_LabGrymace.mylogging as mylogging
mylogging.config(logrecords)
mylogging.handle(logrecords)

logger = logging.getLogger(__name__)
logger.debug('%s: %r', '__name__', __name__)


# !New Update from Wenjin
# Local application/library specific imports.
from LabGym_LabGrymace import gui_main


# !New Update from Wenjin
def main():

	gui_main.main_window()



if __name__=='__main__':

	main()


