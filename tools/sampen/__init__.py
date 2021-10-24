"""SAMPLE ENTROPY

---------------------------------------------------------------------------
Last revised: 19 May 2016 (by joe@kinsa.us)

- Moved functions out of the init method and into separate modules
- Revised normalize data to more closely match the original C script;
  fixing some looping and mathematical bugs

Revised: 15 July 2015 (by joe@kinsa.us)

Modified version (ported C code to Python, modified to run as a callable
function) of original code:

sampen: calculate Sample Entropy
Copyright (C) 2002-2004 Doug Lake

This program is free software; you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation; either version 2 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.  You may also
view the agreement at http://www.fsf.org/copyleft/gpl.html.

You may contact the author via electronic mail (dlake@virginia.edu).
For updates to this software, please visit PhysioNet
(http://www.physionet.org/).
"""

from .sampen2 import sampen2
from .normalize_data import normalize_data
