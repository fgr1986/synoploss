#  Copyright (c) 2019-2019     aiCTX AG 
# (Sadique Sheik, Qian Liu, Martino Sorbaro, Massimo Bortone).
#
#  This file is part of synoploss
#
#  synoploss is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  synoploss is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with synoploss.  If not, see <https://www.gnu.org/licenses/>.

from distutils.core import setup

setup(
    name="synoploss",
    version="0.1.0",
    packages=["synoploss"],
    license="GNU AGPLv3, Copyright (c) 2019 aiCTX AG",
    install_requires=['numpy', 'torch', 'tqdm', 'pytest', 'torchvision', 'tensorboard', 'sinabs'],
    long_description=open("README.md").read(),
)

