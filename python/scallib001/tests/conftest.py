#------------------------------------------------------------------------------------------------
#  Copyright (c) Shell Global Solutions International B.V. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#------------------------------------------------------------------------------------------------

import pytest

try:
   import pytest_benchmark
except:
   # pytest_benchmark does not installed, skip it
   @pytest.fixture
   def benchmark(*args, **kwargs):
       return None
