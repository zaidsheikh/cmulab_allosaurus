#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from .model import recognize

if __name__ == "__main__":
   print(recognize(sys.argv[1]))
