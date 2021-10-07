#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from .model import recognize, fine_tune

if __name__ == "__main__":
   parser = argparse.ArgumentParser("Run allosaurus")
   parser.add_argument('--model', required=True, type=str, help='the pretrained model id which you want to start with' )
   parser.add_argument('--path', required=True, type=str, help='file to transcribe or data dir for fine-tuning')
   parser.add_argument('--params', type=str, default='{}', help='JSON dict')
   parser.add_argument('--new_model', type=str, help='the new fine-tuned model ID')
   args = parser.parse_args()

   params = json.loads(args.params)
   if args.new_model:
      fine_tune(args.path, args.model, args.new_model, params)
   else:
      params["model"] = args.model
      print(recognize(args.path, params))
